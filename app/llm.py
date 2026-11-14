"""LLM reasoning with Gemini, streamed token-by-token.

Two entry points:
  * stream(text, hits, history)        - cascade mode: ASR already produced text,
                                          retrieval picked the top-k KB sections.
  * stream_audio(audio, kb, history)   - direct mode: Gemini hears the audio itself
                                          and gets the whole (small) KB. One network
                                          round trip instead of two. It streams
                                          "TRANSCRIPT: ..." on the first line, then
                                          the answer.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from .audio import float_to_wav_bytes
from .config import settings
from .gemini import (
    QuotaExhausted,
    finish_reason,
    is_quota_error,
    make_config,
    park,
    retry_delay_s,
    thinking_unsupported,
    usable_models,
)
from .rag import Chunk
from .textnorm import is_mostly_english

log = logging.getLogger("voicebot")


class EmptyReplyError(RuntimeError):
    pass


SYSTEM_PROMPT = """তুমি "ডেমোটেল" (DemoTel) নামের একটি কাল্পনিক মোবাইল অপারেটরের ভয়েস কাস্টমার কেয়ার এজেন্ট।
তোমার উত্তর টেক্সট-টু-স্পিচ দিয়ে ফোনে পড়ে শোনানো হবে। তাই:
- খুব ছোট উত্তর দাও: সর্বোচ্চ ২-৩টি বাক্য।
- কোনো markdown, বুলেট, ইমোজি বা লিংক দেবে না।
- কলার যে ভাষায় বলেছে সেই ভাষায় উত্তর দাও (বাংলা, ইংরেজি বা দুটোর মিশ্রণ)।
- দাম ও সংখ্যা পরিষ্কারভাবে বলো, যেমন "২৯৯ টাকা", "৩০ দিন"।
- শুধু নিচের CONTEXT-এর তথ্য ব্যবহার করবে। CONTEXT-এ উত্তর না থাকলে অনুমান করবে না;
  বলবে যে তুমি নিশ্চিত নও এবং একজন মানুষ এজেন্টের সাথে যুক্ত করে দিতে চাইবে।
- কখনো পিন, পাসওয়ার্ড বা ওটিপি চাইবে না।"""

# Extra rules for direct (audio-in) mode. The fixed first line lets the server
# show the caller's words, log them, and keep chat history, before the answer.
DIRECT_FORMAT = """
এই বার্তায় কলারের কণ্ঠস্বরের অডিও আছে। ঠিক এই ফরম্যাটে উত্তর দাও:
প্রথম লাইন: TRANSCRIPT: <কলার যা বলেছে হুবহু; বাংলা শব্দ বাংলা লিপিতে, ইংরেজি শব্দ ইংরেজিতে; অনুবাদ করবে না>
তারপর নতুন লাইনে শুধু কলারকে দেওয়া উত্তর।
অডিওতে পরিষ্কার কোনো কথা না থাকলে শুধু লিখবে: TRANSCRIPT:
SCRIPT RULE for the TRANSCRIPT line: Bangla words MUST be in Bangla script, never romanized
("নতুন সিম", not "Notun SIM"); English words stay in English letters; never Devanagari.
LANGUAGE RULE: if the caller spoke English, the answer MUST be in English only (say prices like "49 taka").
If the caller spoke Bangla or mixed Bangla-English, answer in Bangla."""

# Added to the user message in cascade mode, where the caller's words are known
# before the LLM call. Written in English so the model can't miss it.
ENGLISH_TURN = "LANGUAGE: The caller spoke English. Reply in English only (e.g. \"49 taka\", \"3 days\")."


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    if not hits:
        return "(কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি)"
    return "\n\n".join(f"[{c.title}]\n{c.text}" for c, _ in hits)


async def _first_text(gen):
    """Pull chunks until one has text; returns it. Raises EmptyReplyError if none."""
    last = None
    async for chunk in gen:
        last = chunk
        if chunk.text:
            return chunk
    raise EmptyReplyError(f"Gemini returned no text (finish_reason={finish_reason(last)})")


class GeminiLLM:
    def __init__(self):
        from google.genai import types

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        from .gemini import make_client

        self.client = make_client()
        self.types = types

    def _history(self, history: list[tuple[str, str]]) -> list:
        t = self.types
        # last 3 turns is enough for a phone call
        return [t.Content(role=role, parts=[t.Part(text=text)]) for role, text in history[-6:]]

    async def _stream(self, contents: list, system: str) -> AsyncIterator[str]:
        """Stream the answer, racing models until one starts talking.

        * Start the first usable model.
        * If it hits its quota (429), park it and start the next model right away.
        * If it is silent for HEDGE_AFTER_S (Gemini sometimes just sits on a
          request), ALSO start the next model: a "hedged request". Whichever sends
          text first wins; the other request is cancelled.
        Once any text has been sent we never switch: a half-spoken answer can't restart.
        """
        self.last_model = None
        queue = list(usable_models(settings.llm_model))
        racing: dict[asyncio.Task, tuple[str, AsyncIterator]] = {}
        last_err: Exception | None = None
        hedge_after = settings.hedge_after_s if settings.hedge_after_s > 0 else None

        def start_next() -> bool:
            if not queue:
                return False
            model = queue.pop(0)
            gen = self._stream_one(model, contents, system)
            racing[asyncio.create_task(_first_text(gen))] = (model, gen)
            return True

        async def stop(task: asyncio.Task) -> None:
            _, gen = racing.pop(task)
            task.cancel()
            try:
                await gen.aclose()
            except Exception:
                pass

        start_next()
        try:
            while racing:
                can_hedge = bool(queue) and hedge_after is not None
                done, _ = await asyncio.wait(
                    racing, timeout=hedge_after if can_hedge else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:  # nobody has spoken yet: hedge with the next model
                    log.warning("No reply from %s after %.1fs; also trying %s",
                                ", ".join(m for m, _ in racing.values()), hedge_after, queue[0])
                    start_next()
                    continue
                for task in done:
                    model, gen = racing[task]
                    err = task.exception()
                    if err is None:
                        first = task.result()
                        racing.pop(task)
                        for other in list(racing):
                            await stop(other)
                        self.last_model = model
                        yield first.text
                        async for chunk in gen:
                            if chunk.text:
                                yield chunk.text
                        return
                    racing.pop(task)
                    last_err = err
                    if is_quota_error(err):
                        park(model, err)
                    if not racing:
                        start_next()  # replace the failed model, if any are left
        finally:
            for task in list(racing):
                await stop(task)
        if last_err is None or is_quota_error(last_err):
            raise QuotaExhausted(retry_delay_s(last_err or Exception()))
        raise last_err

    async def _stream_one(self, model: str, contents: list, system: str):
        t = self.types

        async def open_stream():
            return await self.client.aio.models.generate_content_stream(
                model=model, contents=contents,
                config=make_config(t, system_instruction=system, temperature=0.3),
            )

        try:
            stream = await open_stream()
        except Exception as e:
            if not thinking_unsupported(e):
                raise
            stream = await open_stream()
        async for chunk in stream:
            yield chunk

    async def stream(
        self, user_text: str, hits: list[tuple[Chunk, float]], history: list[tuple[str, str]]
    ) -> AsyncIterator[str]:
        t = self.types
        text = f"CONTEXT:\n{build_context(hits)}\n\nকলার বলছে: {user_text}"
        if is_mostly_english(user_text):
            text += f"\n\n{ENGLISH_TURN}"
        contents = self._history(history) + [t.Content(role="user", parts=[t.Part(text=text)])]
        async for tok in self._stream(contents, SYSTEM_PROMPT):
            yield tok

    async def stream_audio(
        self, audio: np.ndarray, kb: list[Chunk], history: list[tuple[str, str]], sr: int = 16000
    ) -> AsyncIterator[str]:
        t = self.types
        context = "\n\n".join(f"[{c.title}]\n{c.text}" for c in kb)
        contents = self._history(history) + [
            t.Content(role="user", parts=[
                t.Part(text=f"CONTEXT:\n{context}"),
                t.Part.from_bytes(data=float_to_wav_bytes(audio, sr), mime_type="audio/wav"),
            ])
        ]
        async for tok in self._stream(contents, SYSTEM_PROMPT + "\n" + DIRECT_FORMAT):
            yield tok


class TranscriptSplitter:
    """Splits the direct-mode stream into the transcript line and the answer.

    Feed tokens in; once the first newline arrives it reports the transcript and
    passes everything after it through as answer text. If the model forgets the
    "TRANSCRIPT:" prefix, the whole output is treated as the answer.
    """

    PREFIX = "TRANSCRIPT:"

    def __init__(self) -> None:
        self.buf = ""
        self.transcript: str | None = None

    def feed(self, tok: str) -> tuple[str | None, str]:
        """Returns (transcript if it just became known, answer text to pass on)."""
        if self.transcript is not None:
            return None, tok
        self.buf += tok
        stripped = self.buf.lstrip()
        if not self.PREFIX.startswith(stripped[: len(self.PREFIX)]):
            self.transcript = ""  # no prefix: everything is the answer
            out, self.buf = self.buf, ""
            return "", out
        if "\n" not in stripped:
            return None, ""
        line, rest = stripped.split("\n", 1)
        self.transcript = line[len(self.PREFIX):].strip()
        self.buf = ""
        return self.transcript, rest.lstrip("\n")

    def finish(self) -> tuple[str | None, str]:
        """Call at end of stream: handles 'TRANSCRIPT: ...' with no newline/answer."""
        if self.transcript is not None:
            return None, ""
        stripped = self.buf.strip()
        self.transcript = stripped[len(self.PREFIX):].strip() if stripped.startswith(self.PREFIX) else ""
        return self.transcript, ""
