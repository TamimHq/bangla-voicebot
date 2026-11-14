"""One phone-call session: audio in -> VAD -> ASR -> RAG -> LLM -> TTS -> audio out.

Latency design:
  1. VAD ends the turn after ~600 ms of silence (tunable).
  2. LLM output is streamed and cut into sentences; the very first chunk is
     cut even earlier, at the first comma, so speech can start sooner.
  3. Every chunk's TTS starts as soon as the chunk exists, in parallel with the
     LLM and with other chunks; audio is still sent in order.

Barge-in: if the caller starts speaking while the bot is talking, the current
turn is cancelled and the client is told to stop playback.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np

from .config import ROOT, settings
from .llm import TranscriptSplitter
from .rag import BM25Retriever
from .textnorm import redact_pii, split_first_clause, split_sentences
from .vad import StreamingVAD

log = logging.getLogger("voicebot")
SendJSON = Callable[[dict], Awaitable[None]]
TURN_LOG = ROOT / "logs" / "turns.jsonl"


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


class NoSpeech(Exception):
    """Direct mode: the model heard no clear words in the audio."""


def has_speakable(s: str) -> bool:
    """True if the chunk has at least one letter or digit (any script)."""
    return any(ch.isalnum() for ch in s)


async def first_item_timeout(agen, timeout: float):
    """Yield from an async generator, but give up if the FIRST item takes too long."""
    it = agen.__aiter__()
    try:
        first = await asyncio.wait_for(it.__anext__(), timeout=timeout)
    except StopAsyncIteration:
        return
    yield first
    async for item in it:
        yield item


class CallSession:
    def __init__(self, send: SendJSON, asr, llm, tts, retriever: BM25Retriever, fillers=None):
        self.send = send
        self.fillers = fillers
        self.asr, self.llm, self.tts, self.retriever = asr, llm, tts, retriever
        self.tts_slots = asyncio.Semaphore(max(1, settings.tts_concurrency))
        self.vad = StreamingVAD(
            threshold=settings.vad_threshold,
            min_silence_ms=settings.vad_min_silence_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            max_utterance_s=settings.vad_max_utterance_s,
        )
        self.history: list[tuple[str, str]] = []
        self.turn_task: asyncio.Task | None = None
        self.bot_speaking = False

    # ---------- inputs ----------
    async def on_audio(self, audio: np.ndarray) -> None:
        for ev in self.vad.feed(audio):
            if ev.kind == "speech_start":
                await self.send({"type": "vad", "state": "speech"})
                if self.bot_speaking or self._turn_running():
                    await self.interrupt()
            elif ev.kind == "utterance":
                await self.send({"type": "vad", "state": "silence"})
                self._start_turn(audio=ev.audio)

    async def on_text(self, text: str) -> None:
        """Typed input (for testing without a mic): skips VAD/ASR."""
        if self._turn_running():
            await self.interrupt()
        self._start_turn(text=text)

    async def on_playback_done(self) -> None:
        self.bot_speaking = False

    # ---------- turn control ----------
    def _turn_running(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()

    def _start_turn(self, audio: np.ndarray | None = None, text: str | None = None) -> None:
        self.turn_task = asyncio.create_task(self._turn(audio, text))

    async def interrupt(self) -> None:
        was_active = self._turn_running() or self.bot_speaking
        if self._turn_running():
            self.turn_task.cancel()
        self.bot_speaking = False
        if was_active:
            await self.send({"type": "interrupt"})

    async def close(self) -> None:
        if self._turn_running():
            self.turn_task.cancel()

    # ---------- one turn ----------
    async def _turn(self, audio: np.ndarray | None, text: str | None) -> None:
        t0 = time.perf_counter()  # = moment the caller stopped speaking
        m: dict = {"audio_s": round(len(audio) / settings.sample_rate, 2) if audio is not None else 0}
        reply_parts: list[str] = []
        tts_tasks: list[asyncio.Task] = []
        answer_started = False  # set once the first real audio chunk is sent

        async def maybe_filler():
            await asyncio.sleep(settings.filler_after_ms / 1000)
            clip = self.fillers.pick() if self.fillers else None
            if answer_started or clip is None:
                return
            text_, mime, data = clip
            m["filler_ms"] = _ms(t0)
            self.bot_speaking = True
            await self.send({"type": "audio", "seq": -1, "mime": mime, "text": text_,
                             "data": data, "filler": True})

        # Skip the filler for very short sounds (likely a cough, not a question).
        use_filler = settings.filler_after_ms > 0 and (
            audio is None or len(audio) / settings.sample_rate >= 0.6
        )
        filler_task = asyncio.create_task(maybe_filler()) if use_filler else None
        direct = (
            audio is not None and text is None
            and settings.pipeline_mode == "direct" and hasattr(self.llm, "stream_audio")
        )
        m["mode"] = "direct" if direct else "cascade"
        try:
            if direct:
                # 1+2+3 in ONE Gemini call: it hears the audio, sees the whole KB,
                # and streams "TRANSCRIPT: ..." then the answer.
                async def direct_tokens():
                    nonlocal text
                    split = TranscriptSplitter()

                    async def on_transcript(tr: str) -> None:
                        nonlocal text
                        m["asr_ms"] = _ms(t0)  # = when the caller's words became known
                        if not tr or len(tr) < 2:
                            raise NoSpeech()
                        text = tr
                        await self.send({"type": "transcript", "role": "user", "text": tr})
                        # Retrieval isn't needed to answer here; it's run only so the
                        # panel and logs still show which KB sections match.
                        t = time.perf_counter()
                        hits_ = self.retriever.search(tr, k=settings.rag_top_k)
                        m["retrieval_ms"] = _ms(t)
                        await self.send({"type": "sources", "items": [c.title for c, _ in hits_],
                                         "note": "direct mode: the model saw the whole KB"})

                    async for tok in self.llm.stream_audio(
                        audio, self.retriever.chunks, self.history, settings.sample_rate
                    ):
                        tr, out = split.feed(tok)
                        if tr is not None:
                            await on_transcript(tr)
                        if out:
                            yield out
                    tr, _ = split.finish()
                    if tr is not None:
                        await on_transcript(tr)

                token_source = direct_tokens()
            else:
                # 1) ASR
                if text is None:
                    # Fail fast instead of hanging while the client library retries.
                    text = await asyncio.wait_for(
                        self.asr.transcribe(audio, settings.sample_rate), timeout=settings.gemini_timeout_s
                    )
                    m["asr_ms"] = _ms(t0)
                if not text or len(text.strip()) < 2:
                    return  # noise / cough: ignore
                await self.send({"type": "transcript", "role": "user", "text": text})

                # 2) Retrieval
                t = time.perf_counter()
                hits = self.retriever.search(text, k=settings.rag_top_k)
                m["retrieval_ms"] = _ms(t)
                await self.send({"type": "sources", "items": [c.title for c, _ in hits]})
                token_source = self.llm.stream(text, hits, self.history)

            # 3) LLM -> chunk queue -> 4) TTS, running concurrently.
            # Each chunk's TTS starts the moment the chunk is ready (in parallel
            # with the LLM and with other chunks); the consumer sends them in order.
            queue: asyncio.Queue = asyncio.Queue()
            n_chunks = 0

            async def synth_timed(s: str):
                # Limit simultaneous TTS requests. Opening several Edge connections at
                # once from one IP makes each one slower, including the first chunk.
                # Chunks still queue up in order, so the next one is synthesized
                # while the previous one plays.
                async with self.tts_slots:
                    t = time.perf_counter()
                    is_cached = getattr(self.tts, "is_cached", None)
                    if is_cached and is_cached(s):
                        m["tts_cache_hits"] = m.get("tts_cache_hits", 0) + 1
                    for attempt in (1, 2):  # one retry: Edge occasionally drops a request
                        try:
                            out = await self.tts.synthesize(s)
                            return out, _ms(t)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            log.warning("TTS attempt %d failed for %r: %s: %s",
                                        attempt, s[:40], type(e).__name__, e)
                    return None, _ms(t)  # skip this chunk; don't end the whole reply

            def emit(s: str) -> None:
                nonlocal n_chunks
                if not has_speakable(s):
                    return  # e.g. a lone "।" or "*": Edge returns NoAudioReceived for these
                if n_chunks == 0:
                    m["llm_first_chunk_ms"] = _ms(t0)
                task = asyncio.create_task(synth_timed(s))
                tts_tasks.append(task)
                queue.put_nowait((s, task))
                n_chunks += 1

            async def producer():
                try:
                    await _produce()
                finally:
                    queue.put_nowait(None)  # always release the consumer, even on errors

            async def _produce():
                buf, first = "", True
                async for tok in first_item_timeout(token_source, settings.gemini_timeout_s):
                    if first:
                        m["llm_first_token_ms"] = _ms(t0)
                        first = False
                    buf += tok
                    await self.send({"type": "bot_delta", "text": tok})
                    sentences, buf = split_sentences(buf)
                    for s in sentences:
                        if n_chunks == 0:
                            # Gemini often streams a whole sentence in one piece. Still
                            # split the first one at a comma: a shorter first chunk is
                            # synthesized faster, so the caller hears audio sooner.
                            clause, rest = split_first_clause(s)
                            if clause and rest.strip():
                                emit(clause)
                                s = rest.strip()
                        emit(s)
                    if n_chunks == 0:
                        # Don't wait for a whole long first sentence: speak the
                        # first clause as soon as there's a natural comma break.
                        clause, buf = split_first_clause(buf)
                        if clause:
                            emit(clause)
                if buf.strip():
                    emit(buf.strip())

            async def consumer():
                nonlocal answer_started
                seq = 0
                while (item := await queue.get()) is not None:
                    s, task = item
                    out, tts_ms = await task
                    if out is None:
                        m["tts_failures"] = m.get("tts_failures", 0) + 1
                        reply_parts.append(s)  # text was shown; audio for it was skipped
                        continue
                    audio_bytes, mime = out
                    answer_started = True
                    if seq == 0:
                        m["tts_first_ms"] = tts_ms
                        m["first_audio_ms"] = _ms(t0)  # voice-to-voice latency
                    self.bot_speaking = True
                    await self.send(
                        {
                            "type": "audio",
                            "seq": seq,
                            "mime": mime,
                            "text": s,
                            "data": base64.b64encode(audio_bytes).decode(),
                        }
                    )
                    reply_parts.append(s)
                    seq += 1

            await asyncio.gather(producer(), consumer())
            m["total_ms"] = _ms(t0)
            if getattr(self.llm, "last_model", None):
                m["model"] = self.llm.last_model  # shows when a quota switch happened
            await self.send({"type": "bot_done"})
            await self.send({"type": "metrics", **m})
        except NoSpeech:
            pass  # direct mode heard no clear words: stay quiet, like the cascade path
        except asyncio.CancelledError:
            m["interrupted"] = True
            for task in tts_tasks:
                task.cancel()
            raise
        except Exception as e:  # keep the call alive on backend errors
            log.exception("turn failed")
            m["error"] = type(e).__name__
            await self.send({"type": "error", "message": f"{type(e).__name__}: {e}"})
            await self._speak_fallback()
        finally:
            if filler_task:
                filler_task.cancel()
            reply = " ".join(reply_parts)
            if text and reply:
                self.history += [("user", text), ("model", reply)]
            self._log_turn(text, reply, m)

    async def _speak_fallback(self) -> None:
        """Never leave the caller in silence: say a short apology."""
        text = "দুঃখিত, একটু সমস্যা হয়েছে। আবার বলবেন?"
        try:
            audio_bytes, mime = await self.tts.synthesize(text)
            self.bot_speaking = True
            await self.send({"type": "bot_delta", "text": text})
            await self.send({"type": "audio", "seq": 0, "mime": mime, "text": text,
                             "data": base64.b64encode(audio_bytes).decode()})
            await self.send({"type": "bot_done"})
        except Exception:
            log.exception("fallback TTS failed")

    def _log_turn(self, user: str | None, bot: str, m: dict) -> None:
        if not user:
            return
        red = redact_pii if settings.redact_pii_in_logs else (lambda s: s)
        TURN_LOG.parent.mkdir(exist_ok=True)
        rec = {"ts": time.time(), "user": red(user), "bot": red(bot), **m}
        with TURN_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
