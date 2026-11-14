"""Offline tests: no API keys or network needed.

Run:  pytest -q
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from app.audio import pcm16_bytes_to_float, telephone_band
from app.config import settings
from app.pipeline import CallSession
from app.rag import BM25Retriever, load_kb
from app.textnorm import normalize_for_wer, redact_pii, split_sentences
from app.vad import StreamingVAD


# ---------------- text normalization ----------------
def test_nukta_forms_are_equal():
    precomposed = "সময়"            # য় as U+09DF
    decomposed = "সময়"        # য + nukta
    assert normalize_for_wer(precomposed) == normalize_for_wer(decomposed)


def test_digits_and_danda():
    assert normalize_for_wer("দাম ২৯৯ টাকা।") == "দাম 299 টাকা"


def test_sentence_split_streaming():
    s, rest = split_sentences("১০ জিবি প্যাকের দাম ২৯৯ টাকা। মেয়াদ ৩০")
    assert s == ["১০ জিবি প্যাকের দাম ২৯৯ টাকা।"]
    assert rest.strip() == "মেয়াদ ৩০"


def test_sentence_split_keeps_decimals():
    s, rest = split_sentences("দাম 1.5 জিবি। ")
    assert s == ["দাম 1.5 জিবি।"]


def test_pii_redaction():
    out = redact_pii("আমার নম্বর 01712345678, মেইল a@b.com")
    assert "<PHONE>" in out and "<EMAIL>" in out and "0171" not in out


# ---------------- audio ----------------
def test_telephone_band_removes_high_freq():
    sr = 16000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 6000 * t).astype(np.float32)  # 6 kHz > 4 kHz Nyquist of 8 kHz
    out = telephone_band(tone, sr)[2000:-2000]  # skip filter edge ringing
    assert np.sqrt(np.mean(out**2)) < 0.01      # input RMS is ~0.707


def test_pcm16_roundtrip():
    x = (np.array([0, 16384, -16384], dtype="<i2")).tobytes()
    assert np.allclose(pcm16_bytes_to_float(x), [0, 0.5, -0.5])


# ---------------- VAD ----------------
def test_vad_ignores_silence_and_noise():
    vad = StreamingVAD()
    rng = np.random.default_rng(0)
    events = vad.feed(np.zeros(16000, dtype=np.float32))
    events += vad.feed((rng.normal(0, 0.01, 32000)).astype(np.float32))
    assert [e.kind for e in events] == []


def _loud_is_speech(vad):
    vad._prob = lambda frame: 0.9 if np.abs(frame).mean() > 0.1 else 0.05
    return vad


def test_vad_state_machine_emits_one_utterance():
    vad = _loud_is_speech(StreamingVAD(min_silence_ms=600, min_speech_ms=250))
    speech = np.full(16000, 0.5, dtype=np.float32)   # 1 s "speech"
    silence = np.zeros(16000, dtype=np.float32)      # 1 s silence ends the turn
    events = vad.feed(silence[:4000]) + vad.feed(speech) + vad.feed(silence)
    assert [e.kind for e in events] == ["speech_start", "utterance"]
    dur = len(events[1].audio) / 16000
    assert 1.0 <= dur <= 1.5   # speech + pre-roll + short tail, trailing silence trimmed


def test_vad_drops_short_blips():
    vad = _loud_is_speech(StreamingVAD(min_speech_ms=250))
    blip = np.full(1600, 0.5, dtype=np.float32)      # 100 ms cough
    events = vad.feed(blip) + vad.feed(np.zeros(16000, dtype=np.float32))
    assert [e.kind for e in events] == ["speech_start"]  # no utterance sent to ASR


# ---------------- retrieval ----------------
def test_retrieval_handles_suffix_variants():
    r = BM25Retriever(load_kb(settings.kb_path))
    top = r.search("আমার সিমটা হারিয়ে গেছে", k=1)[0][0].title
    assert "সিম হারানো" in top


# ---------------- Gemini ASR request shape (mocked client) ----------------
def test_gemini_asr_sends_wav_and_cleans_output(monkeypatch):
    from app import asr as asr_mod

    monkeypatch.setattr(settings, "gemini_api_key", "dummy")
    a = asr_mod.GeminiASR()
    seen = {}

    class Resp:
        text = "  আমার ব্যালেন্স কতয়  "

    async def fake_generate(model, contents, config):
        seen["model"], seen["contents"] = model, contents
        return Resp()

    monkeypatch.setattr(a.client.aio.models, "generate_content", fake_generate)
    out = asyncio.run(a.transcribe(np.zeros(16000, dtype=np.float32)))
    part = seen["contents"][0]
    assert part.inline_data.mime_type == "audio/wav"
    assert part.inline_data.data[:4] == b"RIFF"
    assert out == "আমার ব্যালেন্স কতয়"   # stripped + nukta normalized


# ---------------- pipeline with fake backends ----------------
class FakeASR:
    async def transcribe(self, audio, sr=16000):
        return "ব্যালেন্স কিভাবে দেখব"


class FakeLLM:
    async def stream(self, text, hits, history):
        for tok in ["*১২১# ", "ডায়াল করুন।", " এটা ফ্রি।"]:
            await asyncio.sleep(0)
            yield tok


class FakeTTS:
    async def synthesize(self, text):
        return b"ID3fake", "audio/mpeg"


def test_pipeline_text_turn():
    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        retr = BM25Retriever(load_kb(settings.kb_path))
        s = CallSession(send, FakeASR(), FakeLLM(), FakeTTS(), retr)
        await s.on_text("ব্যালেন্স কিভাবে দেখব")
        await s.turn_task
        return sent

    sent = asyncio.run(run())
    kinds = [m["type"] for m in sent]
    assert kinds[0] == "transcript" and "bot_done" in kinds
    audio = [m for m in sent if m["type"] == "audio"]
    assert [a["text"] for a in audio] == ["*১২১# ডায়াল করুন।", "এটা ফ্রি।"]
    metrics = next(m for m in sent if m["type"] == "metrics")
    assert {"retrieval_ms", "llm_first_token_ms", "first_audio_ms", "total_ms"} <= metrics.keys()
    sources = next(m for m in sent if m["type"] == "sources")
    assert "ব্যালেন্স চেক" in sources["items"][0]


def test_gemini_config_limits_thinking():
    from google.genai import types

    from app.gemini import make_config

    cfg = make_config(types, temperature=0.3)
    assert cfg.thinking_config.thinking_level.value == "MINIMAL"
    assert cfg.max_output_tokens >= 1024          # room for answer even if it thinks a bit
    assert cfg.automatic_function_calling.disable is True


def test_empty_llm_reply_speaks_fallback():
    from app.llm import EmptyReplyError

    class EmptyLLM:
        async def stream(self, text, hits, history):
            raise EmptyReplyError("no text")
            yield  # pragma: no cover  (makes this an async generator)

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, FakeASR(), EmptyLLM(), FakeTTS(), BM25Retriever(load_kb(settings.kb_path)))
        await s.on_text("হ্যালো")
        await s.turn_task
        return sent

    sent = asyncio.run(run())
    kinds = [m["type"] for m in sent]
    assert "error" in kinds and "audio" in kinds   # caller hears an apology, not silence


def test_no_interrupt_message_when_idle():
    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, FakeASR(), FakeLLM(), FakeTTS(), BM25Retriever(load_kb(settings.kb_path)))
        await s.interrupt()
        return sent

    assert asyncio.run(run()) == []


def test_barge_in_cancels_turn():
    class SlowLLM:
        async def stream(self, text, hits, history):
            yield "প্রথম বাক্য।"
            await asyncio.sleep(10)
            yield "কখনো আসবে না।"

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, FakeASR(), SlowLLM(), FakeTTS(), BM25Retriever(load_kb(settings.kb_path)))
        await s.on_text("হ্যালো")
        await asyncio.sleep(0.05)
        await s.interrupt()
        with pytest.raises(asyncio.CancelledError):
            await s.turn_task
        return sent

    sent = asyncio.run(run())
    assert sent[-1]["type"] == "interrupt"
    assert not any(m.get("text") == "কখনো আসবে না।" for m in sent)


# ---------------- latency features ----------------
def test_first_clause_split():
    from app.textnorm import split_first_clause

    clause, rest = split_first_clause("আমাদের সবচেয়ে সস্তা প্যাক হলো ১ জিবি ইন্টারনেট, যা ৪৯ টা")
    assert clause == "আমাদের সবচেয়ে সস্তা প্যাক হলো ১ জিবি ইন্টারনেট,"
    assert rest.strip() == "যা ৪৯ টা"
    assert split_first_clause("হ্যাঁ, ঠিক")[0] is None          # too short to be worth it
    assert split_first_clause("দাম মাত্র ১,০০০ টাকা এবং আরও কিছু")[0] is None  # number comma
    assert split_first_clause("সর্বোচ্চ রিচার্জ এক বারে ১,")[0] is None   # number may continue


def test_parallel_tts_keeps_order_and_splits_first_clause():
    class CommaLLM:
        async def stream(self, text, hits, history):
            for tok in ["আমাদের সবচেয়ে সস্তা প্যাক হলো ১ জিবি, ", "যা ৪৯ টাকায় পাওয়া যায়।", " ধন্যবাদ।"]:
                await asyncio.sleep(0)
                yield tok

    class SlowFirstTTS:
        """First chunk is slowest: if sending weren't ordered, chunk 2 would arrive first."""
        async def synthesize(self, text):
            await asyncio.sleep(0.05 if "সস্তা" in text else 0.0)
            return b"x", "audio/mpeg"

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, FakeASR(), CommaLLM(), SlowFirstTTS(), BM25Retriever(load_kb(settings.kb_path)))
        await s.on_text("সস্তা প্যাক")
        await s.turn_task
        return sent

    sent = asyncio.run(run())
    audio = [m["text"] for m in sent if m["type"] == "audio"]
    assert audio == ["আমাদের সবচেয়ে সস্তা প্যাক হলো ১ জিবি,", "যা ৪৯ টাকায় পাওয়া যায়।", "ধন্যবাদ।"]
    metrics = next(m for m in sent if m["type"] == "metrics")
    assert metrics["tts_first_ms"] >= 40


def test_tts_runs_one_at_a_time_by_default():
    class CountingTTS:
        def __init__(self):
            self.active = self.peak = 0

        async def synthesize(self, text):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return b"x", "audio/mpeg"

    class ManySentencesLLM:
        async def stream(self, text, hits, history):
            yield "প্রথম বাক্য। দ্বিতীয় বাক্য। তৃতীয় বাক্য। চতুর্থ বাক্য।"

    tts = CountingTTS()

    async def run():
        async def send(m):
            pass

        s = CallSession(send, FakeASR(), ManySentencesLLM(), tts, BM25Retriever(load_kb(settings.kb_path)))
        await s.on_text("হ্যালো")
        await s.turn_task

    asyncio.run(run())
    assert tts.peak == 1


def test_first_sentence_split_even_when_streamed_whole():
    class WholeSentenceLLM:
        async def stream(self, text, hits, history):
            yield "ডেমোটেলের সবচেয়ে সস্তা প্যাক হলো ১ জিবি ৪৯ টাকা, যার মেয়াদ ৩ দিন। এটি কিনতে ডায়াল করুন।"

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, FakeASR(), WholeSentenceLLM(), FakeTTS(), BM25Retriever(load_kb(settings.kb_path)))
        await s.on_text("সস্তা প্যাক")
        await s.turn_task
        return sent

    audio = [m["text"] for m in asyncio.run(run()) if m["type"] == "audio"]
    assert audio == ["ডেমোটেলের সবচেয়ে সস্তা প্যাক হলো ১ জিবি ৪৯ টাকা,", "যার মেয়াদ ৩ দিন।", "এটি কিনতে ডায়াল করুন।"]


# ---------------- robustness ----------------
def _session(send, llm, tts=None, asr=None):
    return CallSession(send, asr or FakeASR(), llm, tts or FakeTTS(), BM25Retriever(load_kb(settings.kb_path)))


def test_failed_tts_chunk_is_skipped_not_fatal():
    class FlakyTTS:
        async def synthesize(self, text):
            if "দ্বিতীয়" in text:
                raise RuntimeError("NoAudioReceived")
            return b"x", "audio/mpeg"

    class LLM3:
        async def stream(self, text, hits, history):
            yield "প্রথম বাক্য। দ্বিতীয় বাক্য। তৃতীয় বাক্য।"

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = _session(send, LLM3(), FlakyTTS())
        await s.on_text("হ্যালো")
        await s.turn_task
        return sent

    sent = asyncio.run(run())
    assert [m["text"] for m in sent if m["type"] == "audio"] == ["প্রথম বাক্য।", "তৃতীয় বাক্য।"]
    assert not any(m["type"] == "error" for m in sent)
    assert next(m for m in sent if m["type"] == "metrics")["tts_failures"] == 1


def test_punctuation_only_chunk_is_not_sent_to_tts():
    seen = []

    class RecTTS:
        async def synthesize(self, text):
            seen.append(text)
            return b"x", "audio/mpeg"

    class PunctLLM:
        async def stream(self, text, hits, history):
            yield "স্টার ১২১ হ্যাশ ডায়াল করুন।"
            yield " ।"

    async def run():
        async def send(m):
            pass

        s = _session(send, PunctLLM(), RecTTS())
        await s.on_text("ব্যালেন্স")
        await s.turn_task

    asyncio.run(run())
    assert seen == ["স্টার ১২১ হ্যাশ ডায়াল করুন।"]


def test_slow_llm_times_out_and_apologizes(monkeypatch):
    monkeypatch.setattr(settings, "gemini_timeout_s", 0.05)

    class HangingLLM:
        async def stream(self, text, hits, history):
            await asyncio.sleep(5)
            yield "never"

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = _session(send, HangingLLM())
        await s.on_text("হ্যালো")
        await asyncio.wait_for(s.turn_task, 2)
        return sent

    sent = asyncio.run(run())
    assert any(m["type"] == "error" and "Timeout" in m["message"] for m in sent)
    assert any(m["type"] == "audio" for m in sent)  # spoken apology


def test_gemini_client_has_short_retry(monkeypatch):
    from app.gemini import make_client

    monkeypatch.setattr(settings, "gemini_api_key", "dummy")
    c = make_client()
    r = c._api_client._http_options.retry_options
    assert r.attempts == 2 and r.max_delay <= 2


# ---------------- filler phrases ----------------
def _filler_bank():
    from app.filler import FillerBank

    from app.filler import PHRASES

    bank = FillerBank()  # filled directly; warm() is async and tests call this inside a loop
    bank.clips = [(p, "audio/mpeg", "eA==") for p in PHRASES]
    return bank


def test_filler_plays_when_answer_is_slow(monkeypatch):
    monkeypatch.setattr(settings, "filler_after_ms", 20)

    class SlowLLM:
        async def stream(self, text, hits, history):
            await asyncio.sleep(0.1)
            yield "উত্তর এখানে।"

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, FakeASR(), SlowLLM(), FakeTTS(), BM25Retriever(load_kb(settings.kb_path)), _filler_bank())
        await s.on_text("হ্যালো")
        await s.turn_task
        return sent

    sent = asyncio.run(run())
    audio = [m for m in sent if m["type"] == "audio"]
    assert audio[0].get("filler") and audio[0]["text"] == "জি, একটু দেখছি।"
    assert audio[1]["text"] == "উত্তর এখানে।"
    metrics = next(m for m in sent if m["type"] == "metrics")
    assert metrics["filler_ms"] < metrics["first_audio_ms"]


def test_no_filler_when_answer_is_fast(monkeypatch):
    monkeypatch.setattr(settings, "filler_after_ms", 500)

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, FakeASR(), FakeLLM(), FakeTTS(), BM25Retriever(load_kb(settings.kb_path)), _filler_bank())
        await s.on_text("হ্যালো")
        await s.turn_task
        await asyncio.sleep(0.6)  # past the filler deadline
        return sent

    sent = asyncio.run(run())
    assert not any(m.get("filler") for m in sent)
    assert "filler_ms" not in next(m for m in sent if m["type"] == "metrics")


def test_filler_skipped_for_short_sounds(monkeypatch):
    monkeypatch.setattr(settings, "filler_after_ms", 10)

    class EmptyASR:
        async def transcribe(self, audio, sr=16000):
            await asyncio.sleep(0.1)
            return ""

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = CallSession(send, EmptyASR(), FakeLLM(), FakeTTS(), BM25Retriever(load_kb(settings.kb_path)), _filler_bank())
        s._start_turn(audio=np.zeros(4000, dtype=np.float32))  # 0.25 s cough
        await s.turn_task
        return sent

    assert not any(m.get("filler") for m in asyncio.run(run()))


def test_filler_rotates():
    bank = _filler_bank()
    assert [bank.pick()[0] for _ in range(4)] == ["জি, একটু দেখছি।", "আচ্ছা, এক মুহূর্ত।", "ঠিক আছে, দেখে বলছি।", "জি, একটু দেখছি।"]


def test_filler_warm_uses_tts():
    from app.filler import FillerBank, PHRASES

    bank = FillerBank()
    asyncio.run(bank.warm(FakeTTS()))
    assert len(bank.clips) == len(PHRASES)


# ---------------- direct mode (one Gemini call hears the audio) ----------------
def test_transcript_splitter_streaming():
    from app.llm import TranscriptSplitter

    sp = TranscriptSplitter()
    outs = [sp.feed(t) for t in ["TRANS", "CRIPT: আমার ব্যালেন্স", " কত?\nআপনার ব্যালেন্স ", "জানতে *১২১# ডায়াল করুন।"]]
    assert outs[0] == (None, "") and outs[1] == (None, "")
    assert outs[2] == ("আমার ব্যালেন্স কত?", "আপনার ব্যালেন্স ")
    assert outs[3] == (None, "জানতে *১২১# ডায়াল করুন।")


def test_transcript_splitter_without_prefix_is_all_answer():
    from app.llm import TranscriptSplitter

    sp = TranscriptSplitter()
    assert sp.feed("আপনার ব্যালেন্স জানতে") == ("", "আপনার ব্যালেন্স জানতে")


def test_transcript_splitter_empty_audio():
    from app.llm import TranscriptSplitter

    sp = TranscriptSplitter()
    assert sp.feed("TRANSCRIPT:") == (None, "")
    assert sp.finish() == ("", "")


class DirectLLM(FakeLLM):
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = 0

    async def stream_audio(self, audio, kb, history, sr=16000):
        self.calls += 1
        assert len(kb) == 11  # whole knowledge base, not top-k
        for tok in self.tokens:
            await asyncio.sleep(0)
            yield tok


def _run_audio_turn(llm, asr=None, monkeypatch=None):
    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = _session(send, llm, asr=asr)
        s._start_turn(audio=np.zeros(16000, dtype=np.float32))
        await s.turn_task
        return sent, s

    return asyncio.run(run())


def test_direct_mode_skips_asr_and_streams_answer(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_mode", "direct")

    class MustNotRunASR:
        async def transcribe(self, audio, sr=16000):
            raise AssertionError("ASR should not be called in direct mode")

    llm = DirectLLM(["TRANSCRIPT: আমার ব্যালেন্স কত?\n", "আপনার ব্যালেন্স জানতে *১২১# ডায়াল করুন।"])
    sent, s = _run_audio_turn(llm, asr=MustNotRunASR())
    kinds = [m["type"] for m in sent]
    assert kinds.index("transcript") < kinds.index("bot_delta")
    assert next(m for m in sent if m["type"] == "transcript")["text"] == "আমার ব্যালেন্স কত?"
    assert [m["text"] for m in sent if m["type"] == "audio"] == ["আপনার ব্যালেন্স জানতে *১২১# ডায়াল করুন।"]
    # the transcript line itself is never spoken or shown as the bot's words
    assert "TRANSCRIPT" not in "".join(m["text"] for m in sent if m["type"] == "bot_delta")
    metrics = next(m for m in sent if m["type"] == "metrics")
    assert metrics["mode"] == "direct" and "asr_ms" in metrics
    assert "ব্যালেন্স চেক" in next(m for m in sent if m["type"] == "sources")["items"][0]
    assert s.history[0] == ("user", "আমার ব্যালেন্স কত?")


def test_direct_mode_no_speech_stays_quiet(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_mode", "direct")
    monkeypatch.setattr(settings, "filler_after_ms", 0)
    sent, s = _run_audio_turn(DirectLLM(["TRANSCRIPT:"]))
    assert not any(m["type"] in ("audio", "error", "transcript") for m in sent)
    assert s.history == []


def test_cascade_mode_still_uses_asr(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_mode", "cascade")
    llm = DirectLLM(["never used"])
    sent, _ = _run_audio_turn(llm)
    assert llm.calls == 0
    assert next(m for m in sent if m["type"] == "metrics")["mode"] == "cascade"
    assert next(m for m in sent if m["type"] == "transcript")["text"] == "ব্যালেন্স কিভাবে দেখব"


def test_producer_error_does_not_leave_consumer_hanging(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_mode", "direct")

    class Boom(DirectLLM):
        async def stream_audio(self, audio, kb, history, sr=16000):
            yield "TRANSCRIPT: হ্যালো\n"
            raise RuntimeError("network down")

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = _session(send, Boom([]))
        s._start_turn(audio=np.zeros(16000, dtype=np.float32))
        await asyncio.wait_for(s.turn_task, 2)  # would time out if consumer hung
        return sent

    sent = asyncio.run(run())
    assert any(m["type"] == "error" for m in sent)


def test_gemini_stream_audio_request_shape(monkeypatch):
    from app.llm import GeminiLLM

    monkeypatch.setattr(settings, "gemini_api_key", "dummy")
    llm = GeminiLLM()
    seen = {}

    class Chunk_:
        def __init__(self, text):
            self.text = text

    async def fake_stream(model, contents, config):
        seen["contents"], seen["config"] = contents, config

        async def gen():
            yield Chunk_("TRANSCRIPT: হ্যালো\n")
            yield Chunk_("জি বলুন।")

        return gen()

    monkeypatch.setattr(llm.client.aio.models, "generate_content_stream", fake_stream)
    kb = load_kb(settings.kb_path)

    async def run():
        return [t async for t in llm.stream_audio(np.zeros(8000, dtype=np.float32), kb, [])]

    toks = asyncio.run(run())
    assert toks == ["TRANSCRIPT: হ্যালো\n", "জি বলুন।"]
    parts = seen["contents"][-1].parts
    assert all(c.title in parts[0].text for c in kb)           # whole KB in context
    assert parts[1].inline_data.mime_type == "audio/wav"
    assert "TRANSCRIPT:" in seen["config"].system_instruction


# ---------------- free-tier quota: switch model on 429 ----------------
class _Quota429(Exception):
    code = 429

    def __str__(self):
        return "429 RESOURCE_EXHAUSTED ... Please retry in 41.29s."


def _fake_gemini(monkeypatch, quota_models):
    """Patch GeminiLLM so listed models raise 429 and others stream an answer."""
    from app import gemini
    from app.llm import GeminiLLM

    monkeypatch.setattr(settings, "gemini_api_key", "dummy")
    monkeypatch.setattr(settings, "llm_model", "model-a")
    monkeypatch.setattr(settings, "llm_fallback_models", ["model-b"])
    gemini._parked_until.clear()
    llm = GeminiLLM()
    calls = []

    class C:
        def __init__(self, text):
            self.text = text

    async def fake_one(model, contents, system):
        calls.append(model)
        if model in quota_models:
            raise _Quota429()
        yield C("উত্তর।")

    monkeypatch.setattr(llm, "_stream_one", fake_one)
    return llm, calls


def _collect(llm):
    async def run():
        return [t async for t in llm.stream("প্রশ্ন", [], [])]
    return asyncio.run(run())


def test_quota_switches_to_fallback_and_parks(monkeypatch):
    from app import gemini

    llm, calls = _fake_gemini(monkeypatch, quota_models={"model-a"})
    assert _collect(llm) == ["উত্তর।"]
    assert calls == ["model-a", "model-b"] and llm.last_model == "model-b"
    # next turn goes straight to model-b: model-a is parked for ~41 s
    assert _collect(llm) == ["উত্তর।"] and calls[-1] == "model-b" and calls.count("model-a") == 1
    assert 40 < gemini._parked_until["model-a"] - __import__("time").monotonic() <= 41.3


def test_quota_on_all_models_raises_clear_error(monkeypatch):
    from app.gemini import QuotaExhausted

    llm, _ = _fake_gemini(monkeypatch, quota_models={"model-a", "model-b"})
    with pytest.raises(QuotaExhausted):
        _collect(llm)
    with pytest.raises(QuotaExhausted, match="try again in"):  # both parked: fails instantly
        _collect(llm)


def test_retry_skips_429():
    from app.gemini import make_client

    settings.gemini_api_key = settings.gemini_api_key or "dummy"
    r = make_client()._api_client._http_options.retry_options
    assert 429 not in r.http_status_codes and 503 in r.http_status_codes


# ---------------- hedged requests: a silent model doesn't stall the call ----------------
def _hedge_llm(monkeypatch, delays, hedge_after=0.05):
    """delays: model -> seconds before its first chunk (None = hangs forever)."""
    from app import gemini
    from app.llm import GeminiLLM

    monkeypatch.setattr(settings, "gemini_api_key", "dummy")
    monkeypatch.setattr(settings, "llm_model", "model-a")
    monkeypatch.setattr(settings, "llm_fallback_models", ["model-b"])
    monkeypatch.setattr(settings, "hedge_after_s", hedge_after)
    gemini._parked_until.clear()
    llm = GeminiLLM()
    started, closed = [], []

    class C:
        def __init__(self, text):
            self.text = text

    async def fake_one(model, contents, system):
        started.append(model)
        try:
            d = delays[model]
            await asyncio.sleep(3600 if d is None else d)
            yield C(f"{model} বলছে।")
            yield C(" শেষ।")
        finally:
            closed.append(model)

    monkeypatch.setattr(llm, "_stream_one", fake_one)
    return llm, started, closed


def test_hedge_rescues_a_hanging_model(monkeypatch):
    llm, started, closed = _hedge_llm(monkeypatch, {"model-a": None, "model-b": 0.01})

    async def run():
        t = time.perf_counter()
        out = [x async for x in llm.stream("প্রশ্ন", [], [])]
        return out, time.perf_counter() - t

    out, took = asyncio.run(run())
    assert out == ["model-b বলছে।", " শেষ।"] and llm.last_model == "model-b"
    assert started == ["model-a", "model-b"] and "model-a" in closed  # loser cancelled
    assert took < 1


def test_no_hedge_when_first_model_is_fast(monkeypatch):
    llm, started, _ = _hedge_llm(monkeypatch, {"model-a": 0.0, "model-b": 0.0}, hedge_after=0.5)
    out = asyncio.run(_collect_async(llm))
    assert out[0] == "model-a বলছে।" and started == ["model-a"]


def test_hedge_first_model_still_wins_if_it_answers_first(monkeypatch):
    # a is slow (hedge fires), but b is even slower: a's answer is used
    llm, started, closed = _hedge_llm(monkeypatch, {"model-a": 0.1, "model-b": 1.0}, hedge_after=0.02)
    out = asyncio.run(_collect_async(llm))
    assert out[0] == "model-a বলছে।" and started == ["model-a", "model-b"] and "model-b" in closed


def test_hedging_off(monkeypatch):
    llm, started, _ = _hedge_llm(monkeypatch, {"model-a": 0.1, "model-b": 0.0}, hedge_after=0)
    out = asyncio.run(_collect_async(llm))
    assert out[0] == "model-a বলছে।" and started == ["model-a"]


async def _collect_async(llm):
    return [x async for x in llm.stream("প্রশ্ন", [], [])]


def test_quota_kind_and_daily_parking():
    import time as _t

    from app import gemini

    minute = Exception("""'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier' ... Please retry in 41.29s.""")
    day = Exception("""'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'""")
    assert gemini.quota_kind(minute) == "per-minute" and gemini.quota_kind(day) == "per-day"
    assert gemini.quota_kind(Exception("429")) == "unknown"
    gemini._parked_until.clear()
    gemini.park("m-day", day)
    gemini.park("m-min", minute)
    now = _t.monotonic()
    assert gemini._parked_until["m-day"] - now > 3500
    assert 40 < gemini._parked_until["m-min"] - now < 42


# ---------------- language + TTS cache ----------------
def test_is_mostly_english():
    from app.textnorm import is_mostly_english

    assert is_mostly_english("what is the lowest pack price?")
    assert not is_mostly_english("আমার monthly data pack কত?")
    assert not is_mostly_english("আমার ব্যালেন্স কত?")
    assert not is_mostly_english("১২১")


def test_english_question_gets_english_instruction(monkeypatch):
    from app.llm import ENGLISH_TURN, GeminiLLM

    monkeypatch.setattr(settings, "gemini_api_key", "dummy")
    llm = GeminiLLM()
    seen = []

    async def fake_stream(contents, system):
        seen.append(contents[-1].parts[0].text)
        yield "ok"

    monkeypatch.setattr(llm, "_stream", fake_stream)

    async def ask(q):
        return [t async for t in llm.stream(q, [], [])]

    asyncio.run(ask("what is the lowest pack price?"))
    asyncio.run(ask("সবচেয়ে সস্তা প্যাক কোনটা?"))
    assert ENGLISH_TURN in seen[0] and ENGLISH_TURN not in seen[1]


def test_edge_voice_switches_for_english(monkeypatch):
    from app.tts import edge_voice_for

    monkeypatch.setattr(settings, "tts_voice_en", "en-IN-NeerjaNeural")
    assert edge_voice_for("The cheapest pack is 49 taka.") == "en-IN-NeerjaNeural"
    assert edge_voice_for("সবচেয়ে সস্তা প্যাক ৪৯ টাকা।") == settings.tts_voice
    monkeypatch.setattr(settings, "tts_voice_en", "")
    assert edge_voice_for("The cheapest pack is 49 taka.") == settings.tts_voice


class CountingInnerTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text):
        self.calls.append(text)
        return f"audio:{text}".encode(), "audio/mpeg"


def test_tts_cache_memory_and_disk(tmp_path):
    from app.tts import CachedTTS

    inner = CountingInnerTTS()
    c = CachedTTS(inner, tmp_path)
    s = "ব্যালেন্স জানতে *১২১# ডায়াল করুন।"
    assert not c.is_cached(s)
    a1 = asyncio.run(c.synthesize(s))
    a2 = asyncio.run(c.synthesize(s + "  "))   # same sentence, extra spaces
    assert a1 == a2 and len(inner.calls) == 1 and c.is_cached(s)
    # a fresh cache (server restart) finds it on disk: still no network call
    c2 = CachedTTS(inner, tmp_path)
    assert asyncio.run(c2.synthesize(s)) == a1 and len(inner.calls) == 1


def test_tts_cache_key_changes_with_voice(tmp_path, monkeypatch):
    from app.tts import CachedTTS

    inner = CountingInnerTTS()
    c = CachedTTS(inner, tmp_path)
    asyncio.run(c.synthesize("হ্যালো"))
    monkeypatch.setattr(settings, "tts_voice", "bn-BD-PradeepNeural")
    asyncio.run(c.synthesize("হ্যালো"))
    assert len(inner.calls) == 2   # new voice -> not served stale audio


def test_tts_cache_does_not_store_empty_audio(tmp_path):
    from app.tts import CachedTTS

    class EmptyTTS:
        async def synthesize(self, text):
            return b"", "audio/mpeg"

    c = CachedTTS(EmptyTTS(), tmp_path)
    asyncio.run(c.synthesize("হ্যালো"))
    assert not c.is_cached("হ্যালো") and not list(tmp_path.iterdir())


def test_pipeline_counts_cache_hits(tmp_path):
    from app.tts import CachedTTS

    tts = CachedTTS(CountingInnerTTS(), tmp_path)

    async def run():
        sent = []

        async def send(m):
            sent.append(m)

        s = _session(send, FakeLLM(), tts)
        for _ in range(2):
            await s.on_text("ব্যালেন্স কিভাবে দেখব")
            await s.turn_task
        return [m for m in sent if m["type"] == "metrics"]

    first, second = asyncio.run(run())
    assert "tts_cache_hits" not in first and second["tts_cache_hits"] == 2


# ---------------- WER test set: recorder + eval ----------------
def _recorder_client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import recorder

    monkeypatch.setattr(recorder, "CLIPS", tmp_path / "my_clips")
    monkeypatch.setattr(recorder, "MANIFEST", tmp_path / "manifest.csv")
    app_ = FastAPI()
    app_.include_router(recorder.router)
    return TestClient(app_), recorder


def test_wer_sentences_file_is_valid():
    import csv as _csv

    from app.recorder import SENTENCES

    rows = list(_csv.DictReader(SENTENCES.open(encoding="utf-8")))
    assert len(rows) == 30 and len({r["id"] for r in rows}) == 30
    assert {r["category"] for r in rows} == {"bangla", "code-mixed", "english"}


def test_recorder_saves_wav_and_manifest(tmp_path, monkeypatch):
    import csv as _csv

    import soundfile as sf

    client, rec = _recorder_client(tmp_path, monkeypatch)
    items = client.get("/api/wer/sentences").json()
    assert len(items) == 30 and not any(i["recorded"] for i in items)

    tone = (0.3 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000) * 32767).astype("<i2")
    r = client.post("/api/wer/recordings/bn01", content=tone.tobytes())
    assert r.status_code == 200 and r.json()["seconds"] == 1.0 and r.json()["recorded"] == 1

    audio, sr = sf.read(tmp_path / "my_clips" / "bn01.wav")
    assert sr == 16000 and len(audio) == 16000
    rows = list(_csv.DictReader((tmp_path / "manifest.csv").open(encoding="utf-8")))
    assert rows == [{"path": "my_clips/bn01.wav", "reference": "আমার ব্যালেন্স কত টাকা আছে",
                     "category": "bangla", "id": "bn01"}]
    assert client.get("/api/wer/recordings/bn01.wav").status_code == 200
    assert client.get("/api/wer/sentences").json()[0]["recorded"]


def test_recorder_rejects_bad_input(tmp_path, monkeypatch):
    client, _ = _recorder_client(tmp_path, monkeypatch)
    short = np.zeros(1000, dtype="<i2").tobytes()
    assert client.post("/api/wer/recordings/bn01", content=short).status_code == 400
    assert client.post("/api/wer/recordings/zz99", content=b"\0" * 32000).status_code == 404
    assert client.post("/api/wer/recordings/..%2F..%2Fx", content=b"\0" * 32000).status_code in (400, 404)
    assert client.get("/api/wer/recordings/bn02.wav").status_code == 404


def _eval_setup(tmp_path, monkeypatch, transcripts):
    import argparse
    import csv as _csv

    import soundfile as sf

    from eval import eval_asr

    monkeypatch.chdir(tmp_path)
    (tmp_path / "clips").mkdir()
    rows = [("bn01", "bangla", "আমার ব্যালেন্স কত টাকা আছে"),
            ("mx01", "code-mixed", "আমার balance চেক করব কীভাবে"),
            ("en01", "english", "what is my current balance")]
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["path", "reference", "category", "id"])
        for i, c, ref in rows:
            sf.write(tmp_path / "clips" / f"{i}.wav", np.zeros(8000, dtype="float32"), 16000)
            w.writerow([f"clips/{i}.wav", ref, c, i])

    class FakeASR:
        calls = 0
        last_model = "fake-model"

        async def transcribe(self, audio, sr=16000):
            FakeASR.calls += 1
            return transcripts[(FakeASR.calls - 1) % len(transcripts)]

    monkeypatch.setattr(eval_asr, "build_asr", lambda: FakeASR())
    args = lambda **kw: argparse.Namespace(**{"manifest": str(tmp_path / "manifest.csv"),
                                              "conditions": "clean", "rpm": 0, "resume": None,
                                              "model": None, "rescore": None, **kw})
    return eval_asr, args, FakeASR


def test_eval_asr_per_category_and_resume(tmp_path, monkeypatch):
    eval_asr, args, FakeASR = _eval_setup(tmp_path, monkeypatch, [
        "আমার ব্যালেন্স কত টাকা আছে।",      # perfect (punctuation ignored)
        "আমার ব্যালেন্স চেক করব কীভাবে",    # "balance" written in Bangla script -> 1 error
        "what is my current balance",
    ])
    asyncio.run(eval_asr.main(args()))
    out = next((tmp_path / "eval" / "results").glob("asr_*[0-9].csv"))
    md = out.with_name(out.stem + "_summary.md").read_text(encoding="utf-8")
    assert "| clean | **ALL** | 3 | 6.7% |" in md          # 1 error / 15 words
    assert "| clean | code-mixed | 1 | 20.0% |" in md
    assert "| clean | english | 1 | 0.0% |" in md
    assert "mx01" in md.split("## Worst clips")[1]
    # resume with an extra condition: only the 3 new transcriptions run
    asyncio.run(eval_asr.main(args(conditions="clean,phone", resume=str(out))))
    assert FakeASR.calls == 6


def test_eval_asr_waits_on_quota(tmp_path, monkeypatch):
    from app.gemini import QuotaExhausted

    eval_asr, args, _ = _eval_setup(tmp_path, monkeypatch, ["x"])
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(eval_asr.asyncio, "sleep", fake_sleep)

    class Flaky:
        n = 0

        async def transcribe(self, audio, sr=16000):
            Flaky.n += 1
            if Flaky.n == 1:
                raise QuotaExhausted(12.0)
            return "ok"

    assert asyncio.run(eval_asr.transcribe_with_quota(Flaky(), np.zeros(10))) == "ok"
    assert slept == [13.0]


def test_script_error_detection():
    from eval.eval_asr import script_error

    ref = "নতুন সিম তুলতে কী কী কাগজ লাগবে"
    assert script_error(ref, "Notun SIM tolte ki ki kagoj lagbe?") == "romanized"
    assert script_error(ref, "নতুন সিম তুলতে কী কী কাগজ লাগবে।") == ""   # danda is NOT Hindi
    assert script_error(ref, "নতুন সিম তুলতে কী কী কাগজ লাগবে॥") == ""
    assert script_error("monthly internet package এর দাম কত", "monthly internet package दाम कत?") == "devanagari"
    assert script_error("আমার balance চেক করব কীভাবে", "আমার ব্যালেন্স চেক করব কিভাবে?") == ""
    assert script_error("what is my current balance", "what is my current balance") == ""


def test_spelling_variants_are_not_errors():
    from eval.eval_asr import norm

    assert norm("কী কী লাগবে") == norm("কি কি লাগবে")
    assert norm("চেক করবো কীভাবে?") == norm("চেক করব কিভাবে")
    assert norm("কোনো প্যাকেজ") != norm("কোন প্যাকেজ")   # different words, kept apart


def test_rescore_needs_no_api(tmp_path, monkeypatch):
    import csv as _csv

    from eval import eval_asr

    monkeypatch.setattr(eval_asr, "build_asr", lambda: (_ for _ in ()).throw(AssertionError("no API")))
    p = tmp_path / "asr_run.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=eval_asr.FIELDS)
        w.writeheader()
        w.writerow({"condition": "clean", "id": "bn09", "category": "bangla", "path": "x.wav",
                    "reference": "নতুন সিম তুলতে কী কী কাগজ লাগবে",
                    "hypothesis": "Notun SIM tolte ki ki kagoj lagbe?", "wer": 1, "cer": 1,
                    "latency_s": 1, "audio_s": 2, "model": "m"})
        w.writerow({"condition": "clean", "id": "bn01", "category": "bangla", "path": "y.wav",
                    "reference": "আমার ব্যালেন্স কত টাকা আছে", "hypothesis": "আমার ব্যালেন্স কত টাকা আছে।",
                    "wer": 0, "cer": 0, "latency_s": 1, "audio_s": 2, "model": "m"})
    asyncio.run(eval_asr.main(__import__("argparse").Namespace(
        rescore=str(p), model=None, manifest="", conditions="clean", rpm=0, resume=None)))
    md = (tmp_path / "asr_run_rescored_summary.md").read_text(encoding="utf-8")
    assert "| clean | bangla | 2 | 58.3% |" in md and "| 1 | 0.0% |" in md   # 7/12 words; ok clip perfect


def test_loanword_insensitive_scoring():
    from eval.eval_asr import norm, norm_loan

    ref, hyp = "net speed এত slow কেন", "নেট স্পিড এত স্লো কেন?"
    assert norm_loan(ref) == norm_loan(hyp)          # same words, other script
    assert norm(ref) != norm(hyp)                     # plain WER still counts them
    assert norm_loan("সবচেয়ে cheap data pack") != norm_loan("সবচেয়ে ভালো ডেটা প্যাক")  # real error stays
