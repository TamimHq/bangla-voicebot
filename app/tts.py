"""Text-to-speech backends. Each returns (audio_bytes, mime_type) per chunk.

* EdgeTTS   - Microsoft neural voices (bn-BD-NabanitaNeural / bn-BD-PradeepNeural).
              Free, no account, good quality. Needs internet; each request opens a
              new connection to Microsoft, which can be slow from some regions.
* GeminiTTS - Gemini speech generation (same GEMINI_API_KEY). Supports bn-BD.
* MMSTTS    - Meta MMS VITS model for Bengali (facebook/mms-tts-ben). Offline,
              runs on CPU, more robotic, cannot read digits well.

Run `python -m eval.bench_tts` to measure which is fastest on your connection.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

from .audio import float_to_wav_bytes
from .config import settings
from .textnorm import canonical, is_mostly_english

log = logging.getLogger("voicebot")


class TTS(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> tuple[bytes, str]: ...


class CachedTTS(TTS):
    """Remembers synthesized sentences, in memory and on disk.

    Customer-care answers repeat a lot ("ব্যালেন্স জানতে *১২১# ডায়াল করুন।"),
    so a repeated sentence plays instantly with no network call. The disk copy
    survives server restarts. The key includes backend and voices, so changing
    the voice in .env never plays stale audio.
    """

    def __init__(self, inner: TTS, cache_dir: Path | None, max_memory: int = 300):
        self.inner = inner
        self.cache_dir = cache_dir
        self.max_memory = max_memory
        self.mem: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        norm = " ".join(canonical(text).split())
        raw = f"{settings.tts_backend}|{settings.tts_voice}|{settings.tts_voice_en}|{norm}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str, mime: str) -> Path:
        return self.cache_dir / f"{key}.{'mp3' if mime == 'audio/mpeg' else 'wav'}"

    def _remember(self, key: str, value: tuple[bytes, str]) -> None:
        self.mem[key] = value
        self.mem.move_to_end(key)
        while len(self.mem) > self.max_memory:
            self.mem.popitem(last=False)

    def is_cached(self, text: str) -> bool:
        key = self._key(text)
        if key in self.mem:
            return True
        return bool(self.cache_dir) and any(
            self._disk_path(key, m).exists() for m in ("audio/mpeg", "audio/wav"))

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        key = self._key(text)
        if key in self.mem:
            self.mem.move_to_end(key)
            return self.mem[key]
        if self.cache_dir:
            for mime in ("audio/mpeg", "audio/wav"):
                p = self._disk_path(key, mime)
                if p.exists() and p.stat().st_size > 0:
                    value = (p.read_bytes(), mime)
                    self._remember(key, value)
                    return value
        audio, mime = await self.inner.synthesize(text)
        if audio:  # never cache an empty result
            self._remember(key, (audio, mime))
            if self.cache_dir:
                try:
                    self._disk_path(key, mime).write_bytes(audio)
                except OSError as e:
                    log.warning("TTS cache write failed: %s", e)
        return audio, mime


def edge_voice_for(text: str) -> str:
    """A Bangla voice reads English sentences badly; use an English voice for those."""
    if settings.tts_voice_en and is_mostly_english(text):
        return settings.tts_voice_en
    return settings.tts_voice


class EdgeTTS(TTS):
    async def synthesize(self, text: str) -> tuple[bytes, str]:
        import edge_tts

        comm = edge_tts.Communicate(canonical(text), edge_voice_for(text))
        audio = bytearray()
        async for part in comm.stream():
            if part["type"] == "audio":
                audio.extend(part["data"])
        return bytes(audio), "audio/mpeg"


class GeminiTTS(TTS):
    SAMPLE_RATE = 24000  # Gemini TTS returns raw 16-bit mono PCM at 24 kHz

    def __init__(self):
        from google import genai
        from google.genai import types

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        from .gemini import make_client

        self.client = make_client()
        self.types = types

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        import numpy as np

        t = self.types
        resp = await self.client.aio.models.generate_content(
            model=settings.gemini_tts_model,
            # A short style hint keeps it a natural Bangladeshi customer-care read.
            contents=f"Read this in a clear, friendly Bangladeshi Bangla customer-care voice: {canonical(text)}",
            config=t.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=t.SpeechConfig(
                    language_code="bn-BD",
                    voice_config=t.VoiceConfig(
                        prebuilt_voice_config=t.PrebuiltVoiceConfig(voice_name=settings.gemini_tts_voice)
                    ),
                ),
            ),
        )
        pcm = resp.candidates[0].content.parts[0].inline_data.data
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return float_to_wav_bytes(audio, self.SAMPLE_RATE), "audio/wav"


class MMSTTS(TTS):
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.torch = torch
        self.model = VitsModel.from_pretrained("facebook/mms-tts-ben").eval()
        self.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
        self.sr = self.model.config.sampling_rate

    def _run(self, text: str) -> bytes:
        inputs = self.tok(canonical(text), return_tensors="pt")
        with self.torch.no_grad():
            wav = self.model(**inputs).waveform[0].numpy()
        return float_to_wav_bytes(wav, self.sr)

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        return await asyncio.to_thread(self._run, text), "audio/wav"


def build_tts(backend: str | None = None) -> TTS:
    backend = backend or settings.tts_backend
    if backend == "mms":
        return MMSTTS()
    if backend == "gemini":
        return GeminiTTS()
    return EdgeTTS()
