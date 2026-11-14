"""Speech-to-text backends.

* GeminiASR       - Gemini listens to the audio clip and transcribes it (default;
                    same API key as the LLM, runs from a CPU laptop)
* GroqWhisperASR  - Whisper large-v3 on Groq (optional)
* LocalWhisperASR - faster-whisper on your own machine (offline; GPU recommended)

All take 16 kHz float32 audio and return text.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np

from .audio import float_to_wav_bytes
from .config import settings
from .textnorm import canonical

# Whisper's `prompt` biases decoding toward these spellings. A code-mixed seed
# tells it Bangla + English terms appear together, and teaches telecom vocab.
BIAS_PROMPT = (
    "আমার ব্যালেন্স কত? আমি একটা ইন্টারনেট প্যাকেজ কিনতে চাই। "
    "SIM, recharge, data pack, MB, GB, minute, bKash, offer, validity।"
)


class ASR(ABC):
    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str: ...


TRANSCRIBE_INSTRUCTION = (
    "Transcribe this phone-call audio exactly as spoken. The caller speaks Bangla, "
    "English, or a mix of both (code-mixed).\n"
    "SCRIPT RULES (most important):\n"
    "1. Every Bangla word MUST be written in Bangla script (বাংলা লিপি). NEVER romanize "
    "Bangla: write \"নতুন সিম তুলতে কী লাগবে\", never \"Notun SIM tolte ki lagbe\".\n"
    "2. English words the caller says in English are written in English letters, "
    "e.g. \"আমার balance চেক করব\".\n"
    "3. Never use Hindi/Devanagari script (no दाम, कत).\n"
    "Do not translate, correct, summarize or answer. Output only the transcript, "
    "nothing else. If there is no clear speech, output an empty response.\n"
    f"Domain vocabulary that may appear: {BIAS_PROMPT}"
)


class GeminiASR(ASR):
    def __init__(self):
        from google import genai
        from google.genai import types

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        from .gemini import make_client

        self.client = make_client()
        self.types = types

    async def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        from .gemini import (
            QuotaExhausted, is_quota_error, make_config, park, retry_delay_s,
            thinking_unsupported, usable_models,
        )

        t = self.types
        contents = [
            t.Part.from_bytes(data=float_to_wav_bytes(audio, sr), mime_type="audio/wav"),
            TRANSCRIBE_INSTRUCTION,
        ]

        async def call(model: str):
            return await self.client.aio.models.generate_content(
                model=model, contents=contents, config=make_config(t, temperature=0.0)
            )

        # Same per-model quota rule as the LLM: on 429 park the model, try the next one.
        for model in usable_models(settings.gemini_asr_model):
            try:
                try:
                    resp = await call(model)
                except Exception as e:
                    if not thinking_unsupported(e):
                        raise
                    resp = await call(model)
                self.last_model = model  # eval reports which model produced each transcript
                return canonical((resp.text or "").strip())
            except Exception as e:
                if is_quota_error(e):
                    park(model, e)
                    continue
                raise
        raise QuotaExhausted(retry_delay_s(Exception()))


class GroqWhisperASR(ASR):
    def __init__(self):
        from groq import AsyncGroq

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    async def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        resp = await self.client.audio.transcriptions.create(
            file=("utterance.wav", float_to_wav_bytes(audio, sr)),
            model=settings.groq_asr_model,
            language=settings.asr_language,
            prompt=BIAS_PROMPT,
            temperature=0.0,
            response_format="json",
        )
        return canonical(resp.text.strip())


class LocalWhisperASR(ASR):
    """faster-whisper. On a CPU-only laptop use 'small' or 'base' with int8."""

    def __init__(self):
        from faster_whisper import WhisperModel  # optional dependency

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(settings.local_asr_model, device=device, compute_type=compute)

    def _run(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=settings.asr_language,
            initial_prompt=BIAS_PROMPT,
            beam_size=1,           # greedy: much faster on CPU, small WER cost
            vad_filter=False,      # we already ran Silero upstream
            condition_on_previous_text=False,
        )
        return canonical(" ".join(s.text.strip() for s in segments))

    async def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        return await asyncio.to_thread(self._run, audio)


def build_asr() -> ASR:
    if settings.asr_backend == "local":
        return LocalWhisperASR()
    if settings.asr_backend == "groq":
        return GroqWhisperASR()
    return GeminiASR()
