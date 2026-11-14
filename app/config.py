"""Central settings, read from environment / .env.

Every backend is swappable from .env so the same code runs on a CPU laptop
(cloud ASR/TTS) or a GPU box (local models).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass
class Settings:
    # --- keys ---
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # --- backends ---
    asr_backend: str = field(default_factory=lambda: os.getenv("ASR_BACKEND", "gemini"))  # gemini | groq | local
    gemini_asr_model: str = field(default_factory=lambda: os.getenv("GEMINI_ASR_MODEL", "gemini-3.5-flash"))
    groq_asr_model: str = field(default_factory=lambda: os.getenv("GROQ_ASR_MODEL", "whisper-large-v3"))
    local_asr_model: str = field(default_factory=lambda: os.getenv("LOCAL_ASR_MODEL", "small"))
    asr_language: str = field(default_factory=lambda: os.getenv("ASR_LANGUAGE", "bn"))

    # cascade = ASR call -> retrieval -> LLM call; direct = one Gemini call hears the audio
    pipeline_mode: str = field(default_factory=lambda: os.getenv("PIPELINE_MODE", "direct"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-3.5-flash"))
    # Free-tier quota is per model: when the main model hits its limit, these are used instead.
    llm_fallback_models: list = field(default_factory=lambda: [
        m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "gemini-3.5-flash").split(",") if m.strip()])
    # If the model hasn't started answering after this long, also ask the fallback model
    # and use whichever answers first ("hedged request"). 0 = off.
    hedge_after_s: float = field(default_factory=lambda: _f("HEDGE_AFTER_S", 4.0))
    # Give up on a Gemini call after this long and speak the apology instead of hanging.
    gemini_timeout_s: float = field(default_factory=lambda: _f("GEMINI_TIMEOUT_S", 12.0))
    gemini_thinking: str = field(default_factory=lambda: os.getenv("GEMINI_THINKING", "minimal"))  # minimal|low|medium|high|off

    tts_backend: str = field(default_factory=lambda: os.getenv("TTS_BACKEND", "edge"))  # edge | mms
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "bn-BD-NabanitaNeural"))
    # English sentences get an English voice (Edge only). Empty = always use TTS_VOICE.
    tts_voice_en: str = field(default_factory=lambda: os.getenv("TTS_VOICE_EN", "en-IN-NeerjaNeural"))
    # Reuse audio for sentences already spoken (kept in memory and in cache/tts/). 0 = off.
    tts_cache: bool = field(default_factory=lambda: os.getenv("TTS_CACHE", "1") == "1")
    # Play a pre-made "জি, একটু দেখছি।" if no real audio after this many ms. 0 = off.
    filler_after_ms: int = field(default_factory=lambda: _i("FILLER_AFTER_MS", 1500))
    tts_concurrency: int = field(default_factory=lambda: _i("TTS_CONCURRENCY", 1))
    gemini_tts_model: str = field(default_factory=lambda: os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"))
    gemini_tts_voice: str = field(default_factory=lambda: os.getenv("GEMINI_TTS_VOICE", "Kore"))

    # --- VAD ---
    vad_threshold: float = field(default_factory=lambda: _f("VAD_THRESHOLD", 0.5))
    vad_min_silence_ms: int = field(default_factory=lambda: _i("VAD_MIN_SILENCE_MS", 600))
    vad_min_speech_ms: int = field(default_factory=lambda: _i("VAD_MIN_SPEECH_MS", 250))
    vad_max_utterance_s: float = field(default_factory=lambda: _f("VAD_MAX_UTTERANCE_S", 15.0))

    # --- RAG ---
    kb_path: Path = field(default_factory=lambda: ROOT / os.getenv("KB_PATH", "data/kb/telecom_faq_bn.md"))
    rag_top_k: int = field(default_factory=lambda: _i("RAG_TOP_K", 3))

    # --- privacy ---
    redact_pii_in_logs: bool = field(default_factory=lambda: os.getenv("REDACT_PII", "1") == "1")

    sample_rate: int = 16000


settings = Settings()
