"""Filler phrases: hide slow network TTS/LLM behind an instant, pre-made reply.

At startup a few short neutral phrases are synthesized ONCE with the same TTS
voice and kept in memory. During a call, if the real answer isn't ready
FILLER_AFTER_MS after the caller stops talking, one of them plays immediately
(no network wait), and the real answer follows in the same voice.

This doesn't make the pipeline faster; it lowers *perceived* latency, which is
what callers notice. Both numbers are reported: filler_ms and first_audio_ms.
"""
from __future__ import annotations

import base64
import logging

log = logging.getLogger("voicebot")

# Short and neutral, so they fit any question.
PHRASES = [
    "জি, একটু দেখছি।",
    "আচ্ছা, এক মুহূর্ত।",
    "ঠিক আছে, দেখে বলছি।",
]


class FillerBank:
    def __init__(self) -> None:
        self.clips: list[tuple[str, str, str]] = []  # (text, mime, base64 audio)
        self._next = 0

    async def warm(self, tts) -> None:
        for p in PHRASES:
            try:
                audio, mime = await tts.synthesize(p)
                self.clips.append((p, mime, base64.b64encode(audio).decode()))
            except Exception as e:  # a missing filler is fine; the bot still works
                log.warning("filler %r not prepared: %s: %s", p, type(e).__name__, e)
        log.info("filler phrases ready: %d/%d", len(self.clips), len(PHRASES))

    def pick(self) -> tuple[str, str, str] | None:
        """Rotate through the clips so the caller doesn't hear the same one twice in a row."""
        if not self.clips:
            return None
        clip = self.clips[self._next % len(self.clips)]
        self._next += 1
        return clip
