"""Streaming voice activity detection with Silero VAD.

Audio arrives from the browser in arbitrary-sized chunks. We cut it into
32 ms frames (512 samples @ 16 kHz), score each frame with Silero, and run a
small state machine:

    SILENCE --(prob >= threshold)--> SPEECH --(silence >= min_silence_ms)--> emit utterance

Two events come out:
  * "speech_start": the caller started talking (used for barge-in: stop TTS)
  * "utterance":    a finished utterance (float32 audio) ready for ASR
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad

FRAME = 512  # samples per Silero frame at 16 kHz (32 ms)
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        torch.set_num_threads(1)  # VAD is tiny; one thread per stream is plenty
        _MODEL = load_silero_vad()
    return _MODEL


@dataclass
class VADEvent:
    kind: str  # "speech_start" | "utterance"
    audio: np.ndarray | None = None


class StreamingVAD:
    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        min_speech_ms: int = 250,
        max_utterance_s: float = 15.0,
        preroll_ms: int = 300,
        sr: int = 16000,
    ):
        self.model = get_model()
        self.sr = sr
        self.threshold = threshold
        # Hysteresis: once speaking, require a lower prob to count as silence.
        self.neg_threshold = max(threshold - 0.15, 0.01)
        frame_ms = FRAME * 1000 / sr
        self.silence_frames_needed = int(min_silence_ms / frame_ms)
        self.min_speech_frames = int(min_speech_ms / frame_ms)
        self.max_frames = int(max_utterance_s * 1000 / frame_ms)
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, int(preroll_ms / frame_ms)))
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self._pending = np.zeros(0, dtype=np.float32)
        self.in_speech = False
        self.frames: list[np.ndarray] = []
        self.speech_frames = 0
        self.silence_run = 0
        self.preroll.clear()

    def _prob(self, frame: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.model(torch.from_numpy(frame), self.sr).item())

    def feed(self, audio: np.ndarray) -> list[VADEvent]:
        """Push float32 mono audio; returns any events it produced."""
        events: list[VADEvent] = []
        self._pending = np.concatenate([self._pending, audio.astype(np.float32)])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            p = self._prob(frame)

            if not self.in_speech:
                if p >= self.threshold:
                    self.in_speech = True
                    self.frames = list(self.preroll) + [frame]
                    self.speech_frames, self.silence_run = 1, 0
                    events.append(VADEvent("speech_start"))
                else:
                    self.preroll.append(frame)
                continue

            self.frames.append(frame)
            if p >= self.neg_threshold:
                self.speech_frames += 1
                self.silence_run = 0
            else:
                self.silence_run += 1

            ended = self.silence_run >= self.silence_frames_needed
            too_long = len(self.frames) >= self.max_frames
            if ended or too_long:
                if self.speech_frames >= self.min_speech_frames:
                    # Trim most of the trailing silence, keep ~100 ms of tail.
                    keep_tail = min(self.silence_run, 3)
                    cut = len(self.frames) - self.silence_run + keep_tail
                    events.append(VADEvent("utterance", np.concatenate(self.frames[:cut])))
                self.in_speech = False
                self.frames, self.speech_frames, self.silence_run = [], 0, 0
                self.preroll.clear()
        return events
