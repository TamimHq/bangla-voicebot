"""Small audio helpers: PCM conversion, WAV encoding, telephone-band simulation."""
from __future__ import annotations

import io
import wave

import numpy as np
from scipy.signal import resample_poly


def pcm16_bytes_to_float(buf: bytes) -> np.ndarray:
    """Little-endian int16 PCM -> float32 in [-1, 1]."""
    return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0


def float_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """float32 mono -> in-memory 16-bit WAV (what the ASR APIs accept)."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    g = np.gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def telephone_band(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate a narrowband (8 kHz) phone call: down to 8 kHz and back.

    Everything above 4 kHz is lost, the same as on a real telecom line. Used in
    evaluation to measure how much WER degrades on telephony-grade audio.
    """
    return resample(resample(audio, sr, 8000), 8000, sr)


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white noise at a target signal-to-noise ratio (dB)."""
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(audio**2)) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (audio + rng.normal(0, np.sqrt(p_noise), audio.shape)).astype(np.float32)
