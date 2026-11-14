"""Time cascade vs direct mode on the same audio clip, with NO timeout.

Uses a saved Bangla clip (by default the MMS sample from bench_tts) so no mic
is needed. Prints how long each Gemini call takes and what it returned.

Usage:
  python -m eval.probe_gemini
  python -m eval.probe_gemini --audio path\\to\\clip.wav --runs 2
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.asr import GeminiASR
from app.audio import resample
from app.config import settings
from app.llm import GeminiLLM
from app.rag import BM25Retriever, load_kb

DEFAULT_AUDIO = Path("eval/results/tts_mms_short.wav")


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), sr, 16000)


async def timed_stream(agen) -> tuple[float | None, float, str]:
    t0, first, out = time.perf_counter(), None, ""
    async for tok in agen:
        if first is None:
            first = time.perf_counter() - t0
        out += tok
    return first, time.perf_counter() - t0, out


async def run_once(audio, asr, llm, retriever) -> None:
    print(f"\n--- cascade (ASR model: {settings.gemini_asr_model}, LLM: {settings.llm_model})")
    try:
        t = time.perf_counter()
        text = await asr.transcribe(audio)
        asr_s = time.perf_counter() - t
        print(f"  ASR call:           {asr_s:5.2f}s  -> {text!r}")
        hits = retriever.search(text, k=settings.rag_top_k)
        first, total, out = await timed_stream(llm.stream(text, hits, []))
        print(f"  LLM call:  1st token {first:5.2f}s, done {total:5.2f}s")
        print(f"  => answer starts {asr_s + first:5.2f}s after speech ends")
        print(f"  answer: {out[:120]!r}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    print(f"\n--- direct (one call, model: {settings.llm_model})")
    try:
        first, total, out = await timed_stream(llm.stream_audio(audio, retriever.chunks, []))
        print(f"  1st token {first:5.2f}s, done {total:5.2f}s")
        line = out.split("\n", 1)
        ok = out.lstrip().startswith("TRANSCRIPT:")
        print(f"  format ok: {ok}   first line: {line[0][:100]!r}")
        print(f"  answer: {(line[1] if len(line) > 1 else '')[:120]!r}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


async def main(args) -> None:
    path = Path(args.audio)
    if not path.exists():
        print(f"No audio at {path}. Run `python -m eval.bench_tts --backends mms` first, "
              "or pass --audio with any Bangla .wav file.")
        return
    audio = load(path)
    print(f"clip: {path} ({len(audio) / 16000:.1f}s)   thinking={settings.gemini_thinking}")
    asr, llm = GeminiASR(), GeminiLLM()
    retriever = BM25Retriever(load_kb(settings.kb_path))
    for i in range(args.runs):
        print(f"\n===== run {i + 1}/{args.runs}")
        await run_once(audio, asr, llm, retriever)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default=str(DEFAULT_AUDIO))
    p.add_argument("--runs", type=int, default=2)
    asyncio.run(main(p.parse_args()))
