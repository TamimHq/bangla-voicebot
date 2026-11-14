"""Benchmark TTS backends on YOUR connection: how long until audio is ready?

Usage:
  python -m eval.bench_tts                       # edge and gemini
  python -m eval.bench_tts --backends edge,gemini,mms --runs 5
  python -m eval.bench_tts --backends edge --parallel 3   # do simultaneous requests slow down?

For Edge it also reports time-to-first-audio-byte: if first byte is almost as
slow as the total, the delay is connection setup (network), not synthesis.
Sample outputs are saved to eval/results/ so you can compare voice quality.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

from app.config import settings
from app.textnorm import canonical
from app.tts import build_tts

TEXTS = {
    "short": "আমাদের সবচেয়ে সস্তা প্যাক হলো ১ জিবি ইন্টারনেট,",
    "long": "এটি কিনতে স্টার ১২১ স্টার ১ হ্যাশ ডায়াল করুন। মেয়াদ থাকবে তিন দিন, দাম মাত্র ৪৯ টাকা।",
}
OUT = Path("eval/results")


async def edge_first_byte(text: str) -> tuple[float, float]:
    import edge_tts

    t0 = time.perf_counter()
    first = None
    async for part in edge_tts.Communicate(canonical(text), settings.tts_voice).stream():
        if part["type"] == "audio" and first is None:
            first = time.perf_counter() - t0
    return first or 0.0, time.perf_counter() - t0


async def bench(backend: str, runs: int) -> None:
    try:
        t = time.perf_counter()
        tts = build_tts(backend)
        load_s = time.perf_counter() - t
    except Exception as e:
        print(f"\n[{backend}] could not start: {type(e).__name__}: {e}")
        return
    print(f"\n[{backend}]  (init {load_s:.2f}s)")
    for name, text in TEXTS.items():
        totals, firsts = [], []
        for i in range(runs):
            try:
                if backend == "edge":
                    f, tot = await edge_first_byte(text)
                    firsts.append(f)
                else:
                    t = time.perf_counter()
                    audio, mime = await tts.synthesize(text)
                    tot = time.perf_counter() - t
                    if i == 0:
                        OUT.mkdir(parents=True, exist_ok=True)
                        ext = "mp3" if mime == "audio/mpeg" else "wav"
                        (OUT / f"tts_{backend}_{name}.{ext}").write_bytes(audio)
                totals.append(tot)
            except Exception as e:
                print(f"  {name}: run {i + 1} failed: {type(e).__name__}: {e}")
        if backend == "edge" and totals:
            audio, _ = await tts.synthesize(text)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"tts_edge_{name}.mp3").write_bytes(audio)
        if totals:
            line = f"  {name:<6} total  median {statistics.median(totals):.2f}s  (min {min(totals):.2f}, max {max(totals):.2f})"
            if firsts:
                line += f"   first byte median {statistics.median(firsts):.2f}s"
            print(line)


async def bench_parallel(backend: str, n: int) -> None:
    """Fire n requests at once (what the bot did before TTS_CONCURRENCY) vs one alone."""
    tts = build_tts(backend)
    text = TEXTS["short"]

    async def one():
        t = time.perf_counter()
        await tts.synthesize(text)
        return time.perf_counter() - t

    alone = statistics.median([await one() for _ in range(3)])
    together = await asyncio.gather(*(one() for _ in range(n)))
    print(f"\n[{backend}] one request alone: {alone:.2f}s")
    print(f"[{backend}] {n} at once: first finished {min(together):.2f}s, slowest {max(together):.2f}s")
    if min(together) > alone * 1.5:
        print("-> Parallel requests slow each other down. Keep TTS_CONCURRENCY=1.")
    else:
        print("-> Parallel requests don't hurt much. TTS_CONCURRENCY=2 or 3 is fine.")


async def main(args) -> None:
    if args.parallel:
        for b in args.backends.split(","):
            await bench_parallel(b.strip(), args.parallel)
        return
    print(f"{args.runs} runs per text. Lower is better.")
    for b in args.backends.split(","):
        await bench(b.strip(), args.runs)
    print(f"\nListen to samples in {OUT.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backends", default="edge,gemini")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--parallel", type=int, default=0, help="test N simultaneous requests")
    asyncio.run(main(p.parse_args()))
