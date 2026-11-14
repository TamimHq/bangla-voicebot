"""Summarize per-stage latency from logs/turns.jsonl (written by every call),
separately for each pipeline mode, so cascade vs direct can be compared.

Usage:  python -m eval.latency_report
        python -m eval.latency_report --voice-only   # ignore typed messages
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from app.pipeline import TURN_LOG

STAGES = ["filler_ms", "asr_ms", "llm_first_token_ms", "llm_first_chunk_ms",
          "tts_first_ms", "first_audio_ms", "total_ms"]


def table(rows: list[dict]) -> None:
    print(f"  {'stage (ms from end of speech)':<30}{'n':>4}{'p50':>8}{'p90':>8}{'max':>8}")
    for s in STAGES:
        v = [r[s] for r in rows if s in r]
        if v:
            print(f"  {s:<30}{len(v):>4}{np.percentile(v, 50):>8.0f}{np.percentile(v, 90):>8.0f}{max(v):>8.0f}")


def main(args) -> None:
    if not TURN_LOG.exists():
        print("No turns logged yet. Talk to the bot first.")
        return
    recs = [json.loads(l) for l in TURN_LOG.open(encoding="utf-8") if l.strip()]
    if args.voice_only:
        recs = [r for r in recs if r.get("audio_s", 0) > 0]
    done = [r for r in recs if "total_ms" in r]
    print(f"turns: {len(recs)}   completed: {len(done)}   "
          f"interrupted (barge-in): {sum(1 for r in recs if r.get('interrupted'))}")
    for mode in ("cascade", "direct"):
        # Turns logged before modes existed were all cascade.
        rows = [r for r in done if r.get("mode", "cascade") == mode]
        if rows:
            print(f"\n[{mode}]")
            table(rows)
    print("\nfirst_audio_ms = voice-to-voice latency (caller stops talking -> real answer starts)")
    print("filler_ms      = when the pre-made filler phrase played (if it was needed)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--voice-only", action="store_true")
    main(p.parse_args())
