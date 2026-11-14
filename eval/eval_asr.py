"""ASR evaluation: WER / CER per condition and per category.

Record the test set first at http://localhost:8000/record, which writes
eval/data/manifest.csv. Then (server stopped or not, either is fine):

  python -m eval.eval_asr --manifest eval/data/manifest.csv
  python -m eval.eval_asr --manifest eval/data/manifest.csv --conditions clean,phone,phone+noise10
  python -m eval.eval_asr --manifest eval/data/manifest.csv --resume eval/results/asr_20261114_213000.csv
  python -m eval.eval_asr --model gemini-3.6-flash --conditions clean     # compare models
  python -m eval.eval_asr --rescore eval/results/asr_20261114_213000.csv   # no API calls

Conditions:  clean | phone (8 kHz telephone band) | noiseN (white noise at N dB SNR),
             combined with "+", e.g. phone+noise10.

Manifest columns: path, reference, and optionally category and id.

Free tier: Gemini allows only a few requests per minute, so calls are paced
(--rpm) and a quota error makes the script wait and retry instead of failing.
Results are written row by row, so an interrupted run can be resumed.
Outputs: eval/results/asr_<time>.csv (every clip) and asr_<time>_summary.md.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

from app.asr import build_asr
from app.audio import add_noise, resample, telephone_band
from app.config import settings
from app.gemini import QuotaExhausted, is_quota_error
from app.textnorm import normalize_for_wer

FIELDS = ["condition", "id", "category", "path", "reference", "hypothesis",
          "wer", "cer", "latency_s", "audio_s", "model"]


# Accepted spelling variants: both forms are standard written Bangla, so they are
# not recognition errors. Applied to reference and hypothesis alike.
SPELLING_VARIANTS = {
    "কী": "কি", "কীভাবে": "কিভাবে", "কেমনে": "কিভাবে",
    "করবো": "করব", "বলবো": "বলব", "দেবো": "দেব",
    "ডেটা": "ডাটা", "ইমারজেন্সি": "ইমার্জেন্সি",
}
# English loanwords that Bangla speakers say and write either way ("balance" or
# "ব্যালেন্স"). For the loanword-insensitive score both spellings count as the same
# word. The plain WER does NOT use this, so both numbers are reported.
LOANWORDS = {
    "ব্যালেন্স": "balance", "রিচার্জ": "recharge", "ইন্টারনেট": "internet", "নেট": "net",
    "সিম": "sim", "কার্ড": "card", "স্পিড": "speed", "স্লো": "slow", "মিনিট": "minute",
    "প্যাক": "pack", "প্যাকেজ": "package", "ডাটা": "data", "চিপ": "cheap", "ব্লক": "block",
    "ইমার্জেন্সি": "emergency", "অফার": "offer", "মান্থলি": "monthly", "কাস্টমার": "customer",
    "কেয়ার": "care", "এজেন্ট": "agent", "অটো": "auto", "রিনিউ": "renew", "অফ": "off",
    "চেক": "check", "ই-সিম": "esim", "মেসেজ": "message", "নেটওয়ার্ক": "network",
}


def norm_loan(text: str) -> str:
    """norm() + loanwords mapped to one spelling."""
    return " ".join(LOANWORDS.get(w, w) for w in norm(text).split())


_BN = re.compile(r"[\u0980-\u09FF]")
_LATIN = re.compile(r"[A-Za-z]")
# Hindi letters only. The danda "।" and "॥" (U+0964/5) sit in the Devanagari block but
# are shared with Bangla, so they must not count as Hindi.
_DEVANAGARI = re.compile(r"[\u0900-\u0963\u0966-\u097F]")


def norm(text: str) -> str:
    """normalize_for_wer + accepted spelling variants."""
    return " ".join(SPELLING_VARIANTS.get(w, w) for w in normalize_for_wer(text).split())


def script_error(reference: str, hypothesis: str) -> str:
    """'' if the transcript is in the right script, else what went wrong.

    'romanized':  Bangla speech written in Latin letters ("Notun SIM tolte ki lagbe").
    'devanagari': Hindi script appeared.
    These are output-format failures, not mishearing, so they are reported apart.
    """
    if _DEVANAGARI.search(hypothesis):
        return "devanagari"
    ref_bn, ref_en = len(_BN.findall(reference)), len(_LATIN.findall(reference))
    hyp_bn, hyp_en = len(_BN.findall(hypothesis)), len(_LATIN.findall(hypothesis))
    ref_bn_share = ref_bn / max(1, ref_bn + ref_en)
    hyp_bn_share = hyp_bn / max(1, hyp_bn + hyp_en)
    if ref_bn_share >= 0.3 and hyp_bn_share < ref_bn_share * 0.5:
        return "romanized"
    return ""


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), sr, 16000)


def apply(cond: str, audio: np.ndarray) -> np.ndarray:
    """cond like 'clean', 'phone', 'noise10', 'phone+noise10'."""
    for part in cond.split("+"):
        if part == "phone":
            audio = telephone_band(audio)
        elif part.startswith("noise"):
            audio = add_noise(audio, snr_db=float(part[5:]))
    return audio


def wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Corpus-level (all words pooled), not an average of per-clip scores."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0, 0.0
    r, h = zip(*pairs)
    return jiwer.wer(list(r), list(h)), jiwer.cer(list(r), list(h))


async def transcribe_with_quota(asr, audio: np.ndarray) -> str:
    """Transcribe; if every Gemini model is out of quota, wait and retry."""
    for attempt in range(6):
        try:
            return await asr.transcribe(audio)
        except QuotaExhausted as e:
            wait = max(5.0, e.wait_s + 1)
        except Exception as e:
            if not is_quota_error(e):
                raise
            wait = 30.0
        print(f"    free-tier limit reached; waiting {wait:.0f}s (attempt {attempt + 1}/6)")
        await asyncio.sleep(wait)
    raise RuntimeError("Still rate-limited after 6 waits. Try again later, or lower --rpm.")


def summarize(rows: list[dict]) -> str:
    """Markdown summary: per condition, then per category within each condition."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    lines = ["| condition | category | clips | WER | WER, loanword-insensitive | CER | wrong-script clips "
             "| WER without them | p50 latency |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond, rs in by_cond.items():
        groups = {"ALL": rs}
        for r in rs:
            groups.setdefault(r.get("category") or "uncategorized", []).append(r)
        for cat, g in groups.items():
            w, c = wer_cer([norm(x["reference"]) for x in g], [norm(x["hypothesis"]) for x in g])
            w_loan, _ = wer_cer([norm_loan(x["reference"]) for x in g], [norm_loan(x["hypothesis"]) for x in g])
            ok = [x for x in g if not script_error(x["reference"], x["hypothesis"])]
            w_ok, _ = wer_cer([norm(x["reference"]) for x in ok], [norm(x["hypothesis"]) for x in ok])
            lat = np.median([float(x["latency_s"]) for x in g])
            label = f"**{cat}**" if cat == "ALL" else cat
            bad = len(g) - len(ok)
            lines.append(f"| {cond} | {label} | {len(g)} | {w:.1%} | {w_loan:.1%} | {c:.1%} | {bad} | "
                         f"{w_ok:.1%} | {lat:.2f}s |")
    return "\n".join(lines)


def worst(rows: list[dict], n: int = 5) -> str:
    """The clips with the highest WER: the quickest way to see WHY errors happen."""
    def clip_wer(r):
        ref = norm(r["reference"])
        return jiwer.wer(ref, norm(r["hypothesis"])) if ref else 0.0

    rs = sorted(rows, key=clip_wer, reverse=True)[:n]
    out = []
    for r in rs:
        w = clip_wer(r)
        if w == 0:
            break
        tag = script_error(r["reference"], r["hypothesis"])
        out.append(f"- [{r['condition']}] {r.get('id') or r['path']} WER {w:.0%}"
                   f"{f' ({tag})' if tag else ''}\n  - ref: {r['reference']}\n  - hyp: {r['hypothesis']}")
    return "\n".join(out) or "- (no errors)"


def write_summary(rows: list[dict], out: Path, source: str, models: str) -> None:
    table, bad = summarize(rows), worst(rows)
    summary = (f"# ASR evaluation\n\nSource: `{source}`  \nASR: Gemini ({models})  \n"
               "Scored after normalization (Unicode, punctuation, Bangla digits, accepted spelling "
               "variants such as কী/কি). *Wrong-script clips* are Bangla speech written in Latin "
               "letters or Hindi script: heard right, written wrong. *Loanword-insensitive* WER also "
               "treats common English loanwords as the same word in either script "
               "(\"balance\" = \"ব্যালেন্স\").\n\n"
               f"{table}\n\n## Worst clips\n\n{bad}\n")
    md = out.with_name(out.stem + "_summary.md")
    md.write_text(summary, encoding="utf-8")
    print("\n" + table + "\n\nWorst clips:\n" + bad)
    print(f"\nSummary: {md}")


def rescore(path: Path) -> None:
    """Re-score a finished run with the current rules. No Gemini calls, no quota."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    models = ", ".join(sorted({r.get("model") or "?" for r in rows}))
    write_summary(rows, path.with_name(path.stem + "_rescored"), str(path), models)


async def main(args) -> None:
    if args.rescore:
        rescore(Path(args.rescore))
        return
    if args.model:
        # A clean comparison: this model only, no switching to fallbacks.
        settings.gemini_asr_model = args.model
        settings.llm_fallback_models = []
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"No manifest at {manifest}. Record clips first at http://localhost:8000/record")
        return
    clips = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if not clips:
        print("Manifest is empty. Record some clips first.")
        return
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        out = Path(args.resume)
        done_rows = list(csv.DictReader(out.open(encoding="utf-8")))
    else:
        out = out_dir / f"asr_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        done_rows = []
    done = {(r["condition"], r["path"]) for r in done_rows}

    todo = [(c, r) for c in conds for r in clips if (c, r["path"]) not in done]
    gap = 60.0 / args.rpm if args.rpm > 0 else 0.0
    print(f"{len(clips)} clips x {len(conds)} conditions = {len(clips) * len(conds)} transcriptions "
          f"({len(done)} already done, {len(todo)} to go)")
    print(f"ASR model: {settings.gemini_asr_model}, fallbacks: {settings.llm_fallback_models}, "
          f"pacing: {args.rpm}/min, so about {len(todo) * gap / 60:.0f} min\n")

    asr = build_asr()
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        last_call = 0.0
        for i, (cond, r) in enumerate(todo, 1):
            audio = apply(cond, load_audio(manifest.parent / r["path"]))
            sleep = gap - (time.monotonic() - last_call)
            if sleep > 0:
                await asyncio.sleep(sleep)
            last_call = time.monotonic()
            t = time.perf_counter()
            hyp = await transcribe_with_quota(asr, audio)
            lat = time.perf_counter() - t
            ref_n, hyp_n = norm(r["reference"]), norm(hyp)
            row = {
                "condition": cond, "id": r.get("id", ""), "category": r.get("category", ""),
                "path": r["path"], "reference": r["reference"], "hypothesis": hyp,
                "wer": round(jiwer.wer(ref_n, hyp_n), 4) if ref_n else 0.0,
                "cer": round(jiwer.cer(ref_n, hyp_n), 4) if ref_n else 0.0,
                "latency_s": round(lat, 3), "audio_s": round(len(audio) / 16000, 2),
                "model": getattr(asr, "last_model", "") or "",
            }
            w.writerow(row)
            f.flush()  # saved immediately: safe to stop and --resume
            done_rows.append({k: str(v) for k, v in row.items()})
            mark = "ok " if row["wer"] == 0 else f"{row['wer']:.0%}"
            print(f"[{i}/{len(todo)}] [{cond}] {row['id'] or r['path']} {mark:>4}  {hyp}")

    rows = [r for r in done_rows if r["condition"] in conds]
    used = ", ".join(sorted({r.get("model") or "?" for r in rows}))
    print(f"\nPer-clip results: {out}")
    write_summary(rows, out, str(manifest), used)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="eval/data/manifest.csv")
    p.add_argument("--conditions", default="clean,phone")
    p.add_argument("--rpm", type=float, default=8, help="max transcriptions per minute (free tier)")
    p.add_argument("--resume", help="continue a previous results CSV instead of starting a new one")
    p.add_argument("--model", help="transcribe with this Gemini model only (for comparisons)")
    p.add_argument("--rescore", help="re-score a finished results CSV; makes no API calls")
    asyncio.run(main(p.parse_args()))
