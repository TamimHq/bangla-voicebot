"""Recording studio for the WER test set (open http://localhost:8000/record).

The page shows one sentence from eval/data/wer_sentences.csv at a time. The
browser sends 16 kHz PCM16 audio; it is saved as eval/data/my_clips/<id>.wav
and eval/data/manifest.csv is rewritten so `python -m eval.eval_asr` can run
on it directly.
"""
from __future__ import annotations

import csv
import re

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .audio import float_to_wav_bytes, pcm16_bytes_to_float
from .config import ROOT

router = APIRouter()

DATA = ROOT / "eval" / "data"
SENTENCES = DATA / "wer_sentences.csv"
CLIPS = DATA / "my_clips"
MANIFEST = DATA / "manifest.csv"
SR = 16000
MIN_S, MAX_S = 0.4, 20.0
_ID = re.compile(r"^[a-z]{2}\d{2}$")


def load_sentences() -> list[dict]:
    with SENTENCES.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(sentences: list[dict]) -> int:
    """manifest.csv lists only sentences that have a recording."""
    rows = [
        {"path": f"my_clips/{s['id']}.wav", "reference": s["reference"],
         "category": s["category"], "id": s["id"]}
        for s in sentences if (CLIPS / f"{s['id']}.wav").exists()
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "reference", "category", "id"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _find(sentence_id: str) -> dict:
    if not _ID.match(sentence_id):
        raise HTTPException(400, "bad id")
    for s in load_sentences():
        if s["id"] == sentence_id:
            return s
    raise HTTPException(404, "unknown sentence")


@router.get("/record")
def record_page() -> FileResponse:
    return FileResponse(ROOT / "static" / "record.html")


@router.get("/api/wer/sentences")
def sentences() -> list[dict]:
    out = []
    for s in load_sentences():
        p = CLIPS / f"{s['id']}.wav"
        out.append({**s, "recorded": p.exists()})
    return out


@router.post("/api/wer/recordings/{sentence_id}")
async def save_recording(sentence_id: str, request: Request) -> dict:
    _find(sentence_id)
    body = await request.body()
    audio = pcm16_bytes_to_float(body[: len(body) // 2 * 2])
    dur = len(audio) / SR
    if dur < MIN_S:
        raise HTTPException(400, f"too short ({dur:.1f}s): record the whole sentence")
    if dur > MAX_S:
        raise HTTPException(400, f"too long ({dur:.1f}s)")
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    CLIPS.mkdir(parents=True, exist_ok=True)
    (CLIPS / f"{sentence_id}.wav").write_bytes(float_to_wav_bytes(audio, SR))
    n = write_manifest(load_sentences())
    return {"ok": True, "seconds": round(dur, 2), "peak": round(peak, 3), "recorded": n}


@router.get("/api/wer/recordings/{sentence_id}.wav")
def get_recording(sentence_id: str) -> FileResponse:
    _find(sentence_id)
    p = CLIPS / f"{sentence_id}.wav"
    if not p.exists():
        raise HTTPException(404, "not recorded yet")
    return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "no-store"})
