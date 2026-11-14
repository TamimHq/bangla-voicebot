"""Retrieval evaluation: hit@1, hit@3, MRR over labelled caller questions.

Includes ASR-style misspellings and code-mixed queries, since that is what
the retriever actually receives from Whisper on a phone call.

Usage:  python -m eval.eval_rag
"""
from __future__ import annotations

import csv
from pathlib import Path

from app.config import settings
from app.rag import BM25Retriever, load_kb

QUERIES = Path(__file__).parent / "data" / "rag_queries.csv"


def main() -> None:
    retriever = BM25Retriever(load_kb(settings.kb_path))
    rows = list(csv.DictReader(QUERIES.open(encoding="utf-8")))
    hit1 = hit3 = rr = 0.0
    for r in rows:
        titles = [c.title for c, _ in retriever.search(r["query"], k=5)]
        rank = next((i + 1 for i, t in enumerate(titles) if t == r["expected"]), None)
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0
        mark = "ok " if rank == 1 else ("~  " if rank else "MISS")
        print(f"{mark} rank={rank}  {r['query']}  ->  {titles[0] if titles else '-'}")
    n = len(rows)
    print(f"\nqueries={n}  hit@1={hit1/n:.1%}  hit@3={hit3/n:.1%}  MRR={rr/n:.3f}")


if __name__ == "__main__":
    main()
