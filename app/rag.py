"""Lightweight retrieval over the telecom knowledge base.

Uses BM25 on character n-grams instead of whole words. Bangla is
agglutinative: "প্যাকেজের", "প্যাকেজটি", "প্যাকেজে" are the same word with
different suffixes, and whole-word BM25 treats them as unrelated. Character
3-5-grams share most of their pieces, so they still match. ASR spelling
mistakes are also tolerated for the same reason.

No model download, runs in milliseconds on CPU. You can swap in BGE-M3 +
ChromaDB (as in ragkit) for a dense or hybrid retriever later.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .textnorm import normalize_for_wer


@dataclass
class Chunk:
    title: str
    text: str


def load_kb(path: Path) -> list[Chunk]:
    """Split a markdown file on '## ' headings; each section is one chunk."""
    raw = path.read_text(encoding="utf-8")
    chunks = []
    for block in re.split(r"^## ", raw, flags=re.M)[1:]:
        title, _, body = block.partition("\n")
        chunks.append(Chunk(title.strip(), body.strip()))
    return chunks


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    grams = []
    for word in normalize_for_wer(text).split():
        w = f" {word} "  # word boundaries become part of the n-grams
        for n in range(n_min, n_max + 1):
            grams.extend(w[i : i + n] for i in range(max(1, len(w) - n + 1)))
    return grams


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(char_ngrams(c.title + " " + c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lens) / max(1, len(self.lens))
        df = Counter(g for d in self.docs for g in d)
        n = len(self.docs)
        self.idf = {g: math.log(1 + (n - f + 0.5) / (f + 0.5)) for g, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = Counter(char_ngrams(query))
        scores = []
        for i, doc in enumerate(self.docs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
            for g, qtf in q.items():
                tf = doc.get(g)
                if tf:
                    s += self.idf[g] * tf * (self.k1 + 1) / (tf + norm)
            scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scores[:k] if s > 0]
