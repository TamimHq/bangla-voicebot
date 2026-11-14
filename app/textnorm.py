"""Bangla text normalization shared by evaluation, retrieval and TTS.

Why this matters: the same Bangla word can be stored as different Unicode
sequences. Without normalizing, WER counts "সময়" vs "সময়" as an error even
though they look identical, and retrieval misses exact matches.
"""
from __future__ import annotations

import re
import unicodedata

# য়, ড়, ঢ় are Unicode composition exclusions: NFC will NOT merge the
# decomposed form (base + nukta U+09BC) into the precomposed letter. ASR models
# emit either form, so we force one canonical (decomposed) spelling.
_NUKTA_MAP = {
    "য়": "য়",  # য়
    "ড়": "ড়",  # ড়
    "ঢ়": "ঢ়",  # ঢ়
}

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Danda (।), double danda (॥) and ordinary punctuation.
_PUNCT = re.compile(r"[।॥,.;:!?\"'“”‘’()\[\]{}<>\-–—…/\\|*#@&%~`^_+=]")
_ZW = re.compile(r"[​‌‍﻿]")  # zero-width chars
_SPACE = re.compile(r"\s+")


def canonical(text: str) -> str:
    """Unicode-level cleanup only (keeps punctuation). Safe for display/TTS."""
    text = unicodedata.normalize("NFC", text)
    for k, v in _NUKTA_MAP.items():
        text = text.replace(k, v)
    return _ZW.sub("", text)


def normalize_for_wer(text: str) -> str:
    """Aggressive normalization for scoring: no punctuation, ASCII digits, lowercase."""
    text = canonical(text).translate(_BN_DIGITS).lower()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


# Sentence boundary for streaming TTS: danda, ?, !, full stop, or newline.
_SENT_END = re.compile(r"([।॥?!.\n])")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a growing LLM text buffer into complete sentences + leftover.

    Lets TTS start speaking sentence 1 while the LLM is still writing sentence 2,
    which is the biggest single win for perceived latency.
    """
    parts = _SENT_END.split(buffer)
    sentences, i = [], 0
    while i + 1 < len(parts):
        s = (parts[i] + parts[i + 1]).strip()
        # Don't split decimals like "১.৫" / "1.5"
        if parts[i + 1] == "." and i + 2 < len(parts) and parts[i + 2][:1].isdigit() and parts[i][-1:].isdigit():
            parts[i + 2] = parts[i] + "." + parts[i + 2]
            i += 2
            continue
        if s:
            sentences.append(s)
        i += 2
    leftover = parts[i] if i < len(parts) else ""
    return sentences, leftover


_BN_LETTER = re.compile(r"[ঀ-৿]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def is_mostly_english(text: str, max_bangla_share: float = 0.15) -> bool:
    """True when the sentence is English, i.e. (almost) no Bangla words.

    Counted by words, not letters: code-mixed Bangla puts English NOUNS into
    Bangla grammar, and those nouns would outweigh short Bangla words by letters.
    "what is the lowest pack price?" -> True
    "আমার monthly data pack কত?"     -> False (code-mixed Bangla stays Bangla)
    """
    words = text.split()
    bn = sum(1 for w in words if _BN_LETTER.search(w))
    en = sum(1 for w in words if _LATIN_LETTER.search(w) and not _BN_LETTER.search(w))
    return en > 0 and bn / (bn + en) <= max_bangla_share


_CLAUSE_END = re.compile(r"[,،;]")


def split_first_clause(buffer: str, min_chars: int = 20) -> tuple[str | None, str]:
    """Cut the first clause at the last comma once it is at least `min_chars` long.

    Used only for the first chunk of a reply: TTS can start on
    "আমাদের সবচেয়ে সস্তা প্যাক হলো ১ জিবি ইন্টারনেট," while the LLM is still
    writing the rest. Commas inside numbers ("1,000") are not split points.
    """
    cut = -1
    for mt in _CLAUSE_END.finditer(buffer):
        i = mt.start()
        before, after = buffer[i - 1 : i], buffer[i + 1 : i + 2]
        if before.isdigit() and (after.isdigit() or after == ""):
            continue  # "1,000", or "1," at the end of a stream chunk (more digits may follow)
        if i + 1 >= min_chars:
            cut = i
    if cut < 0:
        return None, buffer
    return buffer[: cut + 1].strip(), buffer[cut + 1 :]


# --- simple PII redaction for logs (phone numbers, emails, long digit runs) ---
_PHONE = re.compile(r"(?:\+?৮৮|\+?88)?[0০][1১][3-9৩-৯][\d০-৯]{8}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_LONGNUM = re.compile(r"[\d০-৯]{6,}")


def redact_pii(text: str) -> str:
    text = _PHONE.sub("<PHONE>", text)
    text = _EMAIL.sub("<EMAIL>", text)
    return _LONGNUM.sub("<NUM>", text)
