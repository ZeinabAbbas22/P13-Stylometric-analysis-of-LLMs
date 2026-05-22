"""Punctuation-distribution features."""
from __future__ import annotations

from collections import Counter

TRACKED_PUNCT = {
    "comma": ",",
    "period": ".",
    "semicolon": ";",
    "colon": ":",
    "question": "?",
    "exclamation": "!",
    "dash": "—",
    "hyphen": "-",
    "quote": "\"",
    "single_quote": "'",
    "ellipsis": "…",
}


def compute(text: str) -> dict[str, float]:
    """Frequency (per 100 characters) of selected punctuation marks."""
    n_chars = max(len(text), 1)
    counts = Counter(text)
    out: dict[str, float] = {}
    for name, ch in TRACKED_PUNCT.items():
        out[f"punct_{name}"] = round(counts.get(ch, 0) / n_chars * 100, 4)
    out["punct_total"] = round(
        sum(counts.get(ch, 0) for ch in TRACKED_PUNCT.values()) / n_chars * 100,
        4,
    )
    return out
