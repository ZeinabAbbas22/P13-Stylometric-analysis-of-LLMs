"""Discourse-marker features."""
from __future__ import annotations

from .. import config


def compute(text: str, n_words: int) -> dict[str, float]:
    """Rate of discourse markers per 100 tokens.

    Looks for both single-token markers (``however``) and multi-word
    markers (``on the other hand``).
    """
    text_lower = text.lower()
    total = sum(text_lower.count(m) for m in config.DISCOURSE_MARKERS)
    rate = (total / n_words * 100) if n_words else 0.0
    return {
        "discourse_marker_count": int(total),
        "discourse_marker_rate": round(rate, 4),
    }
