"""Mosteller–Wallace style function-word frequency vector."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .. import config


def compute(tokens: Sequence[str]) -> dict[str, float]:
    """Relative frequency of each function word in :data:`config.FUNCTION_WORDS`.

    Returns a dict keyed by ``fw_<word>`` so that downstream code can simply
    treat them as additional numeric features.
    """
    n = max(len(tokens), 1)
    counts = Counter(tokens)
    return {
        f"fw_{w}": round(counts.get(w, 0) / n, 5)
        for w in config.FUNCTION_WORDS
    }
