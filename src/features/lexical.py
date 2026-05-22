"""Lexical (vocabulary-level) stylometric features."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np


def type_token_ratio(tokens: Sequence[str]) -> float:
    """Simple TTR = |V| / |N|."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mtld(tokens: Sequence[str], ttr_threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    A length-robust alternative to TTR. Higher MTLD = richer vocabulary.
    """
    if len(tokens) < 50:
        # MTLD is unstable on very short texts; fall back to TTR.
        return type_token_ratio(tokens) * 100

    def _mtld_pass(seq: Sequence[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        token_count = 0
        for tok in seq:
            types.add(tok)
            token_count += 1
            ttr = len(types) / token_count
            if ttr <= ttr_threshold:
                factors += 1
                types.clear()
                token_count = 0
        if token_count > 0:
            partial = (1 - (len(types) / token_count)) / (1 - ttr_threshold)
            factors += partial
        return len(seq) / factors if factors > 0 else len(seq)

    return (_mtld_pass(tokens) + _mtld_pass(list(reversed(tokens)))) / 2


def hapax_legomena_ratio(tokens: Sequence[str]) -> float:
    """Share of words that occur exactly once."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return sum(1 for c in counts.values() if c == 1) / len(tokens)


def avg_word_length(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(np.mean([len(t) for t in tokens]))


def word_length_std(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(np.std([len(t) for t in tokens]))


def long_word_ratio(tokens: Sequence[str], threshold: int = 7) -> float:
    """Share of tokens with length >= threshold characters."""
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if len(t) >= threshold) / len(tokens)


def yule_k(tokens: Sequence[str]) -> float:
    """Yule's K — vocabulary concentration. Lower = more diverse."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    n = len(tokens)
    m1 = n
    m2 = sum(freq ** 2 * count
             for freq, count in Counter(counts.values()).items())
    if m1 == 0:
        return 0.0
    return 1e4 * (m2 - m1) / (m1 ** 2)


def compute(tokens: Sequence[str]) -> dict[str, float]:
    """Compute all lexical features and return as a flat dict."""
    return {
        "ttr": round(type_token_ratio(tokens), 5),
        "mtld": round(mtld(tokens), 3),
        "hapax_ratio": round(hapax_legomena_ratio(tokens), 5),
        "avg_word_length": round(avg_word_length(tokens), 3),
        "word_length_std": round(word_length_std(tokens), 3),
        "long_word_ratio": round(long_word_ratio(tokens), 5),
        "yule_k": round(yule_k(tokens), 3),
    }
