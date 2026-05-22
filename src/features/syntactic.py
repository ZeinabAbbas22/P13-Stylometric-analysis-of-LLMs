"""Syntactic stylometric features computed from a spaCy ``Doc``."""
from __future__ import annotations

from collections import Counter

import numpy as np

# POS tags we want explicit ratios for (UD tagset).
POS_TAGS = (
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "ADP", "CCONJ", "SCONJ", "AUX", "PART", "INTJ", "NUM", "PROPN",
)


def _dep_depth(token) -> int:
    """Length of the path from token to the sentence root."""
    depth = 0
    cur = token
    while cur.head is not cur:
        depth += 1
        cur = cur.head
        if depth > 100:
            break
    return depth


def compute(doc) -> dict[str, float]:
    """Compute POS ratios, sentence-length stats and dependency depth."""
    sentences = list(doc.sents)
    sent_lengths = [
        sum(1 for t in s if t.is_alpha) for s in sentences
    ]
    alpha_tokens = [t for t in doc if t.is_alpha]
    total = len(alpha_tokens) or 1

    pos_counts = Counter(t.pos_ for t in alpha_tokens)
    pos_features = {
        f"pos_{tag.lower()}_ratio": round(pos_counts.get(tag, 0) / total, 5)
        for tag in POS_TAGS
    }

    depths = [_dep_depth(t) for t in doc if not t.is_space]
    max_depth = max(depths) if depths else 0
    mean_depth = float(np.mean(depths)) if depths else 0.0

    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    std_sent_len = float(np.std(sent_lengths)) if sent_lengths else 0.0

    return {
        **pos_features,
        "avg_sent_length": round(avg_sent_len, 3),
        "std_sent_length": round(std_sent_len, 3),
        "num_sentences": len(sentences),
        "num_words": total,
        "max_dep_depth": int(max_depth),
        "mean_dep_depth": round(mean_depth, 3),
    }
