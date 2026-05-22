"""Readability indices (Flesch, Flesch–Kincaid, SMOG, Gunning Fog)."""
from __future__ import annotations

import textstat


def compute(text: str) -> dict[str, float]:
    """Return a dict of standard readability scores for ``text``."""
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 3),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 3),
        "gunning_fog": round(textstat.gunning_fog(text), 3),
        "smog_index": round(textstat.smog_index(text), 3),
        "dale_chall": round(textstat.dale_chall_readability_score(text), 3),
    }
