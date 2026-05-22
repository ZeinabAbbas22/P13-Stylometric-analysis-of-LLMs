"""Main feature-extraction orchestrator.

Combines lexical, syntactic, readability, discourse, punctuation and
function-word features into a single flat dict / DataFrame row.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import spacy
from tqdm import tqdm

from . import (
    discourse,
    function_words,
    lexical,
    punctuation,
    readability,
    syntactic,
)


class FeatureExtractor:
    """Extract a rich stylometric feature vector for each text.

    Parameters
    ----------
    spacy_model :
        Name of the spaCy model to load. Default: ``en_core_web_sm``.
    include_function_words :
        If False, the (large) function-word block is omitted. Useful for
        producing a compact "interpretable" feature set.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        include_function_words: bool = True,
    ) -> None:
        self.nlp = spacy.load(spacy_model)
        self.include_function_words = include_function_words

    # ── single-text API ────────────────────────────────────────────────
    def extract(self, text: str) -> dict[str, float]:
        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc if t.is_alpha]

        feats: dict[str, float] = {}
        feats.update(lexical.compute(tokens))
        feats.update(syntactic.compute(doc))
        feats.update(readability.compute(text))
        feats.update(discourse.compute(text, n_words=len(tokens)))
        feats.update(punctuation.compute(text))
        if self.include_function_words:
            feats.update(function_words.compute(tokens))
        return feats

    # ── batch API ──────────────────────────────────────────────────────
    def extract_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        progress: bool = True,
    ) -> pd.DataFrame:
        """Run :meth:`extract` over each row of ``df`` and join the results.

        The returned DataFrame preserves the original metadata columns
        (everything except ``text_col``) and appends one column per feature.
        """
        rows: list[dict[str, float]] = []
        iterator: Iterable = df[text_col].tolist()
        if progress:
            iterator = tqdm(iterator, desc="extracting features")
        for text in iterator:
            rows.append(self.extract(text))
        feature_df = pd.DataFrame(rows)
        meta = df.drop(columns=[text_col]).reset_index(drop=True)
        return pd.concat([meta, feature_df], axis=1)
