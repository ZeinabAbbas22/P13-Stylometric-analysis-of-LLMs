"""Corpus management: loading, validation, and combining model + human data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


REQUIRED_COLUMNS = ("model", "genre", "prompt_id", "text")


@dataclass
class Corpus:
    """Container for a stylometry corpus.

    The underlying ``DataFrame`` must contain at least the columns
    ``model``, ``genre``, ``prompt_id`` and ``text``.
    """

    df: pd.DataFrame

    @classmethod
    def load(cls, path: Path | str = config.CORPUS_PATH,
             include_human: bool = False) -> "Corpus":
        """Load corpus from CSV. Optionally append the human baseline."""
        df = pd.read_csv(path)
        cls._validate(df)
        if include_human and config.HUMAN_BASELINE_PATH.exists():
            human = pd.read_csv(config.HUMAN_BASELINE_PATH)
            cls._validate(human)
            df = pd.concat([df, human], ignore_index=True)
        return cls(df=df.reset_index(drop=True))

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Corpus missing required columns: {missing}")
        if df["text"].isna().any() or (df["text"].str.len() == 0).any():
            raise ValueError("Corpus contains empty texts.")

    # ── Convenience accessors ──────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    @property
    def models(self) -> list[str]:
        return sorted(self.df["model"].unique().tolist())

    @property
    def genres(self) -> list[str]:
        return sorted(self.df["genre"].unique().tolist())

    def summary(self) -> pd.DataFrame:
        """Cross-tabulation of model × genre counts."""
        return (self.df.groupby(["model", "genre"]).size()
                .unstack(fill_value=0))

    def texts(self) -> list[str]:
        return self.df["text"].tolist()

    def labels(self) -> list[str]:
        return self.df["model"].tolist()
