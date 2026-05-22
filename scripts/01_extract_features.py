"""Extract stylometric features from the corpus and save to CSV.

Usage::

    py -3.12 scripts/01_extract_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.corpus import Corpus  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402


def main() -> None:
    include_human = config.HUMAN_BASELINE_PATH.exists()
    corpus = Corpus.load(include_human=include_human)
    print(f"Loaded {len(corpus)} texts "
          f"(human baseline: {'yes' if include_human else 'no'})")
    extractor = FeatureExtractor()
    df = extractor.extract_dataframe(corpus.df)
    df.to_csv(config.FEATURES_PATH, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> "
          f"{config.FEATURES_PATH}")


if __name__ == "__main__":
    main()
