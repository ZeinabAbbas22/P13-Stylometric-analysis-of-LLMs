"""Robustness experiments.

* Leave-one-genre-out cross-validation (does the stylistic signal
  generalise across genres, or does it just encode topic?).
* Style-mimicry attack (if mimicry texts are present in the corpus).
* Bootstrap CI for the cross-validated accuracy.

Usage::

    py -3.12 scripts/03_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.robustness import (  # noqa: E402
    bootstrap_accuracy_ci,
    genre_holdout,
    style_mimicry_robustness,
)


META_COLS = ("model", "genre", "prompt_id")


def main() -> None:
    if not config.FEATURES_PATH.exists():
        sys.exit(f"Run scripts/01_extract_features.py first "
                 f"({config.FEATURES_PATH} missing).")
    df = pd.read_csv(config.FEATURES_PATH)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    print("-> Leave-one-genre-out cross-validation:")
    holdout = genre_holdout(df, feat_cols)
    print(holdout.to_string(index=False))
    holdout.to_csv(config.TABLES_DIR / "genre_holdout.csv", index=False)

    print("\n-> Bootstrap CI for CV accuracy:")
    ci = bootstrap_accuracy_ci(df, feat_cols, n_boot=100)
    print(json.dumps(ci, indent=2))
    (config.TABLES_DIR / "bootstrap_ci.json").write_text(
        json.dumps(ci, indent=2)
    )

    print("\n-> Style-mimicry attack:")
    mimic = style_mimicry_robustness(df, feat_cols)
    if mimic.empty:
        print("  (no mimicry texts found — add prompts like "
              "`narrative_mimic_claude_1` to corpus.csv to run this test)")
    else:
        print(mimic.to_string(index=False))
        mimic.to_csv(config.TABLES_DIR / "style_mimicry.csv", index=False)


if __name__ == "__main__":
    main()
