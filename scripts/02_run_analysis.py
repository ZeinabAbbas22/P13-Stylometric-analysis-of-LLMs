"""Run the full statistical + classification analysis and produce
all figures and tables under ``results/``.

Usage::

    py -3.12 scripts/02_run_analysis.py [--classifier logreg|rf|svm]
                                        [--include-human]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", default="logreg",
                   choices=["logreg", "rf", "svm"])
    p.add_argument("--include-human", action="store_true")
    p.add_argument("--no-function-words", action="store_true",
                   help="Skip the ~150-dim function-word block.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        include_human=args.include_human,
        classifier_kind=args.classifier,
        include_function_words=not args.no_function_words,
    )


if __name__ == "__main__":
    main()
