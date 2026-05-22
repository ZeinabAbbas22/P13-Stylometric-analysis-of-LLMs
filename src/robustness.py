"""Robustness experiments.

These probe **how stable** the stylometric signal is under realistic
perturbations of the generation conditions:

* :func:`genre_holdout` — cross-genre transfer: train on N−1 genres,
  test on the held-out genre. If the signature is *robust*, accuracy
  should remain well above chance.

* :func:`style_mimicry_robustness` — given a corpus that includes
  texts produced with explicit "write in the style of X" prompts,
  measure how often the classifier is fooled. Requires the user to
  collect such texts and tag them with ``prompt_id`` containing the
  substring ``mimic_<target>``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import StyleClassifier


def genre_holdout(
    df: pd.DataFrame,
    feature_cols: list[str],
    classifier_kind: str = "logreg",
) -> pd.DataFrame:
    """Leave-one-genre-out cross-validation."""
    rows = []
    for held in sorted(df["genre"].unique()):
        train = df[df["genre"] != held]
        test = df[df["genre"] == held]
        if train["model"].nunique() < 2 or len(test) == 0:
            continue
        clf = StyleClassifier(kind=classifier_kind)
        clf.pipeline.fit(train[feature_cols].values, train["model"].values)
        acc = clf.pipeline.score(
            test[feature_cols].values, test["model"].values
        )
        rows.append({
            "held_out_genre": held,
            "n_train": len(train),
            "n_test": len(test),
            "accuracy": float(acc),
            "chance": 1.0 / train["model"].nunique(),
        })
    return pd.DataFrame(rows)


def style_mimicry_robustness(
    df: pd.DataFrame,
    feature_cols: list[str],
    classifier_kind: str = "logreg",
) -> pd.DataFrame:
    """Evaluate attribution accuracy on mimicry texts.

    Rows where ``prompt_id`` contains ``mimic_<target>`` are interpreted
    as: model ``model`` was asked to imitate ``target``. The classifier
    is trained on the non-mimicry texts and queried on the mimicry ones;
    we report how often the classifier is fooled into predicting
    ``target`` instead of the true ``model``.
    """
    is_mimic = df["prompt_id"].str.contains("mimic_", na=False)
    if not is_mimic.any():
        return pd.DataFrame(columns=["true_model", "target",
                                     "n", "predicted_true",
                                     "predicted_target", "predicted_other"])

    train = df[~is_mimic]
    test = df[is_mimic].copy()
    test["target"] = test["prompt_id"].str.extract(r"mimic_([a-z]+)")

    clf = StyleClassifier(kind=classifier_kind)
    clf.pipeline.fit(train[feature_cols].values, train["model"].values)
    preds = clf.pipeline.predict(test[feature_cols].values)
    test["prediction"] = preds

    rows = []
    for (true_m, target), g in test.groupby(["model", "target"]):
        rows.append({
            "true_model": true_m,
            "target": target,
            "n": len(g),
            "predicted_true": int((g["prediction"] == true_m).sum()),
            "predicted_target": int((g["prediction"] == target).sum()),
            "predicted_other": int(
                ((g["prediction"] != true_m) & (g["prediction"] != target)).sum()
            ),
        })
    return pd.DataFrame(rows)


def bootstrap_accuracy_ci(
    df: pd.DataFrame,
    feature_cols: list[str],
    classifier_kind: str = "logreg",
    n_boot: int = 200,
    rng_seed: int = 0,
) -> dict:
    """Bootstrap a 95% CI for cross-validated accuracy (no permutation test)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from .classify import make_classifier
    from . import config as _cfg

    rng = np.random.default_rng(rng_seed)
    X = df[feature_cols].values
    y = df["model"].values
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        # Skip resamples where any class has fewer than 2 instances (CV fails).
        _, counts = np.unique(ys, return_counts=True)
        if counts.min() < 2:
            continue
        try:
            pipe = make_classifier(classifier_kind)
            cv = StratifiedKFold(
                n_splits=min(3, int(counts.min())),
                shuffle=True, random_state=_cfg.RANDOM_SEED,
            )
            scores = cross_val_score(pipe, X[idx], ys, cv=cv,
                                     scoring="accuracy")
            accs.append(float(scores.mean()))
        except Exception:
            continue
    accs = np.array(accs)
    return {
        "mean": float(accs.mean()) if len(accs) else float("nan"),
        "ci_low": float(np.percentile(accs, 2.5)) if len(accs) else float("nan"),
        "ci_high": float(np.percentile(accs, 97.5)) if len(accs) else float("nan"),
        "n_successful_resamples": int(len(accs)),
    }
