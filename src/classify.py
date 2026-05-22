"""Authorship-attribution classifier for LLM stylometry.

Wraps several scikit-learn estimators behind a uniform interface and
provides cross-validated evaluation, a permutation-test sanity check
and permutation-importance based feature ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def make_classifier(kind: str = "logreg") -> Pipeline:
    """Return a standardising pipeline for the requested classifier kind."""
    kind = kind.lower()
    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            random_state=config.RANDOM_SEED,
        )
    elif kind in {"rf", "randomforest"}:
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=config.RANDOM_SEED,
        )
    elif kind == "svm":
        clf = SVC(kernel="rbf", probability=True,
                  random_state=config.RANDOM_SEED)
    else:
        raise ValueError(f"Unknown classifier: {kind}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


@dataclass
class CVReport:
    """Container for cross-validated evaluation results."""

    accuracy_mean: float
    accuracy_std: float
    fold_scores: np.ndarray
    chance_accuracy: float
    permutation_pvalue: float
    confusion: pd.DataFrame
    classification_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "fold_scores": self.fold_scores.tolist(),
            "chance_accuracy": self.chance_accuracy,
            "permutation_pvalue": self.permutation_pvalue,
        }


class StyleClassifier:
    """High-level interface for model-attribution experiments."""

    def __init__(self, kind: str = "logreg", n_splits: int = 5) -> None:
        self.kind = kind
        self.n_splits = n_splits
        self.pipeline = make_classifier(kind)

    # ── core evaluation ────────────────────────────────────────────────
    def cross_validate(
        self,
        X: np.ndarray,
        y: Sequence[str],
        labels: Sequence[str] | None = None,
    ) -> CVReport:
        y = np.asarray(y)
        cv = StratifiedKFold(
            n_splits=min(self.n_splits, _min_class_count(y)),
            shuffle=True,
            random_state=config.RANDOM_SEED,
        )

        scores = cross_val_score(self.pipeline, X, y, cv=cv,
                                 scoring="accuracy")
        preds = cross_val_predict(self.pipeline, X, y, cv=cv)

        chance, pval = self._permutation_test(X, y, cv)

        all_labels = labels if labels is not None else sorted(set(y))
        cm = pd.DataFrame(
            confusion_matrix(y, preds, labels=all_labels),
            index=[f"true_{l}" for l in all_labels],
            columns=[f"pred_{l}" for l in all_labels],
        )
        report = classification_report(y, preds, labels=all_labels,
                                       output_dict=True, zero_division=0)

        return CVReport(
            accuracy_mean=float(scores.mean()),
            accuracy_std=float(scores.std()),
            fold_scores=scores,
            chance_accuracy=float(chance),
            permutation_pvalue=float(pval),
            confusion=cm,
            classification_report=report,
        )

    def _permutation_test(self, X, y, cv) -> tuple[float, float]:
        try:
            score, perm_scores, pval = permutation_test_score(
                self.pipeline, X, y, cv=cv, n_permutations=200,
                random_state=config.RANDOM_SEED, n_jobs=1,
            )
            return float(perm_scores.mean()), float(pval)
        except Exception:
            return 1.0 / len(set(y)), 1.0

    # ── interpretability ───────────────────────────────────────────────
    def permutation_importance(
        self,
        X: np.ndarray,
        y: Sequence[str],
        feature_names: Sequence[str],
        n_repeats: int = 30,
    ) -> pd.DataFrame:
        self.pipeline.fit(X, y)
        result = permutation_importance(
            self.pipeline, X, y, n_repeats=n_repeats,
            random_state=config.RANDOM_SEED,
        )
        return (pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True))


def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(2, int(counts.min()))
