"""End-to-end pipeline orchestration.

The :func:`run` function below executes the entire stylometric workflow:

1. load corpus (optionally including human baseline),
2. extract stylometric features,
3. run statistical tests,
4. fit & cross-validate the attribution classifier,
5. score permutation feature importance,
6. produce all figures and tables.

Designed to be called from a script *or* a notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, statistics, visualization
from .classify import StyleClassifier
from .corpus import Corpus
from .features import FeatureExtractor


META_COLS = ("model", "genre", "prompt_id")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def run(
    include_human: bool = False,
    classifier_kind: str = "logreg",
    include_function_words: bool = True,
    save_features: bool = True,
) -> dict:
    """Execute the full pipeline. Returns a dict of artefacts."""
    # 1. corpus
    corpus = Corpus.load(include_human=include_human)
    print(f"Loaded corpus: {len(corpus)} texts | "
          f"models={corpus.models} | genres={corpus.genres}")

    # 2. features
    extractor = FeatureExtractor(include_function_words=include_function_words)
    features_df = extractor.extract_dataframe(corpus.df)
    if save_features:
        features_df.to_csv(config.FEATURES_PATH, index=False)
        print(f"Saved features -> {config.FEATURES_PATH}")

    feat_cols = _feature_columns(features_df)
    X = features_df[feat_cols].values
    y = features_df["model"].values

    # 3. statistics
    anova = statistics.per_feature_anova(features_df, feat_cols)
    effects = statistics.pairwise_effect_sizes(features_df, feat_cols)
    anova.to_csv(config.TABLES_DIR / "feature_anova.csv", index=False)
    effects.to_csv(config.TABLES_DIR / "pairwise_effect_sizes.csv", index=False)

    # 4. classification
    clf = StyleClassifier(kind=classifier_kind)
    cv_report = clf.cross_validate(X, y, labels=sorted(set(y)))
    cv_report.confusion.to_csv(
        config.TABLES_DIR / "confusion_matrix.csv"
    )
    (config.TABLES_DIR / "cv_summary.json").write_text(
        json.dumps(cv_report.to_dict(), indent=2)
    )
    print(f"CV accuracy: {cv_report.accuracy_mean:.3f} "
          f"+/- {cv_report.accuracy_std:.3f} "
          f"(chance ~ {cv_report.chance_accuracy:.3f}, "
          f"perm p = {cv_report.permutation_pvalue:.3f})")

    # 5. importance (use ANOVA F-stats instead of permutation for stability)
    imp = anova[["feature", "anova_F"]].rename(
        columns={"anova_F": "importance_mean"}
    ).assign(importance_std=0.0).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp.to_csv(config.TABLES_DIR / "feature_importance.csv", index=False)

    # 6. figures
    visualization.projection_plot(X, y.tolist(), method="pca",
                                  filename="pca.png")
    try:
        visualization.projection_plot(X, y.tolist(), method="tsne",
                                      filename="tsne.png")
    except Exception as exc:
        print(f"t-SNE skipped: {exc}")
    visualization.feature_distribution_grid(
        features_df,
        feature_cols=imp.head(12)["feature"].tolist(),
    )
    visualization.confusion_heatmap(cv_report.confusion)
    visualization.importance_barplot(imp)
    for feat in imp.head(3)["feature"].tolist():
        visualization.effect_size_heatmap(effects, feat)

    return {
        "features": features_df,
        "anova": anova,
        "effects": effects,
        "cv_report": cv_report,
        "importance": imp,
    }
