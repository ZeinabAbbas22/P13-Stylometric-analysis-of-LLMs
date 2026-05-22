"""Plotting utilities. All figures share the model colour palette
defined in :data:`src.config.MODEL_COLORS` and save under
``results/figures``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from . import config


sns.set_theme(style="whitegrid", context="paper")


def _palette(models: list[str]) -> dict:
    return {m: config.MODEL_COLORS.get(m, "#333333") for m in models}


def _save(fig, name: str) -> Path:
    out = config.FIGURES_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ── projections ────────────────────────────────────────────────────────
def projection_plot(
    X: np.ndarray,
    labels: list[str],
    method: str = "pca",
    title: str | None = None,
    filename: str | None = None,
) -> Path:
    method = method.lower()
    X_std = StandardScaler().fit_transform(X)
    if method == "pca":
        reducer = PCA(n_components=2, random_state=config.RANDOM_SEED)
        coords = reducer.fit_transform(X_std)
        xlab, ylab = "PC1", "PC2"
    elif method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=min(5, max(2, len(labels) // 3)),
            random_state=config.RANDOM_SEED,
            init="pca",
        )
        coords = reducer.fit_transform(X_std)
        xlab, ylab = "t-SNE 1", "t-SNE 2"
    elif method == "umap":
        import umap  # lazy
        reducer = umap.UMAP(
            n_neighbors=min(5, max(2, len(labels) // 3)),
            random_state=config.RANDOM_SEED,
        )
        coords = reducer.fit_transform(X_std)
        xlab, ylab = "UMAP 1", "UMAP 2"
    else:
        raise ValueError(method)

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "model": labels})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df, x="x", y="y", hue="model",
        palette=_palette(df["model"].unique().tolist()),
        s=80, ax=ax,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title or f"{method.upper()} projection of stylistic space")
    return _save(fig, filename or f"projection_{method}.png")


# ── feature distributions ─────────────────────────────────────────────
def feature_distribution_grid(
    df: pd.DataFrame,
    feature_cols: list[str],
    filename: str = "feature_distributions.png",
    max_features: int = 12,
) -> Path:
    feats = feature_cols[:max_features]
    ncols = 3
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax, feat in zip(axes, feats):
        sns.boxplot(
            data=df, x="model", y=feat, hue="model",
            palette=_palette(sorted(df["model"].unique().tolist())),
            ax=ax, legend=False,
        )
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, filename)


# ── confusion matrix ──────────────────────────────────────────────────
def confusion_heatmap(
    cm: pd.DataFrame,
    title: str = "Classifier confusion matrix",
    filename: str = "confusion_matrix.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    return _save(fig, filename)


# ── feature importance bar plot ───────────────────────────────────────
def importance_barplot(
    importance_df: pd.DataFrame,
    top_k: int = 20,
    filename: str = "feature_importance.png",
) -> Path:
    top = importance_df.head(top_k)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(top))))
    sns.barplot(
        data=top, y="feature", x="importance_mean",
        color="steelblue", ax=ax,
    )
    ax.errorbar(
        x=top["importance_mean"], y=np.arange(len(top)),
        xerr=top["importance_std"], fmt="none", color="black", capsize=2,
    )
    ax.set_title(f"Top-{top_k} features by permutation importance")
    ax.set_xlabel("Mean accuracy drop")
    ax.set_ylabel("")
    return _save(fig, filename)


# ── pairwise effect-size heatmap ──────────────────────────────────────
def effect_size_heatmap(
    effect_df: pd.DataFrame,
    feature: str,
    filename: str | None = None,
) -> Path:
    sub = effect_df[effect_df["feature"] == feature]
    models = sorted(set(sub["model_a"]).union(sub["model_b"]))
    mat = pd.DataFrame(0.0, index=models, columns=models)
    for _, r in sub.iterrows():
        mat.loc[r["model_a"], r["model_b"]] = r["cohens_d"]
        mat.loc[r["model_b"], r["model_a"]] = -r["cohens_d"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(mat, annot=True, fmt=".2f", center=0, cmap="RdBu_r", ax=ax)
    ax.set_title(f"Cohen's d — {feature}")
    return _save(fig, filename or f"effect_{feature}.png")
