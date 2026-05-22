"""Statistical comparison of stylometric features across models.

Provides:

* per-feature one-way ANOVA across models,
* Kruskal–Wallis non-parametric counterpart,
* pairwise Cohen's d effect sizes,
* Holm–Bonferroni correction for multiple comparisons.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-variance Cohen's d between two samples."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def per_feature_anova(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "model",
) -> pd.DataFrame:
    """Run a one-way ANOVA + Kruskal–Wallis per feature."""
    groups = df[group_col].unique()
    rows = []
    for feat in feature_cols:
        samples = [df.loc[df[group_col] == g, feat].dropna().values
                   for g in groups]
        if any(len(s) < 2 for s in samples):
            continue
        try:
            f_stat, f_p = stats.f_oneway(*samples)
        except Exception:
            f_stat, f_p = np.nan, np.nan
        try:
            h_stat, h_p = stats.kruskal(*samples)
        except Exception:
            h_stat, h_p = np.nan, np.nan
        rows.append({
            "feature": feat,
            "anova_F": f_stat,
            "anova_p": f_p,
            "kruskal_H": h_stat,
            "kruskal_p": h_p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["anova_p_holm"] = _holm(out["anova_p"].values)
        out["kruskal_p_holm"] = _holm(out["kruskal_p"].values)
        out = out.sort_values("anova_p").reset_index(drop=True)
    return out


def pairwise_effect_sizes(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "model",
) -> pd.DataFrame:
    """Cohen's d for every (model_a, model_b, feature) triple."""
    groups = sorted(df[group_col].unique())
    rows = []
    for a, b in combinations(groups, 2):
        for feat in feature_cols:
            va = df.loc[df[group_col] == a, feat].dropna().values
            vb = df.loc[df[group_col] == b, feat].dropna().values
            rows.append({
                "model_a": a,
                "model_b": b,
                "feature": feat,
                "cohens_d": cohens_d(va, vb),
                "mean_a": float(np.mean(va)) if len(va) else np.nan,
                "mean_b": float(np.mean(vb)) if len(vb) else np.nan,
            })
    return pd.DataFrame(rows)


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni step-down correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, pvals[idx] * (n - rank))
        adjusted[idx] = min(running, 1.0)
    return adjusted
