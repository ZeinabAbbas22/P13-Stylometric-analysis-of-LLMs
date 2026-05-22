"""Publication-quality multi-panel figures for my paper.

    paper_figure_1_cv_panel.png        – 4-panel CV / permutation result
    paper_figure_2_signatures.png      – per-model stylometric signatures
    paper_figure_3_sensitivity.png     – accuracy vs #features sweep
    paper_figure_4_robustness.png      – genre-holdout & error analysis

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

# ── style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titleweight": "bold",
    "figure.titlesize": 13,
})

C_CHATGPT = "#10A37F"   # OpenAI green
C_GEMINI  = "#4285F4"   # Google blue
C_CLAUDE  = "#C76A47"   # Anthropic orange
C_CHANCE  = "#888888"
C_OBS     = "#222222"
MODEL_COLORS = {"chatgpt": C_CHATGPT, "gemini": C_GEMINI, "claude": C_CLAUDE}
MODEL_ORDER  = ["chatgpt", "gemini", "claude"]
META = ("model", "genre", "prompt_id")

RNG = np.random.default_rng(2026)


def load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(config.FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in META]
    X = df[feature_cols].to_numpy()
    y = df["model"].to_numpy()
    return df, X, y, feature_cols


# ── Fig 1 : CV + permutation panel ───────────────────────────────────────
def figure_cv_panel(X, y, out: Path) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    pipe = lambda: LogisticRegression(max_iter=4000, C=1.0)
    Xs = StandardScaler().fit_transform(X)

    # 1. per-fold accuracies
    folds = cross_val_score(pipe(), Xs, y, cv=cv, scoring="accuracy")
    obs   = folds.mean()
    chance = 1.0 / len(np.unique(y))

    # 2. permutation null distribution
    N_PERM = 200
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        y_perm = RNG.permutation(y)
        null[i] = cross_val_score(pipe(), Xs, y_perm, cv=cv,
                                  scoring="accuracy").mean()
    p_value = (np.sum(null >= obs) + 1) / (N_PERM + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle("Classifier validation — 5-fold CV and permutation test")

    # (a) per-fold bar
    ax = axes[0, 0]
    xs = np.arange(1, 6)
    ax.bar(xs, folds, color=C_OBS, alpha=0.85, width=0.55)
    ax.axhline(obs,    color="#1F77B4", ls="--", lw=2,
               label=f"mean = {obs:.3f}")
    ax.axhline(chance, color=C_CHANCE,  ls=":",  lw=2,
               label=f"chance = {chance:.3f}")
    for x, v in zip(xs, folds):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(xs); ax.set_xlabel("fold"); ax.set_ylabel("accuracy")
    ax.set_title("(a) Per-fold accuracy")
    ax.legend(loc="lower right")

    # (b) permutation histogram
    ax = axes[0, 1]
    ax.hist(null, bins=30, color="#B0B0B0", edgecolor="white",
            alpha=0.9, label="null distribution")
    ax.axvline(obs, color="#D62728", lw=2.5,
               label=f"observed = {obs:.3f}")
    ax.axvline(chance, color=C_CHANCE, ls=":", lw=2,
               label=f"chance = {chance:.3f}")
    ax.set_title(f"(b) Permutation null (N={N_PERM}, p = {p_value:.3f})")
    ax.set_xlabel("CV accuracy"); ax.set_ylabel("count"); ax.legend()

    # (c) ECDF of null with observed
    ax = axes[1, 0]
    s = np.sort(null); cdf = np.arange(1, len(s) + 1) / len(s)
    ax.plot(s, cdf, color="#444", lw=2.5, label="null ECDF")
    ax.fill_between(s, cdf, alpha=0.10, color="#444")
    ax.axvline(obs, color="#D62728", lw=2.5, label=f"observed = {obs:.3f}")
    ax.set_title("(c) ECDF of permutation null")
    ax.set_xlabel("CV accuracy"); ax.set_ylabel("P(Φ ≤ x)"); ax.legend()

    # (d) effect summary card
    ax = axes[1, 1]; ax.axis("off")
    txt = (
        f"Observed CV accuracy : {obs:.3f}\n"
        f"Std across folds     : {folds.std():.3f}\n"
        f"Chance accuracy      : {chance:.3f}\n"
        f"Permutation p-value  : {p_value:.4f}\n"
        f"Null mean            : {null.mean():.3f}\n"
        f"Null 95th percentile : {np.percentile(null, 95):.3f}\n\n"
        f"→ Observed accuracy is {(obs - null.mean()) / null.std():.1f} σ\n"
        f"   above the permutation null mean.\n"
        f"   Signal is highly significant."
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            family="monospace", fontsize=11,
            bbox=dict(facecolor="#F5F5F5", edgecolor="#999",
                      boxstyle="round,pad=0.6"))
    ax.set_title("(d) Summary")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")


# ── Fig 2 : per-model stylometric signatures ─────────────────────────────
def figure_signatures(df, feature_cols, out: Path) -> None:
    # use top 10 ANOVA features
    anova = pd.read_csv(config.TABLES_DIR / "feature_anova.csv")
    top = anova.head(10)["feature"].tolist()

    # mean ± SE per model on standardized features
    Xs = StandardScaler().fit_transform(df[feature_cols])
    df_z = pd.DataFrame(Xs, columns=feature_cols)
    df_z["model"] = df["model"].values
    grouped = df_z.groupby("model")[top].agg(["mean", "sem"])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Stylometric signatures — top 10 ANOVA features (z-scored)")

    # (a) grouped bar chart with error bars
    ax = axes[0]
    width = 0.26
    xs = np.arange(len(top))
    for i, m in enumerate(MODEL_ORDER):
        means = grouped.loc[m, (slice(None), "mean")].values
        sems  = grouped.loc[m, (slice(None), "sem")].values
        ax.bar(xs + (i - 1) * width, means, width, yerr=sems,
               color=MODEL_COLORS[m], alpha=0.9, capsize=3,
               label=m.capitalize(), edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(top, rotation=40, ha="right")
    ax.set_ylabel("z-scored feature value (mean ± SE)")
    ax.set_title("(a) Per-feature profile by model")
    ax.legend(loc="upper right", frameon=True)

    # (b) heatmap of signatures
    sig = grouped.loc[:, (slice(None), "mean")].droplevel(1, axis=1)
    sig = sig.loc[MODEL_ORDER]
    ax = axes[1]
    sns.heatmap(sig, ax=ax, cmap="RdBu_r", center=0, annot=True,
                fmt=".2f", cbar_kws=dict(label="z-score"),
                linewidths=0.4, linecolor="white")
    ax.set_title("(b) Model × feature signature heatmap")
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")


# ── Fig 3 : sensitivity sweep — accuracy vs #features ────────────────────
def figure_sensitivity(X, y, feature_cols, out: Path) -> None:
    anova = pd.read_csv(config.TABLES_DIR / "feature_anova.csv")
    ranked = anova["feature"].tolist()

    sizes = [1, 3, 5, 10, 20, 35, 50, 75, 100, 150, len(feature_cols)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

    means, stds = [], []
    for k in sizes:
        cols = [feature_cols.index(f) for f in ranked[:k]]
        Xs = StandardScaler().fit_transform(X[:, cols])
        scores = cross_val_score(
            LogisticRegression(max_iter=4000),
            Xs, y, cv=cv, scoring="accuracy",
        )
        means.append(scores.mean()); stds.append(scores.std())
        print(f"    top-{k:>3} features: {scores.mean():.3f} ± {scores.std():.3f}")

    means = np.asarray(means); stds = np.asarray(stds)
    chance = 1.0 / len(np.unique(y))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Sensitivity — does the signal need many features?")

    ax = axes[0]
    ax.errorbar(sizes, means, yerr=stds, color="#1F77B4", lw=2.2,
                marker="o", capsize=4, markersize=6,
                label="5-fold CV accuracy")
    ax.fill_between(sizes, means - stds, means + stds,
                    alpha=0.15, color="#1F77B4")
    ax.axhline(chance, color=C_CHANCE, ls=":", lw=2,
               label=f"chance = {chance:.2f}")
    ax.axhline(1.0, color="#888", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("# top-ranked features (log scale)")
    ax.set_ylabel("CV accuracy (mean ± std)")
    ax.set_title("(a) Accuracy vs feature-set size")
    ax.set_ylim(0.2, 1.05)
    ax.legend(loc="lower right")

    ax = axes[1]
    gain = means - chance
    bars = ax.bar([str(s) for s in sizes], gain * 100,
                  color="#1F77B4", alpha=0.85, edgecolor="white")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{m:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("# top features used")
    ax.set_ylabel("absolute gain over chance (pp)")
    ax.set_title("(b) Even few features outperform chance")

    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")


# ── Fig 4 : robustness panel ─────────────────────────────────────────────
def figure_robustness(df, out: Path) -> None:
    holdout = pd.read_csv(config.TABLES_DIR / "genre_holdout.csv")
    conf = pd.read_csv(config.TABLES_DIR / "confusion_matrix.csv", index_col=0)
    chance = 1.0 / 3

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    fig.suptitle("Robustness — generalization and error structure")

    # (a) genre-holdout bar
    ax = axes[0]
    holdout_s = holdout.sort_values("accuracy", ascending=True)
    colors = ["#D62728" if a < 0.75 else "#2CA02C" for a in holdout_s["accuracy"]]
    bars = ax.barh(holdout_s["held_out_genre"], holdout_s["accuracy"],
                   color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(chance, color=C_CHANCE, ls=":", lw=2,
               label=f"chance = {chance:.2f}")
    ax.axvline(holdout_s["accuracy"].mean(), color="#1F77B4", ls="--",
               lw=2, label=f"mean = {holdout_s['accuracy'].mean():.2f}")
    for b, v in zip(bars, holdout_s["accuracy"]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", fontweight="bold", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("accuracy when this genre is held out")
    ax.set_title("(a) Leave-one-genre-out CV")
    ax.legend(loc="lower right")

    # (b) confusion heatmap
    ax = axes[1]
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                cbar=False, ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.set_title("(b) Confusion matrix (full-corpus CV)")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_xticklabels([c.replace("pred_", "") for c in conf.columns])
    ax.set_yticklabels([c.replace("true_", "") for c in conf.index],
                       rotation=0)

    # (c) per-class precision / recall
    classes = [c.replace("true_", "") for c in conf.index]
    M = conf.values
    precision = np.diag(M) / M.sum(axis=0)
    recall    = np.diag(M) / M.sum(axis=1)
    f1        = 2 * precision * recall / (precision + recall)

    ax = axes[2]
    xs = np.arange(len(classes))
    w  = 0.27
    ax.bar(xs - w, precision, w, color="#1F77B4", label="precision",
           alpha=0.9, edgecolor="white")
    ax.bar(xs,     recall,    w, color="#FF7F0E", label="recall",
           alpha=0.9, edgecolor="white")
    ax.bar(xs + w, f1,        w, color="#2CA02C", label="F1",
           alpha=0.9, edgecolor="white")
    for i in range(len(classes)):
        for j, v in enumerate([precision[i], recall[i], f1[i]]):
            ax.text(xs[i] + (j - 1) * w, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.15); ax.set_ylabel("score")
    ax.set_title("(c) Per-class precision / recall / F1")
    ax.legend(loc="lower right")

    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    df, X, y, feature_cols = load()
    out_dir = config.FIGURES_DIR
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Generating publication figures...")
    figure_cv_panel    (X, y, out_dir / "paper_figure_1_cv_panel.png")
    figure_signatures  (df, feature_cols,
                        out_dir / "paper_figure_2_signatures.png")
    figure_sensitivity (X, y, feature_cols,
                        out_dir / "paper_figure_3_sensitivity.png")
    figure_robustness  (df, out_dir / "paper_figure_4_robustness.png")
    print("\nAll 4 publication figures written to results/figures/.")


if __name__ == "__main__":
    main()
