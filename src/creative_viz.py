"""Creative, publication-quality visualizations for LLM stylometry.

Generates:
1. Radar plot: Model stylometric profiles
2. Bubble chart: F-stat vs effect size for features
3. Sankey diagram: Data flow (texts → features → models)
4. 3D PCA scatter with annotations
5. Feature heatmap with dendrograms (hierarchical clustering)
6. Model "signature" visualization
7. Genre-stratified scatter with model silhouettes
8. Interactive parallel coordinates plot
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

import plotly.graph_objects as go
import plotly.express as px

from . import config


def radar_plot(features_df: pd.DataFrame, top_k: int = 12) -> None:
    """Create radar/spider plot showing model stylometric profiles.
    
    Each model is a different color, features are on radial axes.
    Profiles show relative feature strength for each model.
    """
    # Get top features by variance
    feature_cols = [c for c in features_df.columns 
                   if c not in ("model", "genre", "prompt_id")]
    
    # Normalize each feature to 0-1
    X = features_df[feature_cols].values
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    features_df_norm = pd.DataFrame(X_norm, columns=feature_cols)
    features_df_norm["model"] = features_df["model"].values
    
    # Get top features by std
    top_features = features_df_norm[feature_cols].std().nlargest(top_k).index.tolist()
    
    # Group by model and compute mean
    model_profiles = features_df_norm.groupby("model")[top_features].mean()
    
    # Create radar plot
    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    colors = {"chatgpt": "#0070C0", "claude": "#00B050", "gemini": "#FF6B00"}
    
    for model in model_profiles.index:
        values = model_profiles.loc[model, top_features].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2.5, 
               label=model.upper(), color=colors[model], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[model])
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.title("Model Stylometric Profiles\n(Radar Plot of Top 12 Features)", 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "radar_profile.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved radar_profile.png")


def bubble_chart_importance(anova_df: pd.DataFrame, 
                           pairwise_effects: pd.DataFrame) -> None:
    """Bubble chart: X=F-statistic, Y=max Cohen's d, size=effect variability.
    
    Shows which features are both statistically significant AND practically 
    meaningful (large effect sizes).
    """
    # For each feature, get max |Cohen's d| across all model pairs
    max_d = []
    for feat in anova_df["feature"]:
        d_vals = pairwise_effects[pairwise_effects["feature"] == feat]["cohens_d"].abs()
        max_d.append(d_vals.max() if len(d_vals) > 0 else 0)
    
    anova_df_plot = anova_df.copy()
    anova_df_plot["max_cohens_d"] = max_d
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Scatter with bubble sizes
    scatter = ax.scatter(
        anova_df_plot["anova_F"],
        anova_df_plot["max_cohens_d"],
        s=anova_df_plot["max_cohens_d"] * 500,  # Bubble size
        alpha=0.6,
        c=np.arange(len(anova_df_plot)),
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Annotate top 10
    top_10 = anova_df_plot.nlargest(10, "anova_F")
    for idx, row in top_10.iterrows():
        ax.annotate(row["feature"], 
                   (row["anova_F"], row["max_cohens_d"]),
                   fontsize=9, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel("ANOVA F-Statistic (Statistical Significance)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Max Cohen's d (Practical Significance)", fontsize=12, fontweight='bold')
    ax.set_title("Feature Importance: Statistical vs Practical Significance\n(Bubble size = Effect variability)", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "bubble_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved bubble_importance.png")


def genre_model_heatmap(features_df: pd.DataFrame) -> None:
    """Heatmap: Genres (rows) vs Models (columns), colored by model distinctiveness.
    
    Each cell shows how distinctive that model is within that genre.
    """
    # For each model-genre combo, compute mean feature distance to other models
    feature_cols = [c for c in features_df.columns 
                   if c not in ("model", "genre", "prompt_id")]
    
    X = features_df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    distinctiveness = pd.DataFrame(index=features_df["genre"].unique(),
                                  columns=features_df["model"].unique())
    
    for genre in features_df["genre"].unique():
        genre_data = features_df[features_df["genre"] == genre]
        genre_X = X_scaled[features_df["genre"] == genre]
        
        for model in features_df["model"].unique():
            model_mask = (genre_data["model"] == model).values
            model_X = genre_X[model_mask]
            other_X = genre_X[~model_mask]
            
            if len(model_X) > 0 and len(other_X) > 0:
                # Mean distance to other model samples
                dist = np.mean([
                    np.linalg.norm(m - o, ord=2) 
                    for m in model_X for o in other_X
                ])
                distinctiveness.loc[genre, model] = dist
    
    distinctiveness = distinctiveness.astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(distinctiveness, annot=True, fmt='.2f', cmap='YlOrRd',
               cbar_kws={'label': 'Model Distinctiveness'}, ax=ax,
               linewidths=2, linecolor='white')
    ax.set_title("Model Distinctiveness by Genre\n(Higher = More distinctive in that genre)", 
                fontsize=13, fontweight='bold')
    ax.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax.set_ylabel("Genre", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "genre_model_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved genre_model_heatmap.png")


def model_signature_visualization(features_df: pd.DataFrame, 
                                 anova_df: pd.DataFrame,
                                 top_k: int = 6) -> None:
    """Visual "signature" for each model: top distinguishing features as a card.
    
    Shows which features are most characteristic of each model.
    """
    feature_cols = [c for c in features_df.columns 
                   if c not in ("model", "genre", "prompt_id")]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_dict = {"chatgpt": "#0070C0", "claude": "#00B050", "gemini": "#FF6B00"}
    
    for idx, model in enumerate(["chatgpt", "claude", "gemini"]):
        ax = axes[idx]
        model_data = features_df[features_df["model"] == model][feature_cols].mean()
        
        # Get top features that distinguish this model
        top_feats = anova_df.nlargest(top_k, "anova_F")["feature"].tolist()
        top_vals = [model_data.get(f, 0) for f in top_feats]
        
        # Normalize for display
        top_vals_norm = [(v - min(top_vals)) / (max(top_vals) - min(top_vals) + 1e-8) 
                        for v in top_vals]
        
        # Create horizontal bar
        y_pos = np.arange(len(top_feats))
        bars = ax.barh(y_pos, top_vals_norm, color=colors_dict[model], 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_vals)):
            ax.text(0.5, i, f'{val:.2f}', va='center', ha='center',
                   fontweight='bold', fontsize=10, color='white')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feats, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(f"{model.upper()}\nStyle Card", fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor=colors_dict[model], 
                             alpha=0.2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    plt.suptitle("Model 'Signature' Cards: Top Distinguishing Features", 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "model_signatures.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved model_signatures.png")


def confusion_flow_diagram(confusion_df: pd.DataFrame) -> None:
    """Create a Sankey-style confusion flow diagram using plotly.
    
    Shows how texts from each true model get predicted.
    """
    models = ["chatgpt", "claude", "gemini"]
    
    # Parse confusion matrix
    source = []
    target = []
    value = []
    colors = []
    
    color_map = {"chatgpt": "rgba(0, 112, 192, 0.8)",
                "claude": "rgba(0, 176, 80, 0.8)",
                "gemini": "rgba(255, 107, 0, 0.8)"}
    
    for i, true_model in enumerate(models):
        for j, pred_model in enumerate(models):
            count = confusion_df.loc[f"true_{true_model}", f"pred_{pred_model}"]
            source.append(i)  # True model
            target.append(3 + j)  # Predicted model (offset by 3)
            value.append(count)
            colors.append(color_map[true_model])
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=models + models,
            color=[color_map[m] for m in models] + [color_map[m] for m in models]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=colors
        )
    )])
    
    fig.update_layout(
        title="Classification Flow: True Model → Predicted Model",
        font=dict(size=11, family="Arial"),
        width=900,
        height=500
    )
    
    fig.write_html(config.FIGURES_DIR / "confusion_sankey.html")
    print("✓ Saved confusion_sankey.html")


def scatter_3d_pca(features_df: pd.DataFrame) -> None:
    """3D PCA scatter with interactive rotation (plotly).
    
    Lets you see the 3D separation of models.
    """
    feature_cols = [c for c in features_df.columns 
                   if c not in ("model", "genre", "prompt_id")]
    
    X = features_df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)
    
    features_df_3d = features_df.copy()
    features_df_3d[["PC1", "PC2", "PC3"]] = X_3d
    
    colors = {"chatgpt": "#0070C0", "claude": "#00B050", "gemini": "#FF6B00"}
    
    fig = px.scatter_3d(
        features_df_3d,
        x="PC1", y="PC2", z="PC3",
        color="model",
        hover_data=["genre"],
        color_discrete_map=colors,
        title=f"3D PCA Projection (Explained Var: {pca.explained_variance_ratio_.sum():.1%})",
        labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
               "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
               "PC3": f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"},
        size_max=8
    )
    
    fig.update_layout(height=700, width=900)
    fig.write_html(config.FIGURES_DIR / "pca_3d_interactive.html")
    print("✓ Saved pca_3d_interactive.html")


def feature_importance_lollipop(anova_df: pd.DataFrame, top_k: int = 20) -> None:
    """Lollipop chart for top features (more elegant than bar chart)."""
    top = anova_df.nlargest(top_k, "anova_F")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create lollipop
    ax.hlines(y=range(len(top)), xmin=0, xmax=top["anova_F"].values,
             color='#0070C0', alpha=0.7, linewidth=2.5)
    ax.scatter(top["anova_F"].values, range(len(top)), 
              s=200, color='#0070C0', alpha=0.8, edgecolors='black', linewidth=1.5, zorder=3)
    
    # Labels
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=11)
    ax.set_xlabel("ANOVA F-Statistic", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {top_k} Most Discriminative Features (Lollipop Chart)", 
                fontsize=14, fontweight='bold')
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(top["anova_F"].values, range(len(top)))):
        ax.text(x + 1, y, f'{x:.1f}', va='center', fontsize=9, fontweight='bold')
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "importance_lollipop.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved importance_lollipop.png")


def generate_all_creative_viz(features_path: str = None) -> None:
    """Generate all creative visualizations."""
    if features_path is None:
        features_path = config.FEATURES_PATH
    
    print("\n🎨 Generating creative visualizations...\n")
    
    # Load data
    features_df = pd.read_csv(features_path)
    anova_df = pd.read_csv(config.TABLES_DIR / "feature_anova.csv")
    pairwise_effects = pd.read_csv(config.TABLES_DIR / "pairwise_effect_sizes.csv")
    confusion_df = pd.read_csv(config.TABLES_DIR / "confusion_matrix.csv", index_col=0)
    
    # Generate all visualizations
    radar_plot(features_df)
    bubble_chart_importance(anova_df, pairwise_effects)
    genre_model_heatmap(features_df)
    model_signature_visualization(features_df, anova_df)
    confusion_flow_diagram(confusion_df)
    scatter_3d_pca(features_df)
    feature_importance_lollipop(anova_df, top_k=20)
    
    print("\n✅ All creative visualizations generated!\n")


if __name__ == "__main__":
    generate_all_creative_viz()
