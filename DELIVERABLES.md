# P13 Stylometry Project — Final Deliverables

## Project Overview
**Research Question:** Do different large language models (ChatGPT, Gemini, Claude) have measurable, identifiable "writing styles"?

**Result:** **YES** — 97.1% classification accuracy with 36 statistically significant stylometric features.

---

## 📊 Data & Analysis

### Dataset
- **105 texts** (perfectly balanced: 35 per model, 7 per genre)
- **5 genres:** Narrative, Argumentative, Descriptive, Dialogue, Creative
- **3 models:** ChatGPT, Gemini, Claude

### Features Extracted
- **170 stylometric features** per text:
  - Lexical (7): TTR, MTLD, hapax ratio, word length stats, etc.
  - Syntactic (18): POS ratios, sentence length, dependency depth
  - Readability (5): Flesch, Flesch-Kincaid, SMOG, Gunning Fog, Dale-Chall
  - Discourse (2): Discourse markers (count & rate)
  - Punctuation (11): Frequency of all punctuation marks
  - Function words (~150): Mosteller-Wallace style word frequencies

### Key Results

| Metric | Value |
|--------|-------|
| **Classification Accuracy (CV)** | 97.1% ± 3.8% |
| **Chance Baseline** | 33.3% |
| **Permutation Test p-value** | 0.005 |
| **Significant Features (p<0.05, Holm-corrected)** | 36/170 |
| **Confusion Matrix** | 102/105 correct (3 misclassifications) |

### Top 5 Discriminative Features
1. **punct_dash** (F=50.66) — ChatGPT uses em-dashes ~2.2× more than Claude
2. **fw_a** (F=39.14) — Article "a" frequency differs significantly
3. **pos_aux_ratio** (F=29.78) — Auxiliary verb usage patterns
4. **mtld** (F=26.08) — Lexical diversity differs by model
5. **pos_adv_ratio** (F=25.02) — Adverb frequency varies

### Statistical Validation
- **ANOVA + Kruskal-Wallis** testing each feature across 3 models
- **Holm-Bonferroni correction** for multiple comparisons
- **Cohen's d effect sizes** computed for all pairwise comparisons
- **Leave-one-genre-out validation:** 93.3% mean accuracy (vs. 33% chance)

---

## 📁 Project Structure

```
p13-stylometry/
├── data/
│   ├── corpus.csv                    # 105 texts with model/genre/text
│   └── features.csv                  # 105×173 feature matrix
├── src/
│   ├── config.py                     # Central configuration
│   ├── corpus.py                     # Data loading & validation
│   ├── features/                     # Feature extraction modules
│   │   ├── lexical.py, syntactic.py, readability.py, etc.
│   │   └── extractor.py              # Main FeatureExtractor class
│   ├── statistics.py                 # ANOVA, effect sizes, corrections
│   ├── classify.py                   # Logistic Regression classifier
│   ├── pipeline.py                   # End-to-end orchestration
│   ├── visualization.py              # Standard visualizations (PCA, confusion, etc.)
│   └── creative_viz.py               # Creative publication-quality visualizations
├── scripts/
│   ├── 01_extract_features.py        # Extract all 170 features
│   ├── 02_run_analysis.py            # Full analysis pipeline
│   ├── 03_robustness.py              # Leave-one-genre-out validation
│   └── collect_responses.py          # Interactive prompt response collection
├── results/
│   ├── tables/
│   │   ├── feature_anova.csv         # All 170 features with ANOVA stats
│   │   ├── feature_importance.csv    # Top features by F-statistic
│   │   ├── confusion_matrix.csv      # 3×3 classification matrix
│   │   ├── cv_summary.json           # Cross-validation metrics
│   │   ├── pairwise_effect_sizes.csv # Cohen's d for all pairs/features
│   │   └── genre_holdout.csv         # Leave-one-genre-out results
│   └── figures/
│       ├── pca.png                   # PCA projection
│       ├── tsne.png                  # t-SNE projection
│       ├── confusion_matrix.png      # Heatmap of misclassifications
│       ├── feature_distributions.png # Boxplots of top features
│       ├── feature_importance.png    # Bar chart of F-statistics
│       ├── effect_*.png              # Cohen's d heatmaps
│       ├── radar_profile.png         # ✨ Model stylometric profiles (polar)
│       ├── bubble_importance.png     # ✨ Statistical vs practical significance
│       ├── genre_model_heatmap.png   # ✨ Model distinctiveness by genre
│       ├── model_signatures.png      # ✨ Top features per model
│       ├── confusion_sankey.html     # ✨ Interactive classification flow
│       ├── pca_3d_interactive.html   # ✨ Interactive 3D PCA
│       └── importance_lollipop.png   # ✨ Top 20 features (lollipop chart)
└── paper/
    └── paper_draft.md                # 4,500-word academic paper
```

---

## 📈 Creative Visualizations (NEW)

Seven publication-quality visualizations created to replace standard heatmaps:

| # | File | Type | Description |
|---|------|------|---|
| 1 | **radar_profile.png** | Polar Chart | Model stylometric fingerprints (top 12 features) |
| 2 | **bubble_importance.png** | Bubble Scatter | F-statistic vs Cohen's d (significance vs effect size) |
| 3 | **genre_model_heatmap.png** | Heatmap | Which genres best discriminate between models |
| 4 | **model_signatures.png** | Signature Cards | Top 6 features per model (side-by-side comparison) |
| 5 | **confusion_sankey.html** | Interactive Flow | Text classification paths (true→predicted) |
| 6 | **pca_3d_interactive.html** | 3D Scatter | Rotatable 3D PCA with model separation |
| 7 | **importance_lollipop.png** | Lollipop Chart | Top 20 discriminative features |

All PNG visualizations are 300 dpi, publication-ready. HTML files are interactive (rotatable, zoomable).

---

## 📄 Academic Paper

**File:** `paper/paper_draft.md` (4,500+ words)

### Sections:
1. **Abstract** — Executive summary of research and findings
2. **Introduction** — Motivation, related work, research gaps
3. **Methodology**
   - Corpus design (105 texts, 5 genres, 3 models)
   - Feature extraction (170 features across 6 categories)
   - Statistical testing (ANOVA, Kruskal-Wallis, Holm correction)
   - Classification approach (Logistic Regression, 5-fold CV)
4. **Results**
   - Statistical significance of features
   - Classification accuracy (97.1%)
   - Confusion matrix analysis
   - Effect sizes (Cohen's d)
   - Robustness validation (leave-one-genre-out)
5. **Discussion**
   - Interpretation of top discriminative features
   - Implications for AI detection, copyright protection, linguistics
   - Limitations (corpus size, prompt diversity, temporal variation)
   - Future work (larger corpora, temporal tracking, multilingual)
6. **Visualizations Section** — Description of all 7 figures
7. **References** — Academic citations

---

## 🚀 How to Run

### Extract Features
```bash
python scripts/01_extract_features.py
```
**Output:** `data/features.csv` (105×173 matrix)

### Run Full Analysis
```bash
python scripts/02_run_analysis.py [--classifier logreg|rf|svm]
```
**Outputs:** All tables, figures, and statistics

### Generate Creative Visualizations
```bash
python -c "from src.creative_viz import generate_all_creative_viz; generate_all_creative_viz()"
```
**Output:** 7 new PNG + HTML files in `results/figures/`

### Leave-One-Genre-Out Robustness Test
```bash
python scripts/03_robustness.py
```
**Output:** `results/tables/genre_holdout.csv` (93.3% mean accuracy)

---

## 📊 Key Findings Summary

### Stylometric Signatures Exist
- **97.1% accuracy** in model attribution far exceeds random (33%)
- Only **3 misclassifications** out of 105 texts
- Signal is **robust across genres** (93.3% leave-one-genre-out accuracy)

### ChatGPT is Most Distinctive
- **Highest em-dash usage** (~2.2× more than Claude)
- **Specific punctuation patterns** set it apart
- Strong stylometric "fingerprint"

### Features that Matter Most
1. **Punctuation** — Dash, comma, and semicolon frequencies
2. **Function words** — Articles, conjunctions, prepositions
3. **Syntax** — Auxiliary verb ratios, sentence length variability
4. **Lexical diversity** — MTLD (vocabulary maintenance metrics)

### Genre Effects
- **Narrative** and **Argumentative** texts show strongest model separation
- **Dialogue** texts are harder to classify (more natural, less distinct)
- All genres remain >85% accurate in hold-out validation

---

## 🎯 Conclusions

This project conclusively demonstrates that **different LLMs have statistically significant and practically meaningful stylometric differences**. These differences are:

- **Robust:** Consistent across genres (93%+ accuracy in hold-out validation)
- **Identifiable:** 97.1% accuracy in blind attribution testing
- **Quantifiable:** 36 significant features with interpretable effect sizes
- **Actionable:** Could be used for AI content detection, copyright protection, and authorship verification

The work also raises important questions about **adversarial robustness**: Can these signatures be deliberately obscured? This opens new research directions in LLM watermarking and style anonymization.

---

## 📞 Contact / Notes

- **Dataset:** Balanced design (35 texts per model, 7 per genre)
- **Reproducible:** All code, data, and results included
- **Publication-Ready:** Academic paper + figures meet journal standards

---

**Generated:** May 2026  
**Research Question:** Do different LLMs have measurable, identifiable writing styles?  
**Answer:** ✅ YES, with 97.1% classification accuracy.

