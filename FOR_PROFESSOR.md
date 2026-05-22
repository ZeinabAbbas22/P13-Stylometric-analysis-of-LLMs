# P13 EXECUTIVE SUMMARY — For Professor

**Project:** The Aesthetics of Generation  
**Research Question:** Do Large Language Models have measurable, identifiable "writing styles"?  
**Answer:** **YES** — with 97.1% classification accuracy and 36 statistically significant features

---

## The Study

**Dataset:** 105 parallel texts from ChatGPT, Gemini, Claude (35 per model)  
**Genres:** Narrative, Argumentative, Descriptive, Dialogue, Creative (7 per genre per model)  
**Features:** 170 stylometric features (lexical, syntactic, readability, discourse, punctuation, function words)

---

## Key Finding

**Can we identify which LLM produced a text based on style alone?**

| Metric | Result |
|--------|--------|
| Classification Accuracy (5-fold CV) | **97.1% ± 3.8%** |
| Chance Baseline | 33.3% |
| Permutation Test p-value | **0.005** (statistically real) |
| Misclassifications | 3/105 texts (2.9% error rate) |

---

## Top 3 Stylistic Signatures

| Rank | Feature | ChatGPT | Gemini | Claude | F-stat |
|------|---------|---------|--------|--------|--------|
| 1 | **Em-dash frequency** | 2.2× | 1.0× | 1.3× | **50.7** |
| 2 | **Article "a" frequency** | High | Medium | Low | **39.1** |
| 3 | **Auxiliary verb ratio** | 0.156 | 0.122 | 0.131 | **29.8** |

**Interpretation:** ChatGPT uses em-dashes ~2.2× more than Gemini, creating a more "emphatic" style with clear breaks. Function words reveal deeper stylistic differences.

---

## Robustness Validation

1. **Across-Genre Testing:** Leave-one-genre-out CV = **93.3%** (vs 33% chance)  
   → Signal is robust; not overfitting to specific genres

2. **Statistical Significance:** **36/170 features** differ significantly (p<0.05, Holm-corrected)  
   → Strong feature signal, not noise

3. **Effect Sizes:** Cohen's d computed for all pairwise comparisons  
   → Top features show large practical differences (d > 0.8)

4. **Permutation Importance:** Top 20 features account for **95%** of predictive power  
   → Clear feature hierarchy; not all features matter equally

---

## Visualizations Provided

| Figure | Shows |
|--------|-------|
| `pca.png` | 2D embedding — models cluster with minimal overlap |
| `radar_profile.png` | Each model's stylometric "fingerprint" (polar plot) |
| `model_signatures.png` | Top 5 features that distinguish each model |
| `confusion_sankey.html` | Interactive: which texts misclassified & why |
| `confusion_matrix.png` | 3×3 matrix (102/105 correct) |
| `importance_lollipop.png` | Top 20 discriminative features ranked |

---

## Methodology Highlights

1. **Controlled Corpus Design:** Identical prompts across models to isolate stylistic variation from content variation

2. **Comprehensive Features:** Not just simple metrics (avg word length), but sophisticated stylometric features (MTLD, syntactic depth, dependency structure, discourse markers, Mosteller-Wallace function words)

3. **Rigorous Statistics:** 
   - ANOVA for feature significance testing
   - Kruskal-Wallis non-parametric validation
   - Holm-Bonferroni correction for multiple comparisons
   - Cohen's d effect sizes for practical significance

4. **Cross-Validation:** Proper train-test split with 5-fold CV to avoid overfitting

5. **Robustness Testing:** Leave-one-genre-out validates that signal generalizes across rhetorical contexts

---

## Interpretation & Discussion

**What do these stylistic differences reveal about the models?**

1. **ChatGPT:** Emphatic style with frequent em-dashes and structural breaks; formal, conversational tone
2. **Gemini:** Balanced narrative fluency; consistent article usage; flowing rhythm
3. **Claude:** Explicit, deliberate phrasing; discourse markers signal clear reasoning; higher lexical diversity

**Why does this matter?**
- Model fingerprinting for content attribution
- AI-content detection for academic integrity
- Understanding how architecture shapes language generation
- Potential for style anonymization / standardization

**Limitations discussed in paper:**
- Small corpus (could expand to 300+ texts)
- Only 3 commercial models (could include open-source)
- Single collection date (temporal variation untested)

---

## Files for Review

**For Quick Overview:**
- `DELIVERABLES.md` — Headline results & project structure
- `REQUIREMENTS_MAPPING.md` — How each requirement is met

**For Reproducibility:**
- `scripts/02_run_analysis.py` — Re-runs entire pipeline (5 min runtime)
- `data/corpus.csv` — Input corpus
- `data/features.csv` — Extracted features

**For Details:**
- `results/tables/` — 7 CSV/JSON tables with raw results
- `results/figures/` — 15+ visualizations (PNG + interactive HTML)

**This project demonstrates that LLMs have statistically and practically significant stylistic signatures that can be reliably identified with high accuracy. The study contributes to emerging field of AI stylistics and authorship attribution.**

