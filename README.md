# P13 — The Aesthetics of Generation

**Stylometric analysis of large language models.**
University of Milan — NLP final project (Prof. Alfio Ferrara).

---

## Research question

> *Do different large language models have measurable, identifiable
> "writing styles", and if so, what stylometric features distinguish
> them — and how do they compare to human writing?*

We answer this empirically by:

1. **Collecting** parallel text samples from three LLMs (ChatGPT,
   Gemini, Claude) across five rhetorical genres (narrative,
   argumentative, descriptive, dialogue, creative).
2. **Engineering** ~200 interpretable stylometric features per text
   (lexical, syntactic, readability, discourse, punctuation,
   function-word frequencies).
3. **Testing** whether those features differ significantly between
   models (one-way ANOVA + Kruskal–Wallis, Holm-corrected).
4. **Classifying** model authorship with cross-validated logistic
   regression / random forest / SVM and a permutation-test sanity
   check.
5. **Interpreting** the model with permutation feature importance and
   pairwise Cohen's *d* effect sizes.
6. **Stress-testing** the signal with leave-one-genre-out CV, a
   bootstrap confidence interval, and an optional style-mimicry attack.
7. **Comparing** against a Project-Gutenberg human baseline.

---

## Project layout

```
p13-stylometry/
├── data/
│   ├── corpus.csv              # LLM-generated texts (model, genre, prompt_id, text)
│   ├── human_baseline.csv      # produced by scripts/fetch_human_baseline.py
│   ├── prompts.json            # 35 prompts × 3 models = 105 max samples
│   └── features.csv            # produced by scripts/01_extract_features.py
├── src/
│   ├── config.py               # paths, constants, model registry
│   ├── corpus.py               # Corpus loader + validator
│   ├── features/               # feature-extraction subpackage
│   │   ├── lexical.py
│   │   ├── syntactic.py
│   │   ├── readability.py
│   │   ├── discourse.py
│   │   ├── punctuation.py
│   │   ├── function_words.py
│   │   └── extractor.py        # combines everything
│   ├── embeddings.py           # SBERT encoder (optional)
│   ├── statistics.py           # ANOVA, Kruskal, Cohen's d, Holm
│   ├── classify.py             # cross-validated attribution classifier
│   ├── robustness.py           # genre holdout, mimicry, bootstrap
│   ├── visualization.py        # all plots
│   └── pipeline.py             # end-to-end orchestration
├── scripts/
│   ├── 01_extract_features.py
│   ├── 02_run_analysis.py
│   ├── 03_robustness.py
│   ├── collect_responses.py    # paste helper for growing the corpus
│   └── fetch_human_baseline.py
├── notebooks/
│   └── demo.ipynb              # narrative walkthrough
├── results/
│   ├── figures/                # PNG plots
│   └── tables/                 # CSV / JSON tables
├── paper/
│   └── outline.md              # paper structure
├── requirements.txt
└── README.md
```

---

## Setup

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m spacy download en_core_web_sm
```

(Sentence-transformers and SHAP are optional — only needed for the
embedding-based and SHAP-importance analyses.)

---

## Reproducing the results

```powershell
# 1. (optional) fetch a human Project-Gutenberg baseline
py -3.12 scripts/fetch_human_baseline.py

# 2. extract features for every text
py -3.12 scripts/01_extract_features.py

# 3. run the full statistical + classification analysis
py -3.12 scripts/02_run_analysis.py --classifier logreg --include-human

# 4. robustness experiments
py -3.12 scripts/03_robustness.py
```

All figures land in `results/figures/`, all tables in
`results/tables/`.

---

## Growing the corpus

The starter `data/corpus.csv` has **15 texts** (3 models × 5 genres ×
1 prompt). `data/prompts.json` already contains **7 prompts per
genre**, so the maximum collectible dataset is **3 × 5 × 7 = 105
texts**. Use the helper to paste in more responses interactively:

```powershell
py -3.12 scripts/collect_responses.py --model chatgpt
py -3.12 scripts/collect_responses.py --model claude
py -3.12 scripts/collect_responses.py --model gemini
```

The script remembers which `prompt_id`s a given model has already
answered and skips them.

---

## Disclosures

* **Generation.** All LLM samples were collected manually from the
  official ChatGPT, Gemini, and Claude web UIs. No API keys were used.
* **AI assistance.** This project's code skeleton was scaffolded with
  GitHub Copilot in agent mode; all design decisions, prompt choices,
  data collection, and final interpretation are the author's own.
* **Human baseline.** Drawn from Project Gutenberg (public domain).

---

## Methodological notes

* The dataset is intentionally **small but parallel**: every prompt is
  posed identically to every model, so any classifier accuracy above
  chance reflects stylistic — not topical — differences.
* We report **permutation p-values** and **bootstrap confidence
  intervals** so that the reader can judge what the small sample size
  permits.
* We report **leave-one-genre-out** accuracy as a guard against the
  classifier learning topic rather than style.
* The `style_mimicry_robustness` experiment is optional: tag any
  mimicry prompts as `narrative_mimic_claude_1` etc. and the script
  will pick them up automatically.
