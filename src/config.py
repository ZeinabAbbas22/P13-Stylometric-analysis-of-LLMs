"""Project configuration: paths, constants, model registry."""
from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

CORPUS_PATH: Path = DATA_DIR / "corpus.csv"
FEATURES_PATH: Path = DATA_DIR / "features.csv"
EMBEDDINGS_PATH: Path = DATA_DIR / "embeddings.npy"
HUMAN_BASELINE_PATH: Path = DATA_DIR / "human_baseline.csv"
PROMPTS_PATH: Path = DATA_DIR / "prompts.json"

for _d in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────
MODELS = ("chatgpt", "gemini", "claude")
HUMAN_LABEL = "human"

MODEL_COLORS = {
    "chatgpt": "#10a37f",
    "claude": "#cc785c",
    "gemini": "#4285f4",
    "human": "#555555",
}

GENRES = ("narrative", "argumentative", "descriptive", "dialogue", "creative")

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Embeddings ─────────────────────────────────────────────────────────
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Function-word list (classic Mosteller–Wallace + standard English) ──
FUNCTION_WORDS = (
    "a", "about", "above", "after", "again", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
)

DISCOURSE_MARKERS = (
    "however", "furthermore", "therefore", "moreover", "nevertheless",
    "although", "additionally", "consequently", "meanwhile", "instead",
    "ultimately", "finally", "indeed", "thus", "hence", "whereas",
    "nonetheless", "specifically", "notably", "in fact", "in addition",
    "on the other hand", "as a result", "for instance", "for example",
)
