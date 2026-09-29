"""
Phase 2 Configuration
Vedic Shloka Intelligence — Model Development and Transfer Learning Pipeline
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR      = PROJECT_ROOT / "data" / "training"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
LOGS_DIR      = PROJECT_ROOT / "logs"

VEDA_MODEL_DIR   = MODELS_DIR / "veda_classifier"
DOMAIN_MODEL_DIR = MODELS_DIR / "domain_classifier"
BRANCH_MODEL_DIR = MODELS_DIR / "branch_classifier"
VERSION_REGISTRY = MODELS_DIR / "version_registry.json"

# ── Pretrained backbone ────────────────────────────────────────────────────
# IndicBERT — multilingual model covering Sanskrit/Indic scripts
# Falls back to multilingual-BERT if IndicBERT unavailable offline
PRETRAINED_MODEL_NAME = os.getenv(
    "PRETRAINED_MODEL", "ai4bharat/indic-bert"
)
FALLBACK_MODEL_NAME = "google-bert/bert-base-multilingual-cased"

# ── Tokeniser ─────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 256
PADDING        = True
TRUNCATION     = True

# ── Training hyperparameters ──────────────────────────────────────────────
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5
EPOCHS         = 5
DROPOUT        = 0.1
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1      # fraction of total steps for LR warm-up
GRAD_CLIP      = 1.0
RANDOM_SEED    = 42

# ── Label definitions ──────────────────────────────────────────────────────
VEDA_LABELS = [
    "Rigveda", "Samaveda", "Yajurveda", "Atharvaveda", "Other", "Unknown"
]

DOMAIN_LABELS = [
    "Mathematics", "Astronomy", "Ayurveda", "Philosophy",
    "Ritual", "Grammar", "Linguistics", "Cosmology", "Ethics",
    "Medicine", "Other", "Unknown"
]

BRANCH_LABELS = [
    "Geometry", "Algebra", "Arithmetic", "Trigonometry",
    "Number Theory", "Combinatorics", "Jyotisha", "Vyakarana",
    "Vedanta", "Mimamsa", "Chandas", "Kalpa", "Other", "Unknown"
]

# ── Task registry ──────────────────────────────────────────────────────────
# Maps task_name → (target_column, labels_list, model_dir)
TASKS = {
    "veda": {
        "column":    "veda",
        "labels":    VEDA_LABELS,
        "model_dir": VEDA_MODEL_DIR,
    },
    "domain": {
        "column":    "domain",
        "labels":    DOMAIN_LABELS,
        "model_dir": DOMAIN_MODEL_DIR,
    },
    "branch": {
        "column":    "branch",
        "labels":    BRANCH_LABELS,
        "model_dir": BRANCH_MODEL_DIR,
    },
}
