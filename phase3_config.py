"""
Phase 3 Configuration
Vedic Shloka Intelligence — Hybrid Retrieval, Knowledge Graph & Reasoning Engine
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# ── Directory paths ────────────────────────────────────────────────────────
RETRIEVAL_DIR   = PROJECT_ROOT / "retrieval"
GRAPH_DIR       = PROJECT_ROOT / "graph"
REASONING_DIR   = PROJECT_ROOT / "reasoning"
CACHE_DIR       = PROJECT_ROOT / "cache"
LOGS_DIR        = PROJECT_ROOT / "logs"

DATABASE_DIR       = PROJECT_ROOT / "database"
VECTOR_INDEX_DIR   = DATABASE_DIR / "vector_index"
KG_DIR             = DATABASE_DIR / "knowledge_graph"
SEARCH_INDEX_DIR   = DATABASE_DIR / "search_index"

ANNOTATED_DIR   = PROJECT_ROOT / "data" / "annotated"

# ── Retrieval settings ─────────────────────────────────────────────────────
TOP_K                   = 5
SEMANTIC_WEIGHT         = 0.45
KEYWORD_WEIGHT          = 0.35
GRAPH_WEIGHT            = 0.20

# ── Embedding model ────────────────────────────────────────────────────────
# Multilingual model — handles Devanagari well
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM   = 384          # dimension for the above model
EMBEDDING_BATCH = 32

# ── Confidence weights ─────────────────────────────────────────────────────
CONF_MODEL_WEIGHT     = 0.50
CONF_RETRIEVAL_WEIGHT = 0.30
CONF_GRAPH_WEIGHT     = 0.20

# ── Cache ──────────────────────────────────────────────────────────────────
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB    = int(os.getenv("REDIS_DB",   "0"))
CACHE_TTL   = 3600          # seconds

# ── Math detection rules ───────────────────────────────────────────────────
MATH_DOMAINS  = {"Mathematics", "Astronomy", "Geometry", "Algebra",
                 "Arithmetic", "Trigonometry", "Number Theory"}
MATH_BRANCHES = {"Geometry", "Algebra", "Arithmetic", "Trigonometry",
                 "Number Theory", "Combinatorics", "Calculus"}

MATH_KEYWORDS_DEVANAGARI = [
    "गणित", "संख्या", "क्षेत्र", "कोण", "वर्ग", "घन", "सूत्र",
    "भुज", "कर्ण", "त्रिभुज", "वृत्त", "व्यास", "परिधि", "क्षेत्रफल",
]
MATH_KEYWORDS_LATIN = [
    "ganita", "sankhya", "kshetra", "kona", "varga", "ghana", "sutra",
    "bhuja", "karna", "tribhuja", "vritta", "vyasa", "paridhi",
]

# ── Known formula patterns (symbolic) ─────────────────────────────────────
# Maps keyword/source hint → SymPy-compatible expression string
KNOWN_FORMULAS = {
    "pythagorean":     {"expr": "a**2 + b**2 - c**2",   "latex": "a^2 + b^2 = c^2",
                        "source": "Sulba Sutra",          "name": "Pythagorean theorem"},
    "circle_area":     {"expr": "pi * r**2",             "latex": "\\pi r^2",
                        "source": "Aryabhatiya",          "name": "Area of circle"},
    "triangle_area":   {"expr": "(b * h) / 2",           "latex": "\\frac{1}{2}bh",
                        "source": "Sulba Sutra",          "name": "Area of triangle"},
    "square_area":     {"expr": "a**2",                  "latex": "a^2",
                        "source": "Sulba Sutra",          "name": "Area of square"},
    "arithmetic_sum":  {"expr": "n*(a + l)/2",           "latex": "\\frac{n(a+l)}{2}",
                        "source": "Aryabhatiya",          "name": "Arithmetic series sum"},
    "quadratic":       {"expr": "(-b + sqrt(b**2 - 4*a*c)) / (2*a)",
                        "latex": "\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}",
                        "source": "Brahmagupta",          "name": "Quadratic formula"},
    "vedic_square":    {"expr": "(a + b)**2 - 2*a*b",   "latex": "(a+b)^2 - 2ab",
                        "source": "Vedic Mathematics",    "name": "Vedic square identity"},
}

# ── Reasoning trace templates ──────────────────────────────────────────────
REASONING_TEMPLATES = {
    "keyword_match":    "Keyword '{kw}' detected in shloka text",
    "domain_classify":  "Domain classified as '{domain}' by Phase 2 model (conf={conf:.2f})",
    "veda_classify":    "Veda identified as '{veda}' (conf={conf:.2f})",
    "math_detected":    "Mathematical content detected — domain is '{domain}'",
    "formula_found":    "Formula '{name}' retrieved from knowledge graph ({source})",
    "formula_generated":"Formula generated symbolically: {expr}",
    "no_formula":       "No formula pattern detected — shloka is non-mathematical",
    "retrieval_hit":    "Top retrieved shloka: '{sid}' (score={score:.3f})",
    "graph_relation":   "Knowledge graph relation found: {rel}",
    "low_confidence":   "Low confidence ({conf:.2f}) — insufficient training data",
}
