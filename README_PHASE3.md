# Phase 3 — Hybrid Retrieval, Knowledge Graph & Mathematical Reasoning Engine

## Overview

Integrates Phase 1 (data/KG/index) and Phase 2 (trained classifiers) into a
unified reasoning pipeline that retrieves, classifies, and extracts mathematical
knowledge from Sanskrit shlokas.

---

## Architecture

```
Input Shloka
     │
     ├─► Phase 2 Classifiers ──► veda / domain / branch / confidence
     │
     ├─► Semantic Search ────────► cosine-similarity top-k (embeddings)
     │
     ├─► Symbolic Search ────────► keyword / exact / filter search
     │
     ├─► Hybrid Ranker ──────────► weighted merge → top-k results
     │
     ├─► KG Query Engine ────────► relations / formulas / domains
     │
     ├─► Math Detector ──────────► rule-based math flag + confidence
     │
     ├─► Formula Extractor ──────► KG lookup → known formula
     │   └─► Formula Generator ──► SymPy symbolic generation (fallback)
     │
     ├─► Confidence Estimator ───► model × retrieval × graph (weighted)
     │
     └─► Reasoning Engine ───────► structured JSON + trace
```

---

## Quick Start

```bash
# Install Phase 3 extras
pip install sentence-transformers sympy

# Run the pipeline (downloads embedding model ~90MB on first run)
python p3_pipeline.py "समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।" --no-neo4j

# JSON output mode
python p3_pipeline.py "अग्निमीळे पुरोहितं।" --no-neo4j --json

# Run all tests (92 total across all 3 phases)
pytest tests/ -v
```

---

## Module Reference

### `phase3_config.py`
All Phase 3 settings: embedding model, retrieval weights, math keyword lists,
known formula registry, confidence weights.

### `retrieval/semantic_search.py` — Module 2
- `build_vector_index()` — encodes all annotated shlokas → `database/vector_index/`
- `SemanticSearchEngine.search(query, top_k)` — cosine-similarity search
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (handles Devanagari)
- Embeddings stored as `embeddings.npy` + `metadata.json`

### `retrieval/symbolic_search.py` — Module 3
- `SymbolicSearchEngine.keyword_search(query, filters, top_k)` 
- `exact_search(field, value)` — term-level match
- `filter_search(filters)` — multi-field AND filter
- Backed by Phase 1 `LocalSearchIndex` (Elasticsearch optional)

### `retrieval/hybrid_engine.py` — Module 1
- `HybridRetrievalEngine.retrieve(query, filters, top_k)`
- Merges semantic + keyword + graph signals with configurable weights
- Default: semantic=0.45, keyword=0.35, graph=0.20

### `graph/graph_queries.py` — Module 4
- `KnowledgeGraphQueryEngine` — Neo4j primary, local JSON fallback
- `find_by_domain(domain)` — shlokas in a domain
- `find_by_veda(veda)` — shlokas from a Veda
- `find_formulas(shloka_id)` — KG formula edges
- `get_shloka_relations(shloka_id)` — all edges for a node
- `graph_statistics()` — node/edge counts

### `reasoning/math_engine.py` — Modules 5, 6, 7
**MathContentDetector** — 7-rule cascade:
  1. Phase 2 domain in MATH_DOMAINS
  2. Phase 2 branch in MATH_BRANCHES
  3. Phase 2 probability mass on Mathematics
  4. Devanagari keyword scan
  5. Latin transliteration keyword scan
  6. Numeric pattern scan
  7. Math operation term scan

**FormulaExtractor** — priority order:
  1. KG `HAS_FORMULA` edge lookup
  2. Text keyword → formula hint map
  3. Domain/branch heuristic

**FormulaGenerator** — SymPy-backed:
  1. Square-sum pattern → `a² + b² = c²`
  2. Square-root pattern → `√(a² + b²)`
  3. Product/sum patterns
  4. Domain/branch defaults
  5. Full SymPy validation + simplification

### `reasoning/reasoning_engine.py` — Modules 8, 9
**ConfidenceEstimator** — weighted composite:
```
score = 0.50 × model_conf + 0.30 × retrieval_conf + 0.20 × graph_conf
```

**ReasoningEngine.reason(text)** — 8-step pipeline returning structured JSON
with reasoning trace.

### `cache/redis_cache.py` — Module 10
- `RetrievalCache` — Redis primary, in-process LRU fallback
- `make_key(text)` — SHA-256 hash of normalised query
- TTL: 3600s (configurable)
- `stats()` — hit/miss rate

### `p3_pipeline.py` — Master Orchestrator
- `VedicReasoningPipeline.query(text, top_k)` — full pipeline with caching
- Auto-builds vector index on first run
- Gracefully stubs Phase 2 if models not found

---

## Output Schema

```json
{
  "input_text":  "समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।",
  "veda":        "Yajurveda",
  "domain":      "Mathematics",
  "branch":      "Geometry",
  "math_flag":   true,
  "formula": {
    "name":   "Pythagorean theorem",
    "expr":   "a**2 + b**2 - c**2",
    "latex":  "a^2 + b^2 = c^2",
    "source": "Sulba Sutra"
  },
  "confidence": 0.831,
  "confidence_breakdown": {
    "model":     0.91,
    "retrieval": 0.74,
    "graph":     0.60
  },
  "reasoning": [
    "Veda identified as 'Yajurveda' (conf=0.87)",
    "Domain classified as 'Mathematics' (conf=0.84)",
    "Top retrieved shloka: 'SHLOKA_000000' (score=0.921)",
    "Knowledge graph relation found: HAS_FORMULA",
    "Devanagari math keyword: 'क्षेत्र'",
    "Mathematical content detected — domain is 'Mathematics'",
    "Formula 'Pythagorean theorem' retrieved from knowledge graph (Sulba Sutra)"
  ],
  "top_retrieved": [
    {"id": "SHLOKA_000000", "text": "समचतुरश्रस्य...", "final_score": 0.921}
  ],
  "phase2_predictions": {
    "veda":   {"label": "Yajurveda",   "conf": 0.87},
    "domain": {"label": "Mathematics", "conf": 0.84},
    "branch": {"label": "Geometry",    "conf": 0.79}
  },
  "elapsed_ms": 47.3,
  "cache_hit":  false
}
```

---

## Known Formula Registry

| Key | Formula | Source |
|-----|---------|--------|
| pythagorean | a² + b² = c² | Sulba Sutra |
| circle_area | πr² | Aryabhatiya |
| triangle_area | ½bh | Sulba Sutra |
| square_area | a² | Sulba Sutra |
| arithmetic_sum | n(a+l)/2 | Aryabhatiya |
| quadratic | (−b ± √(b²−4ac)) / 2a | Brahmagupta |
| vedic_square | (a+b)² − 2ab | Vedic Mathematics |

---

## Retrieval Weights

| Signal | Weight | Source |
|--------|--------|--------|
| Semantic similarity | 0.45 | SentenceTransformer cosine sim |
| Keyword match | 0.35 | Inverted index token overlap |
| Graph relevance | 0.20 | KG edge count density |

All weights configurable in `phase3_config.py`.

---

## File Structure

```
vedic_shloka_system/
├── phase3_config.py
├── p3_pipeline.py               ← Master orchestrator + CLI
│
├── retrieval/
│   ├── hybrid_engine.py         ← Module 1
│   ├── semantic_search.py       ← Module 2
│   └── symbolic_search.py       ← Module 3
│
├── graph/
│   └── graph_queries.py         ← Module 4
│
├── reasoning/
│   ├── math_engine.py           ← Modules 5, 6, 7
│   └── reasoning_engine.py      ← Modules 8, 9
│
├── cache/
│   └── redis_cache.py           ← Module 10
│
├── database/
│   └── vector_index/            ← embeddings.npy + metadata.json
│
└── tests/phase3/
    └── test_phase3.py           ← 39 unit + integration tests
```

---

## Phase 4 Compatibility

The pipeline exposes a clean Python API:

```python
from p3_pipeline import VedicReasoningPipeline

pipe   = VedicReasoningPipeline()
result = pipe.query("your shloka text here")
# result is a plain dict — JSON serialisable, ready for FastAPI
```

For Phase 4 (deployment), wrap `pipe.query()` in a FastAPI endpoint.
The pipeline is stateful (models loaded once) and thread-safe for read operations.
