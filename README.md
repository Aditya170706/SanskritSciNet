# Vedic Shloka Intelligence and Mathematical Knowledge Extraction System
## Phase 1 — Data and Database Infrastructure

---

## Overview

A fully reproducible, modular, research-grade pipeline for ingesting, cleaning, segmenting, annotating, and storing Sanskrit shloka data from **public-domain sources only**, prepared for machine learning and knowledge graph workflows.

---

## Project Structure

```
vedic_shloka_system/
├── config.py                          # Central configuration
├── run_pipeline.py                    # Master pipeline orchestrator
├── demo.py                            # Quick-start demo (no external services required)
├── requirements.txt
├── docker-compose.yml                 # Neo4j + Elasticsearch + Kibana
│
├── data/
│   ├── raw/                           # Original ingested files
│   ├── cleaned/                       # Unicode-normalized, artifact-free text
│   ├── segmented/                     # Shloka JSON files (SHLOKA_000001…)
│   ├── annotated/                     # Annotated shloka JSON files
│   └── training/
│       ├── train.csv                  # 70% training split
│       ├── validation.csv             # 15% validation split
│       ├── test.csv                   # 15% test split
│       └── dataset_manifest.json
│
├── database/
│   ├── knowledge_graph/               # Neo4j or local JSON graph
│   └── search_index/                  # Elasticsearch or local inverted index
│
├── metadata/
│   └── source_registry.json           # All ingested source metadata
│
├── logs/
│   ├── system.log
│   ├── pipeline_audit.jsonl
│   ├── ingestion_audit.jsonl
│   ├── cleaning_audit.jsonl
│   ├── segmentation_audit.jsonl
│   ├── annotation_audit.jsonl
│   ├── dataset_audit.jsonl
│   ├── kg_audit.jsonl
│   ├── search_audit.jsonl
│   └── phase1_summary.json
│
├── scripts/
│   ├── logger.py                      # Structured logging utility
│   ├── module1_ingestion.py           # Data Ingestion Pipeline
│   ├── module2_cleaning.py            # Data Cleaning Pipeline
│   ├── module3_segmentation.py        # Shloka Segmentation Engine
│   ├── module4_annotation.py          # Annotation Engine
│   ├── module5_dataset.py             # Training Dataset Builder
│   ├── module6_knowledge_graph.py     # Knowledge Graph Database
│   ├── module7_search_index.py        # Symbolic Search Index
│   └── seed_sample_data.py            # Sample data seeder
│
└── tests/
    └── test_phase1.py                 # 30 unit + integration tests
```

---

## Quick Start (No External Services)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo on sample corpus (uses local JSON fallbacks)
python demo.py

# 3. Run tests
pytest tests/ -v
```

---

## Full Setup with Neo4j + Elasticsearch

```bash
# 1. Start database services
docker-compose up -d

# 2. Wait for services to be healthy (~30s), then run pipeline
python run_pipeline.py \
  --input /path/to/your/sanskrit/texts \
  --source-name "Rigveda Complete" \
  --author "Traditional" \
  --license "Public Domain" \
  --citation "Rigveda. Public domain Sanskrit text archive."
```

---

## Module Reference

### Module 1 — Data Ingestion
**Script:** `scripts/module1_ingestion.py`

Accepts PDF, TXT, XML, HTML, CSV. Extracts Sanskrit text, stores raw files in `data/raw/`, registers metadata in `metadata/source_registry.json`.

```bash
# Single file
python -m scripts.module1_ingestion /path/to/rigveda.txt \
  --source-name "Rigveda" --author "Traditional" --license "Public Domain"

# Directory
python -m scripts.module1_ingestion /path/to/texts/ --source-name "Sanskrit Corpus"
```

Each source entry in `source_registry.json` contains:
- `source_name`, `author`, `publication_year`, `license_type`, `citation_reference`
- `sha256`, `char_count`, `format`, `ingested_at`

---

### Module 2 — Data Cleaning
**Script:** `scripts/module2_cleaning.py`

Applies in order:
1. Remove invisible/zero-width characters
2. Unicode NFC normalization
3. UTF-8 safety re-encoding
4. Remove page numbers (standalone numeric lines)
5. Remove header/footer lines
6. Remove decorative dash separators
7. Remove roman numeral section markers
8. Normalize whitespace (collapse multiple blanks)

```bash
python -m scripts.module2_cleaning data/raw/ --output data/cleaned/
```

---

### Module 3 — Shloka Segmentation
**Script:** `scripts/module3_segmentation.py`

Splits text on Devanagari danda markers (`।` and `॥`). Assigns globally unique IDs (`SHLOKA_000001` format). Validates each segment contains ≥5 Devanagari characters. Falls back to paragraph splitting if no dandas found.

Output JSON format:
```json
{
  "id": "SHLOKA_000001",
  "text": "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।",
  "source": "Rigveda",
  "chapter": "1.1.1"
}
```

```bash
python -m scripts.module3_segmentation data/cleaned/ \
  --source "Rigveda" --chapter "1.1"
```

---

### Module 4 — Annotation Engine
**Script:** `scripts/module4_annotation.py`

Annotates each shloka with:

| Field | Options |
|-------|---------|
| `veda` | Rigveda, Samaveda, Yajurveda, Atharvaveda, Other, Unknown |
| `domain` | Mathematics, Astronomy, Grammar, Philosophy, Ritual, Medicine, Linguistics, Cosmology, Ethics, Unknown |
| `branch` | Arithmetic, Geometry, Algebra, Jyotisha, Vyakarana, Vedanta, Chandas… |
| `formula` | Formula string or empty |
| `source` | Source name |
| `language` | Sanskrit, Prakrit, Mixed, Unknown |
| `keywords` | List of strings |

Supports:
- **Manual annotation** via `manual_annotations` dict (`shloka_id → annotation`)
- **AI-assisted** heuristic annotation (Devanagari keyword matching for domain/branch/veda detection)

```bash
# AI-assisted (default)
python -m scripts.module4_annotation data/segmented/

# Manual only
python -m scripts.module4_annotation data/segmented/ --no-ai
```

---

### Module 5 — Training Dataset Builder
**Script:** `scripts/module5_dataset.py`

Flattens annotated shlokas into tabular format. Stratified split by `domain` field.

| Split | Ratio | File |
|-------|-------|------|
| Train | 70% | `data/training/train.csv` |
| Validation | 15% | `data/training/validation.csv` |
| Test | 15% | `data/training/test.csv` |

Columns: `id, text, source, chapter, veda, domain, branch, formula, language, keywords, annotated_by, confidence, notes`

```bash
python -m scripts.module5_dataset --seed 42
```

---

### Module 6 — Knowledge Graph Database
**Script:** `scripts/module6_knowledge_graph.py`

**Node Types:** `Shloka`, `Veda`, `Domain`, `Branch`, `Formula`, `Source`, `Keyword`

**Relationships:** `BELONGS_TO`, `HAS_DOMAIN`, `HAS_BRANCH`, `HAS_FORMULA`, `FROM_SOURCE`, `HAS_KEYWORD`

- **Primary backend:** Neo4j (with full-text index on shloka text)
- **Fallback:** Local JSON graph in `database/knowledge_graph/graph.json`

```bash
# With Neo4j running
python -m scripts.module6_knowledge_graph

# Local JSON fallback
python -m scripts.module6_knowledge_graph --no-neo4j
```

---

### Module 7 — Symbolic Search Index
**Script:** `scripts/module7_search_index.py`

Indexed fields: `shloka_text`, `keywords`, `domain`, `branch`, `veda`, `source`, `chapter`

Search types:
- **Keyword search** — multi-token match with scoring
- **Exact match** — term-level exact field matching
- **Filter search** — multi-field AND filter

- **Primary backend:** Elasticsearch
- **Fallback:** Local inverted index in `database/search_index/`

```bash
# Keyword search (local fallback)
python -m scripts.module7_search_index --search "गणित" --no-es

# Exact match
python -m scripts.module7_search_index --exact-field domain --exact-value Mathematics --no-es
```

---

## Legal and Data Source Compliance

All data must come from public-domain or open-license sources. The following are approved:

| Source | URL |
|--------|-----|
| Digital Corpus of Sanskrit (DCS) | http://www.sanskrit-linguistics.org/dcs/ |
| GRETIL Sanskrit Text Repository | http://gretil.sub.uni-goettingen.de/ |
| Sanskrit Documents Archive | https://sanskritdocuments.org/ |
| Muktabodha Digital Library | https://www.muktabodha.org/ |

Every ingested file stores `license_type` and `citation_reference` in `metadata/source_registry.json`.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data processing | pandas, numpy, regex |
| PDF extraction | pdfplumber |
| HTML/XML parsing | beautifulsoup4, lxml |
| Unicode | unicodedata (stdlib) |
| Knowledge graph | Neo4j 5.x (local JSON fallback) |
| Search index | Elasticsearch 8.x (local inverted index fallback) |
| Testing | pytest, pytest-cov |
| Containerization | Docker Compose |
| Logging | Custom structured logger + JSONL audit trails |

---

## Environment Variables

Copy `.env.example` to `.env` to override defaults:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
ES_HOST=http://localhost:9200
ES_INDEX_NAME=vedic_shlokas
LOG_LEVEL=INFO
```

---

## Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=scripts --cov-report=term-missing

# Single module tests
pytest tests/test_phase1.py::TestSegmentation -v
pytest tests/test_phase1.py::TestKnowledgeGraph -v
```

30 tests covering: ingestion, cleaning, segmentation, annotation, dataset generation, knowledge graph (local), search index (local), and full end-to-end integration.

---

## Reproducibility

- All random splits use a configurable `--seed` (default: 42)
- All pipeline steps are deterministic given the same input
- SHA-256 checksums stored per source file in `source_registry.json`
- Full audit trails in `logs/*.jsonl` (machine-readable)
- `logs/phase1_summary.json` captures complete run statistics
