# Phase 4 — API, User Interface & Deployment System

## Overview

Exposes the Phase 3 reasoning pipeline as a production-ready application
with a REST API, interactive web UI, structured logging, and performance metrics.

---

## Quick Start (3 commands)

```bash
# 1. Install Phase 4 dependencies
pip install fastapi uvicorn streamlit pydantic pyyaml psutil

# 2. Start everything (API + UI)
python scripts/run_system.py

# 3. Open browser
#    UI   → http://localhost:8501
#    API  → http://localhost:8000
#    Docs → http://localhost:8000/docs
```

---

## Starting Services Individually

```bash
# API only
python scripts/run_system.py --api

# UI only (requires API already running)
python scripts/run_system.py --ui

# Direct uvicorn (for development)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Direct streamlit (for development)
streamlit run ui/app.py --server.port 8501
```

---

## API Reference

### POST /analyze
Analyse a Sanskrit shloka.

**Request**
```json
{ "shloka": "समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।" }
```

**Response**
```json
{
  "veda":       "Yajurveda",
  "domain":     "Mathematics",
  "branch":     "Geometry",
  "math":       true,
  "formula": {
    "name":   "Pythagorean theorem",
    "expr":   "a**2 + b**2 - c**2",
    "latex":  "a^2 + b^2 = c^2",
    "source": "Sulba Sutra"
  },
  "confidence": 0.831,
  "confidence_breakdown": {
    "model": 0.91, "retrieval": 0.74, "graph": 0.60
  },
  "reasoning": [
    "Domain classified as 'Mathematics' (conf=0.84)",
    "Formula 'Pythagorean theorem' retrieved from knowledge graph"
  ],
  "elapsed_ms": 47.3,
  "cache_hit":  false
}
```

### GET /health
```json
{ "status": "running", "version": "4.0.0", "pipeline_ready": true, "uptime_seconds": 142.3 }
```

### GET /metrics
```json
{ "total_requests": 25, "avg_latency_ms": 43.2, "cpu_percent": 12.4, "memory_mb": 890.5 }
```

### GET /examples
Returns 5 sample shlokas for the UI.

### GET /docs
Interactive Swagger UI documentation.

---

## Web Interface Features

- Sanskrit text input with example shloka buttons
- Color-coded classification cards (Veda / Domain / Branch)
- Mathematical content badge
- Formula display with LaTeX and source
- Animated confidence score with per-component breakdown bars
- Full reasoning trace with numbered steps
- Top retrieved shlokas with similarity scores
- Response time and cache hit indicators
- Dark theme optimised for Devanagari text
- API status sidebar with live health check

---

## Configuration

Edit `app_config/config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

ui:
  port: 8501
  api_url: "http://localhost:8000"

pipeline:
  top_k: 5
  use_neo4j: false
  use_cache: true

security:
  max_input_length: 1000
  min_input_length: 2

logging:
  level: "INFO"
  dir:   "logs"
```

---

## File Structure

```
vedic_shloka_system/
├── api/
│   ├── main.py              ← FastAPI app + lifespan + middleware
│   ├── routes.py            ← /analyze /health /metrics /examples
│   ├── schemas.py           ← Pydantic request/response models
│   └── metrics_collector.py ← Thread-safe perf metrics
│
├── ui/
│   └── app.py               ← Streamlit web interface
│
├── app_config/
│   ├── config.yaml          ← All runtime settings
│   └── settings.py          ← Typed settings loader
│
├── scripts/
│   └── run_system.py        ← Master launcher (API + UI)
│
├── metrics/                 ← Periodic JSON metrics snapshots
├── logs/                    ← Structured request logs
│   └── api_requests.jsonl   ← Per-request audit trail
│
├── Dockerfile
└── tests/phase4/
    └── test_phase4.py       ← 37 unit + integration tests
```

---

## Error Handling

| Scenario | HTTP Code | Response |
|----------|-----------|----------|
| Empty input | 422 | Validation error detail |
| Input too short (<2 chars) | 422 | Validation error detail |
| Input too long (>1000 chars) | 422 | Validation error detail |
| Control characters in input | 422 | Validation error detail |
| Pipeline not ready | 503 | Service unavailable |
| Pipeline internal error | 500 | Internal server error |
| Unknown route | 404 | Route not found |

---

## Docker Deployment

```bash
# Build and run
docker build -t vedic-shloka .
docker run -p 8000:8000 -p 8501:8501 vedic-shloka

# Or use docker-compose (includes Neo4j + Elasticsearch)
docker-compose up -d
```

---

## Running Tests

```bash
# Phase 4 only
pytest tests/phase4/ -v

# All 4 phases (129 tests)
pytest tests/ -v
```

---

## Complete System Startup (E Drive)

```
cd E:\vedic_shloka\vedic_shloka_system
venv\Scripts\activate
python scripts/run_system.py
```

Open http://localhost:8501 in your browser.
