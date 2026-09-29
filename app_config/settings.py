"""
config/settings.py
──────────────────
Loads config/config.yaml and exposes a typed settings object.
Falls back to safe defaults if the file is missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_FILE = Path(__file__).parent / 'config.yaml'
_ROOT        = Path(__file__).parent.parent


def _load() -> dict:
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = _load()


def get(section: str, key: str, default: Any = None) -> Any:
    return _cfg.get(section, {}).get(key, default)


def section(name: str) -> dict:
    return _cfg.get(name, {})


# ── Convenience constants ──────────────────────────────────────────────────
SERVER_HOST        = get("server", "host",    "0.0.0.0")
SERVER_PORT        = int(get("server", "port", 8000))
UI_PORT            = int(get("ui",     "port", 8501))
API_URL            = get("ui",     "api_url",  "http://localhost:8000")

PIPELINE_TOP_K     = int(get("pipeline", "top_k",       5))
PIPELINE_USE_NEO4J = bool(get("pipeline", "use_neo4j",  False))
PIPELINE_USE_CACHE = bool(get("pipeline", "use_cache",  True))
PIPELINE_BUILD_IDX = bool(get("pipeline", "build_index",True))

MAX_INPUT_LEN      = int(get("security", "max_input_length", 1000))
MIN_INPUT_LEN      = int(get("security", "min_input_length",    2))

LOG_LEVEL          = get("logging", "level",   "INFO")
LOG_DIR            = _ROOT / get("logging", "dir", "logs")
METRICS_DIR        = _ROOT / get("metrics", "dir", "metrics")
METRICS_ENABLED    = bool(get("metrics", "enabled", True))
