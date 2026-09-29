"""
api/schemas.py
──────────────
Pydantic request / response models for the FastAPI server.
Includes input validation with clear error messages.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config.settings import MAX_INPUT_LEN, MIN_INPUT_LEN


# ─────────────────────────────────────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """POST /analyze request body."""
    shloka: str

    @field_validator("shloka")
    @classmethod
    def validate_shloka(cls, v: str) -> str:
        v = v.strip()

        if len(v) < MIN_INPUT_LEN:
            raise ValueError(
                f"Input too short (min {MIN_INPUT_LEN} characters). "
                "Please enter a valid Sanskrit shloka."
            )
        if len(v) > MAX_INPUT_LEN:
            raise ValueError(
                f"Input too long (max {MAX_INPUT_LEN} characters)."
            )

        # Must contain at least some printable content
        if not re.search(r'\S', v):
            raise ValueError("Input contains only whitespace.")

        # Block null bytes and control characters (keep newlines/tabs)
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', v):
            raise ValueError("Input contains invalid control characters.")

        return v


# ─────────────────────────────────────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────────────────────────────────────

class FormulaInfo(BaseModel):
    name:   str = ""
    expr:   str = ""
    latex:  str = ""
    source: str = ""


class ConfidenceBreakdown(BaseModel):
    model:     float = 0.0
    retrieval: float = 0.0
    graph:     float = 0.0


class Phase2Prediction(BaseModel):
    label: str  = "Unknown"
    conf:  float = 0.0


class Phase2Predictions(BaseModel):
    veda:   Phase2Prediction = Phase2Prediction()
    domain: Phase2Prediction = Phase2Prediction()
    branch: Phase2Prediction = Phase2Prediction()


class RetrievedShloka(BaseModel):
    id:          str   = ""
    text:        str   = ""
    source:      str   = ""
    final_score: float = 0.0


class AnalyzeResponse(BaseModel):
    """POST /analyze response body."""
    veda:       str = "Unknown"
    domain:     str = "Unknown"
    branch:     str = "Unknown"
    math:       bool = False
    formula:    Optional[FormulaInfo] = None
    confidence: float = 0.0
    confidence_breakdown: ConfidenceBreakdown = ConfidenceBreakdown()
    reasoning:  list[str] = []
    top_retrieved: list[RetrievedShloka] = []
    phase2_predictions: Phase2Predictions = Phase2Predictions()
    elapsed_ms: float = 0.0
    cache_hit:  bool  = False

    @classmethod
    def from_pipeline(cls, raw: dict) -> "AnalyzeResponse":
        """Convert raw pipeline output dict to typed response."""
        formula = None
        raw_f   = raw.get("formula")
        if raw_f:
            formula = FormulaInfo(
                name   = raw_f.get("name",   ""),
                expr   = raw_f.get("expr",   ""),
                latex  = raw_f.get("latex",  ""),
                source = raw_f.get("source", ""),
            )

        cb_raw = raw.get("confidence_breakdown", {})
        cb = ConfidenceBreakdown(
            model     = cb_raw.get("model",     0.0),
            retrieval = cb_raw.get("retrieval", 0.0),
            graph     = cb_raw.get("graph",     0.0),
        )

        p2_raw = raw.get("phase2_predictions", {})
        p2 = Phase2Predictions(
            veda   = Phase2Prediction(**p2_raw.get("veda",   {})),
            domain = Phase2Prediction(**p2_raw.get("domain", {})),
            branch = Phase2Prediction(**p2_raw.get("branch", {})),
        )

        retrieved = [
            RetrievedShloka(**r)
            for r in raw.get("top_retrieved", [])[:3]
        ]

        return cls(
            veda       = raw.get("veda",       "Unknown"),
            domain     = raw.get("domain",     "Unknown"),
            branch     = raw.get("branch",     "Unknown"),
            math       = bool(raw.get("math_flag", False)),
            formula    = formula,
            confidence = float(raw.get("confidence", 0.0)),
            confidence_breakdown = cb,
            reasoning  = raw.get("reasoning", []),
            top_retrieved        = retrieved,
            phase2_predictions   = p2,
            elapsed_ms = float(raw.get("elapsed_ms", 0.0)),
            cache_hit  = bool(raw.get("cache_hit",   False)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Health / Error
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str  = "running"
    version: str  = "4.0.0"
    pipeline_ready: bool = False
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
    code:    int = 400


class MetricsResponse(BaseModel):
    total_requests:    int   = 0
    successful:        int   = 0
    failed:            int   = 0
    avg_latency_ms:    float = 0.0
    p95_latency_ms:    float = 0.0
    cache_hit_rate:    float = 0.0
    cpu_percent:       float = 0.0
    memory_mb:         float = 0.0
