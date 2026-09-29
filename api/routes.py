"""
api/routes.py
─────────────
FastAPI route definitions.
Pipeline is injected via app.state so it is loaded once at startup.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    HealthResponse, ErrorResponse, MetricsResponse,
)
from api.metrics_collector import get_collector
from app_config.settings import LOG_DIR
from scripts.logger import setup_logger, JSONLogger

logger     = setup_logger("api.routes", LOG_DIR)
req_audit  = JSONLogger(LOG_DIR / "api_requests.jsonl")

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(request: Request):
    """Module 8 — Health check endpoint."""
    pipeline_ready = getattr(request.app.state, "pipeline", None) is not None
    uptime = time.time() - getattr(request.app.state, "start_time", time.time())
    return HealthResponse(
        status          = "running",
        version         = "4.0.0",
        pipeline_ready  = pipeline_ready,
        uptime_seconds  = round(uptime, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# /analyze
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, tags=["reasoning"])
async def analyze_shloka(body: AnalyzeRequest, request: Request):
    """
    Main analysis endpoint.
    Runs the Phase 3 reasoning pipeline on the input shloka.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready. Please try again in a moment.",
        )

    t0 = time.perf_counter()
    collector = get_collector()
    shloka = body.shloka

    logger.info(f"Analyze request: '{shloka[:60]}'")

    try:
        raw    = pipeline.query(shloka)
        result = AnalyzeResponse.from_pipeline(raw)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        collector.record(elapsed_ms, success=True,
                         cache_hit=result.cache_hit)

        req_audit.log(
            "request",
            shloka      = shloka[:80],
            veda        = result.veda,
            domain      = result.domain,
            math        = result.math,
            confidence  = result.confidence,
            elapsed_ms  = round(elapsed_ms, 1),
            cache_hit   = result.cache_hit,
        )
        logger.info(
            f"  → veda={result.veda} domain={result.domain} "
            f"math={result.math} conf={result.confidence:.3f} "
            f"elapsed={elapsed_ms:.0f}ms"
        )
        return result

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        collector.record(elapsed_ms, success=False)
        logger.error(f"Pipeline error for '{shloka[:40]}': {exc}", exc_info=True)
        req_audit.log("error", shloka=shloka[:80], error=str(exc))
        raise HTTPException(status_code=500, detail="Internal pipeline error.")


# ─────────────────────────────────────────────────────────────────────────────
# /metrics
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse, tags=["system"])
async def get_metrics():
    """Module 9 — Live performance metrics."""
    snap = get_collector().snapshot()
    return MetricsResponse(**{k: snap[k] for k in MetricsResponse.model_fields
                               if k in snap})


# ─────────────────────────────────────────────────────────────────────────────
# /examples  (helper for UI)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/examples", tags=["helper"])
async def get_examples():
    """Return sample shlokas for the UI demo."""
    return {
        "examples": [
            {
                "label":  "Rigveda Opening",
                "shloka": "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।",
            },
            {
                "label":  "Sulba Sutra — Geometry",
                "shloka": "समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।",
            },
            {
                "label":  "Vedic Mathematics Sutra",
                "shloka": "एकाधिकेन पूर्वेण। इति सूत्रम्।",
            },
            {
                "label":  "Atharvaveda — Astronomy",
                "shloka": "ज्योतिष ग्रह नक्षत्र सूर्य।",
            },
            {
                "label":  "Philosophy",
                "shloka": "विद्या ददाति विनयम्।",
            },
        ]
    }
