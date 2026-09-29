"""
api/main.py
───────────
Module 1 — FastAPI application entry point.

Features
  • Pipeline loaded once at startup (lifespan event)
  • CORS enabled for Streamlit UI
  • Structured request logging middleware
  • Global exception handler
  • /health, /analyze, /metrics, /examples endpoints
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.routes import router
from api.metrics_collector import get_collector
from app_config.settings import (
    SERVER_HOST, SERVER_PORT, LOG_DIR,
    PIPELINE_TOP_K, PIPELINE_USE_NEO4J,
    PIPELINE_USE_CACHE, PIPELINE_BUILD_IDX,
)
from scripts.logger import setup_logger

logger = setup_logger("api.main", LOG_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline at startup, release resources at shutdown."""
    app.state.start_time = time.time()
    app.state.pipeline   = None

    logger.info("=" * 56)
    logger.info("  Vedic Shloka Intelligence System — API Starting")
    logger.info("=" * 56)

    try:
        from p3_pipeline import VedicReasoningPipeline
        logger.info("Loading Phase 3 pipeline…")
        t0 = time.time()
        app.state.pipeline = VedicReasoningPipeline(
            build_index = PIPELINE_BUILD_IDX,
            use_neo4j   = PIPELINE_USE_NEO4J,
            use_cache   = PIPELINE_USE_CACHE,
            top_k       = PIPELINE_TOP_K,
        )
        logger.info(f"Pipeline ready in {time.time() - t0:.1f}s")
    except Exception as exc:
        logger.error(f"Pipeline failed to load: {exc}. "
                     "API will return 503 until resolved.")

    logger.info(f"API server listening on {SERVER_HOST}:{SERVER_PORT}")
    yield

    # Shutdown
    logger.info("API server shutting down.")
    try:
        get_collector().flush()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Vedic Shloka Intelligence API",
    description = (
        "Phase 4 REST API for the Vedic Shloka Intelligence and "
        "Mathematical Knowledge Extraction System."
    ),
    version     = "4.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — allow Streamlit UI and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware — request logging
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = (time.perf_counter() - t0) * 1000
    logger.debug(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({ms:.1f}ms)"
    )
    response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {"error": "Internal server error", "detail": str(exc)},
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code = 404,
        content     = {"error": f"Route '{request.url.path}' not found."},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["root"])
async def root():
    return {
        "name":    "Vedic Shloka Intelligence API",
        "version": "4.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "analyze": "POST /analyze",
    }


# Include routes
app.include_router(router)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host    = SERVER_HOST,
        port    = SERVER_PORT,
        reload  = False,
        workers = 1,
    )
