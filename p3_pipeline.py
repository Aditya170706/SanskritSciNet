"""
p3_pipeline.py
──────────────
Master Phase 3 pipeline — assembles all components and exposes a single
query(text) method that returns the full structured reasoning result.

Also provides a CLI for interactive testing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase3_config import (
    TOP_K, VECTOR_INDEX_DIR, KG_DIR, ANNOTATED_DIR, LOGS_DIR,
)
from retrieval.p3_logger import get_logger, get_audit

logger = get_logger("p3_pipeline")
audit  = get_audit("p3_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline assembly
# ─────────────────────────────────────────────────────────────────────────────

class VedicReasoningPipeline:
    """
    End-to-end Phase 3 pipeline.

    Initialization order:
    1. Build vector index (if not already built)
    2. Load Phase 2 inference pipeline
    3. Load KG query engine
    4. Load hybrid retrieval engine
    5. Load math detector, formula extractor/generator
    6. Load reasoning engine + confidence estimator
    7. Load cache
    """

    def __init__(
        self,
        build_index:   bool = True,
        use_neo4j:     bool = False,
        use_cache:     bool = True,
        top_k:         int  = TOP_K,
    ):
        self._top_k     = top_k
        self._use_cache = use_cache

        logger.info("Initialising Phase 3 pipeline…")
        t0 = time.time()

        # ── Vector index ─────────────────────────────────────────────────
        if build_index and not (VECTOR_INDEX_DIR / "embeddings.npy").exists():
            logger.info("Building vector index (first run)…")
            from retrieval.semantic_search import build_vector_index
            build_vector_index()

        # ── Phase 2 inference ─────────────────────────────────────────────
        logger.info("Loading Phase 2 classifiers…")
        try:
            from scripts.phase2.p2_inference import ShlokaInferencePipeline
            self._inference = ShlokaInferencePipeline()
            logger.info(f"  Loaded tasks: {self._inference.loaded_tasks}")
        except Exception as exc:
            logger.warning(f"Phase 2 models not found: {exc}. "
                           "Run p2_run_training.py first. Using stub.")
            self._inference = _StubInference()

        # ── Knowledge graph ───────────────────────────────────────────────
        from graph.graph_queries import KnowledgeGraphQueryEngine
        self._kg = KnowledgeGraphQueryEngine(kg_dir=KG_DIR, use_neo4j=use_neo4j)

        # ── Hybrid retrieval ──────────────────────────────────────────────
        from retrieval.hybrid_engine import HybridRetrievalEngine
        self._retrieval = HybridRetrievalEngine(top_k=top_k, kg_engine=self._kg)

        # ── Math engine ───────────────────────────────────────────────────
        from reasoning.math_engine import (
            MathContentDetector, FormulaExtractor, FormulaGenerator
        )
        self._math_det    = MathContentDetector()
        self._formula_ext = FormulaExtractor(kg_engine=self._kg)
        self._formula_gen = FormulaGenerator()

        # ── Reasoning + confidence ────────────────────────────────────────
        from reasoning.reasoning_engine import ReasoningEngine, ConfidenceEstimator
        self._reasoning = ReasoningEngine(
            inference_pipeline  = self._inference,
            hybrid_engine       = self._retrieval,
            kg_engine           = self._kg,
            math_detector       = self._math_det,
            formula_extractor   = self._formula_ext,
            formula_generator   = self._formula_gen,
            confidence_estimator= ConfidenceEstimator(),
        )

        # ── Cache ─────────────────────────────────────────────────────────
        if use_cache:
            from cache.redis_cache import RetrievalCache
            self._cache = RetrievalCache()
            logger.info(f"  Cache backend: {self._cache.backend}")
        else:
            self._cache = None

        logger.info(f"Pipeline ready in {time.time() - t0:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = None) -> dict:
        """
        Full pipeline query for a single shloka.

        Args:
            text:  Sanskrit shloka string
            top_k: Number of similar shlokas to retrieve

        Returns:
            Structured reasoning result dict
        """
        k = top_k or self._top_k

        # Cache check
        if self._cache:
            key = self._cache.make_key(text)
            hit = self._cache.get(key)
            if hit:
                hit["cache_hit"] = True
                logger.debug(f"Cache hit for '{text[:40]}'")
                return hit

        result = self._reasoning.reason(text, top_k=k)
        result["cache_hit"] = False

        if self._cache:
            self._cache.set(key, result)

        audit.log("query", text=text[:60],
                  domain=result.get("domain"),
                  math_flag=result.get("math_flag"),
                  confidence=result.get("confidence"))
        return result

    def cache_stats(self) -> dict:
        return self._cache.stats() if self._cache else {}

    def kg_stats(self) -> dict:
        return self._kg.graph_statistics()


# ─────────────────────────────────────────────────────────────────────────────
# Stub for when Phase 2 models are missing
# ─────────────────────────────────────────────────────────────────────────────

class _StubInference:
    loaded_tasks = []

    def predict(self, text: str) -> dict:
        return {
            "veda": "Unknown", "domain": "Unknown", "branch": "Unknown",
            "confidence": {"veda": 0.0, "domain": 0.0, "branch": 0.0, "overall": 0.0},
            "probabilities": {"veda": {}, "domain": {}, "branch": {}},
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    """Pretty-print the reasoning result."""
    print("\n" + "═" * 64)
    print("  VEDIC REASONING ENGINE — RESULT")
    print("═" * 64)
    print(f"  Veda    : {result['veda']}")
    print(f"  Domain  : {result['domain']}")
    print(f"  Branch  : {result['branch']}")
    print(f"  Math    : {'✓' if result['math_flag'] else '✗'}")
    if result.get("formula"):
        f = result["formula"]
        print(f"  Formula : {f.get('name', f.get('expr', ''))}")
        print(f"  LaTeX   : {f.get('latex', '')}")
        print(f"  Source  : {f.get('source', '')}")
    print(f"  Confidence: {result['confidence']:.3f}  "
          f"(model={result['confidence_breakdown']['model']:.2f}  "
          f"retrieval={result['confidence_breakdown']['retrieval']:.2f}  "
          f"graph={result['confidence_breakdown']['graph']:.2f})")
    print("\n  Reasoning trace:")
    for i, step in enumerate(result["reasoning"], 1):
        print(f"    {i}. {step}")
    if result.get("top_retrieved"):
        print("\n  Top retrieved shlokas:")
        for r in result["top_retrieved"]:
            print(f"    [{r['id']}] score={r['final_score']:.3f}  {r['text'][:50]}…")
    print(f"\n  Elapsed : {result.get('elapsed_ms', 0):.0f}ms  "
          f"cache={'HIT' if result.get('cache_hit') else 'MISS'}")
    print("═" * 64 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3 — Vedic Shloka Reasoning Engine"
    )
    parser.add_argument(
        "text", nargs="?",
        default="समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।",
        help="Sanskrit shloka to analyse",
    )
    parser.add_argument("--top-k",      type=int,  default=TOP_K)
    parser.add_argument("--no-neo4j",   action="store_true")
    parser.add_argument("--no-cache",   action="store_true")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Skip rebuilding vector index")
    parser.add_argument("--json",       action="store_true",
                        help="Print raw JSON output")
    args = parser.parse_args()

    pipe   = VedicReasoningPipeline(
        build_index = not args.no_rebuild,
        use_neo4j   = not args.no_neo4j,
        use_cache   = not args.no_cache,
        top_k       = args.top_k,
    )
    result = pipe.query(args.text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_result(result)
