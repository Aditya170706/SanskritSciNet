"""
reasoning/reasoning_engine.py
──────────────────────────────
Module 8 — Reasoning Engine
Module 9 — Confidence Estimator

Orchestrates all Phase 3 components into a single structured result.
Produces a human-readable reasoning trace and a composite confidence score.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_config import (
    CONF_MODEL_WEIGHT, CONF_RETRIEVAL_WEIGHT, CONF_GRAPH_WEIGHT,
    REASONING_TEMPLATES as T,
)
from retrieval.p3_logger import get_logger, get_audit

logger = get_logger("reasoning_engine")
audit  = get_audit("reasoning_engine")


# ─────────────────────────────────────────────────────────────────────────────
# Module 9 — Confidence Estimator
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceEstimator:
    """
    Weighted composite confidence score.

    score = model_weight   * model_conf
          + retrieval_weight * retrieval_conf
          + graph_weight     * graph_conf
    """

    def __init__(
        self,
        model_w:     float = CONF_MODEL_WEIGHT,
        retrieval_w: float = CONF_RETRIEVAL_WEIGHT,
        graph_w:     float = CONF_GRAPH_WEIGHT,
    ):
        self.model_w     = model_w
        self.retrieval_w = retrieval_w
        self.graph_w     = graph_w

    def estimate(
        self,
        model_conf:     float,
        retrieval_conf: float,
        graph_conf:     float,
    ) -> float:
        """
        Compute weighted confidence in [0, 1].

        Args:
            model_conf:     Phase 2 classifier overall confidence
            retrieval_conf: Top-1 retrieval final_score (hybrid)
            graph_conf:     KG relation density score

        Returns:
            Composite confidence in [0, 1]
        """
        score = (
            self.model_w     * _clamp(model_conf)
            + self.retrieval_w * _clamp(retrieval_conf)
            + self.graph_w     * _clamp(graph_conf)
        )
        return round(_clamp(score), 4)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ─────────────────────────────────────────────────────────────────────────────
# Module 8 — Reasoning Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningEngine:
    """
    Integrates Phase 2 classifier output, hybrid retrieval, KG queries,
    math detection, formula extraction/generation, and confidence
    estimation into a single structured JSON response.

    Args:
        inference_pipeline:  Phase 2 ShlokaInferencePipeline
        hybrid_engine:       Phase 3 HybridRetrievalEngine
        kg_engine:           Phase 3 KnowledgeGraphQueryEngine
        math_detector:       MathContentDetector
        formula_extractor:   FormulaExtractor
        formula_generator:   FormulaGenerator
        confidence_estimator:ConfidenceEstimator
    """

    def __init__(
        self,
        inference_pipeline,
        hybrid_engine,
        kg_engine,
        math_detector,
        formula_extractor,
        formula_generator,
        confidence_estimator: Optional[ConfidenceEstimator] = None,
    ):
        self.inference    = inference_pipeline
        self.retrieval    = hybrid_engine
        self.kg           = kg_engine
        self.math_det     = math_detector
        self.formula_ext  = formula_extractor
        self.formula_gen  = formula_generator
        self.conf_est     = confidence_estimator or ConfidenceEstimator()

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def reason(self, text: str, top_k: int = 5) -> dict:
        """
        Full reasoning pipeline for a single shloka.

        Returns structured JSON-serialisable dict.
        """
        t0    = time.perf_counter()
        trace = []   # reasoning trace lines

        # ── Step 1 — Phase 2 classification ──────────────────────────────
        p2_result   = self.inference.predict(text)
        veda        = p2_result.get("veda",   "Unknown")
        domain      = p2_result.get("domain", "Unknown")
        branch      = p2_result.get("branch", "Unknown")
        p2_confs    = p2_result.get("confidence", {})
        p2_probs    = p2_result.get("probabilities", {})
        model_conf  = p2_confs.get("overall", 0.0)

        trace.append(T["veda_classify"].format(
            veda=veda, conf=p2_confs.get("veda", 0.0)))
        trace.append(T["domain_classify"].format(
            domain=domain, conf=p2_confs.get("domain", 0.0)))

        # ── Step 2 — Hybrid retrieval ─────────────────────────────────────
        ret_results    = self.retrieval.retrieve(text, top_k=top_k)
        retrieval_conf = ret_results[0]["final_score"] if ret_results else 0.0

        if ret_results:
            top = ret_results[0]
            trace.append(T["retrieval_hit"].format(
                sid=top.get("id", "?"), score=top.get("final_score", 0)))

        # ── Step 3 — KG relations ─────────────────────────────────────────
        top_id    = ret_results[0].get("id", "") if ret_results else ""
        relations = self.kg.get_shloka_relations(top_id) if top_id else []
        graph_conf = min(len(relations) / 10.0, 1.0)

        for rel in relations[:2]:
            rel_str = rel.get("rel") or str(rel)
            trace.append(T["graph_relation"].format(rel=rel_str))

        # ── Step 4 — Math detection ────────────────────────────────────────
        math_result = self.math_det.detect(
            text, domain=domain, branch=branch, phase2_probs=p2_probs
        )
        math_flag  = math_result["math_flag"]
        math_conf  = math_result["confidence"]

        for trig in math_result["triggers"][:2]:
            trace.append(trig)

        if math_flag:
            trace.append(T["math_detected"].format(domain=domain))

        # ── Step 5 — Formula extraction or generation ─────────────────────
        formula_result = None
        if math_flag:
            formula_result = self.formula_ext.extract(
                text,
                shloka_id=top_id or None,
                domain=domain,
                branch=branch,
            )
            if formula_result:
                trace.append(T["formula_found"].format(
                    name=formula_result.get("name", formula_result.get("expr", "")),
                    source=formula_result.get("source", ""),
                ))
            else:
                formula_result = self.formula_gen.generate(
                    text, domain=domain, branch=branch
                )
                trace.append(T["formula_generated"].format(
                    expr=formula_result.get("expr", "")))
        else:
            trace.append(T["no_formula"])

        # ── Step 6 — Composite confidence ─────────────────────────────────
        final_conf = self.conf_est.estimate(
            model_conf=model_conf,
            retrieval_conf=retrieval_conf,
            graph_conf=graph_conf,
        )

        if final_conf < 0.35:
            trace.append(T["low_confidence"].format(conf=final_conf))

        # ── Assemble output ───────────────────────────────────────────────
        elapsed = (time.perf_counter() - t0) * 1000

        output: dict = {
            "input_text":  text,
            "veda":        veda,
            "domain":      domain,
            "branch":      branch,
            "math_flag":   math_flag,
            "formula":     _format_formula(formula_result) if formula_result else None,
            "confidence":  final_conf,
            "confidence_breakdown": {
                "model":     round(model_conf,     4),
                "retrieval": round(retrieval_conf, 4),
                "graph":     round(graph_conf,     4),
            },
            "reasoning":          trace,
            "top_retrieved":      _format_retrieved(ret_results[:3]),
            "phase2_predictions": {
                "veda":   {"label": veda,   "conf": p2_confs.get("veda",   0)},
                "domain": {"label": domain, "conf": p2_confs.get("domain", 0)},
                "branch": {"label": branch, "conf": p2_confs.get("branch", 0)},
            },
            "elapsed_ms": round(elapsed, 1),
        }

        logger.info(
            f"Reasoning complete in {elapsed:.0f}ms — "
            f"domain={domain}  math={math_flag}  conf={final_conf}"
        )
        audit.log("reasoning_complete", domain=domain, math_flag=math_flag,
                  confidence=final_conf, elapsed_ms=round(elapsed, 1))
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_formula(f: dict) -> dict:
    return {
        "name":   f.get("name", f.get("expr", "")),
        "expr":   f.get("expr", ""),
        "latex":  f.get("latex", ""),
        "source": f.get("source", ""),
    }


def _format_retrieved(results: list[dict]) -> list[dict]:
    return [
        {
            "id":            r.get("id", ""),
            "text":          r.get("shloka_text", r.get("text", ""))[:80],
            "source":        r.get("source", ""),
            "final_score":   r.get("final_score", 0),
        }
        for r in results
    ]
