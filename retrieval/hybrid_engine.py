"""
retrieval/hybrid_engine.py
──────────────────────────
Module 1 — Hybrid Retrieval Engine

Merges semantic (dense) + symbolic (keyword) + knowledge-graph signals
into a single ranked list of top-k shlokas.

Ranking formula
  final_score = semantic_weight  * semantic_score
              + keyword_weight   * keyword_score
              + graph_weight     * graph_score
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_config import (
    TOP_K, SEMANTIC_WEIGHT, KEYWORD_WEIGHT, GRAPH_WEIGHT,
)
from retrieval.p3_logger import get_logger, get_audit
from retrieval.semantic_search import SemanticSearchEngine
from retrieval.symbolic_search import SymbolicSearchEngine

logger = get_logger("hybrid_engine")
audit  = get_audit("hybrid_engine")


class HybridRetrievalEngine:
    """
    Combines semantic similarity, keyword matching, and KG relevance
    into a single ranked retrieval result.

    Args:
        top_k:            Number of final results to return
        semantic_weight:  Weight for dense embedding score  (default 0.45)
        keyword_weight:   Weight for keyword overlap score  (default 0.35)
        graph_weight:     Weight for knowledge-graph score  (default 0.20)
        kg_engine:        Optional KnowledgeGraphQueryEngine instance
    """

    def __init__(
        self,
        top_k:            int   = TOP_K,
        semantic_weight:  float = SEMANTIC_WEIGHT,
        keyword_weight:   float = KEYWORD_WEIGHT,
        graph_weight:     float = GRAPH_WEIGHT,
        kg_engine=None,
    ):
        self.top_k           = top_k
        self.semantic_weight = semantic_weight
        self.keyword_weight  = keyword_weight
        self.graph_weight    = graph_weight
        self.kg_engine       = kg_engine   # injected; may be None

        self._semantic = SemanticSearchEngine()
        self._symbolic = SymbolicSearchEngine()

        logger.info(
            f"HybridRetrievalEngine ready  "
            f"semantic={self._semantic.ready}  "
            f"symbolic={self._symbolic.ready}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve and rank shlokas relevant to query.

        Args:
            query:   Input shloka or query text
            filters: Optional field=value filters (e.g. {"veda": "Rigveda"})
            top_k:   Override default top-k

        Returns:
            Ranked list of result dicts, each with:
              id, text, source, semantic_score, keyword_score,
              graph_score, final_score
        """
        k  = top_k or self.top_k
        t0 = time.perf_counter()

        # ── Run both retrievers ───────────────────────────────────────────
        sem_results = (
            self._semantic.search(query, top_k=k * 2)
            if self._semantic.ready else []
        )
        kw_results = (
            self._symbolic.keyword_search(query, filters=filters, top_k=k * 2)
            if self._symbolic.ready else []
        )

        # ── Build score map keyed by shloka id ───────────────────────────
        scores: dict[str, dict] = {}

        for r in sem_results:
            sid = r.get("id", "")
            scores.setdefault(sid, {"doc": r, "semantic": 0.0, "keyword": 0.0, "graph": 0.0})
            scores[sid]["semantic"] = float(r.get("semantic_score", 0))

        for r in kw_results:
            sid = r.get("id", "")
            scores.setdefault(sid, {"doc": r, "semantic": 0.0, "keyword": 0.0, "graph": 0.0})
            scores[sid]["keyword"] = float(r.get("keyword_score", 0))

        # ── Graph relevance boost ─────────────────────────────────────────
        if self.kg_engine:
            for sid in list(scores.keys()):
                g_score = self._graph_relevance(sid, query)
                scores[sid]["graph"] = g_score

        # ── Compute final score ───────────────────────────────────────────
        results = []
        for sid, data in scores.items():
            final = (
                self.semantic_weight * data["semantic"]
                + self.keyword_weight * data["keyword"]
                + self.graph_weight   * data["graph"]
            )
            doc = data["doc"].copy()
            doc.update({
                "id":             sid,
                "semantic_score": round(data["semantic"], 4),
                "keyword_score":  round(data["keyword"],  4),
                "graph_score":    round(data["graph"],    4),
                "final_score":    round(final,            4),
            })
            results.append(doc)

        # ── Sort and truncate ─────────────────────────────────────────────
        results.sort(key=lambda x: x["final_score"], reverse=True)
        results = results[:k]

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Hybrid retrieval: {len(results)} results in {elapsed:.1f}ms  "
            f"query='{query[:40]}'"
        )
        audit.log("retrieval", query=query[:60], results=len(results),
                  elapsed_ms=round(elapsed, 1))
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    def _graph_relevance(self, shloka_id: str, query: str) -> float:
        """
        Compute a graph-based relevance score for a shloka.
        Returns a value in [0, 1].
        """
        if self.kg_engine is None:
            return 0.0
        try:
            relations = self.kg_engine.get_shloka_relations(shloka_id)
            # Simple heuristic: more relations = higher relevance, capped at 1.0
            return min(len(relations) / 10.0, 1.0)
        except Exception:
            return 0.0
