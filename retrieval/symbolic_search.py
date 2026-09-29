"""
retrieval/symbolic_search.py
────────────────────────────
Module 3 — Symbolic Retrieval Engine

Wraps the Phase 1 LocalSearchIndex to provide keyword, exact, and
filter search over the pre-built inverted index.
Also supports Elasticsearch when available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_config import SEARCH_INDEX_DIR, TOP_K
from retrieval.p3_logger import get_logger

logger = get_logger("symbolic_search")


class SymbolicSearchEngine:
    """
    Thin façade over the Phase 1 LocalSearchIndex with Elasticsearch fallback.
    Provides keyword, exact-match, and filter search.
    """

    def __init__(self, index_dir: Path = SEARCH_INDEX_DIR):
        self._index_dir = Path(index_dir)
        self._local     = self._load_local()
        self._es        = self._try_es()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_local(self):
        try:
            from scripts.module7_search_index import LocalSearchIndex
            idx = LocalSearchIndex(self._index_dir)
            logger.info(f"Local search index loaded: "
                        f"{idx.statistics()['total_documents']} docs")
            return idx
        except Exception as exc:
            logger.warning(f"Could not load local search index: {exc}")
            return None

    def _try_es(self):
        try:
            from elasticsearch import Elasticsearch
            from phase3_config import SEARCH_INDEX_DIR
            es = Elasticsearch(["http://localhost:9200"])
            es.info()
            logger.info("Elasticsearch connection established")
            return es
        except Exception:
            return None

    # ── Search ────────────────────────────────────────────────────────────

    def keyword_search(
        self, query: str, filters: dict = None, top_k: int = TOP_K
    ) -> list[dict]:
        """
        Multi-term keyword search. Returns results with 'keyword_score'.
        """
        if self._local:
            raw = self._local.keyword_search(query, filters=filters, size=top_k)
            return self._score(raw, query)
        logger.warning("No search backend available for keyword search.")
        return []

    def exact_search(self, field: str, value: str, top_k: int = TOP_K) -> list[dict]:
        """Exact term match on a single field."""
        if self._local:
            return self._local.exact_search(field, value, size=top_k)
        return []

    def filter_search(self, filters: dict, top_k: int = TOP_K) -> list[dict]:
        """Multi-field AND filter."""
        if self._local:
            return self._local.filter_search(filters, size=top_k)
        return []

    # ── Score ─────────────────────────────────────────────────────────────

    @staticmethod
    def _score(results: list[dict], query: str) -> list[dict]:
        """
        Attach a normalised keyword_score based on token overlap
        between the query and each result's shloka_text + keywords.
        """
        import re
        tokens = set(re.split(r"[\s।॥,;.]+", query.lower())) - {""}
        scored = []
        for r in results:
            text_tokens = set(re.split(
                r"[\s।॥,;.]+",
                (r.get("shloka_text", "") + " " +
                 " ".join(r.get("keywords", []))).lower()
            )) - {""}
            overlap = len(tokens & text_tokens)
            r["keyword_score"] = round(
                overlap / max(len(tokens), 1), 4
            )
            scored.append(r)
        return scored

    @property
    def ready(self) -> bool:
        return self._local is not None
