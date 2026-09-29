"""
graph/graph_queries.py
──────────────────────
Module 4 — Knowledge Graph Query Engine

Provides a unified query interface over:
  • Neo4j (primary)
  • Phase 1 LocalKnowledgeGraph JSON fallback (automatic)

Supports:
  find_related_shlokas(shloka_id)
  find_by_domain(domain)
  find_by_veda(veda)
  find_formulas(shloka_id)
  get_shloka_relations(shloka_id)
  find_by_branch(branch)
  graph_statistics()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_config import KG_DIR
from retrieval.p3_logger import get_logger

logger = get_logger("graph_queries")


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j backend
# ─────────────────────────────────────────────────────────────────────────────

class _Neo4jBackend:
    def __init__(self):
        try:
            from neo4j import GraphDatabase
            import os
            uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER",      "neo4j")
            pwd  = os.getenv("NEO4J_PASSWORD",  "password")
            self._driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self._driver.verify_connectivity()
            self.available = True
            logger.info("Neo4j connected")
        except Exception as exc:
            logger.warning(f"Neo4j unavailable: {exc}")
            self._driver  = None
            self.available = False

    def run(self, query: str, **params) -> list[dict]:
        with self._driver.session() as s:
            return s.run(query, **params).data()

    def find_related_shlokas(self, shloka_id: str) -> list[dict]:
        return self.run(
            "MATCH (s:Shloka {id:$id})-[r]-(n) RETURN type(r) AS rel, n",
            id=shloka_id
        )

    def find_by_domain(self, domain: str) -> list[dict]:
        return self.run(
            "MATCH (s:Shloka)-[:HAS_DOMAIN]->(d:Domain {name:$d}) RETURN s",
            d=domain
        )

    def find_formulas(self, shloka_id: str) -> list[dict]:
        return self.run(
            "MATCH (s:Shloka {id:$id})-[:HAS_FORMULA]->(f:Formula) RETURN f",
            id=shloka_id
        )

    def find_by_veda(self, veda: str) -> list[dict]:
        return self.run(
            "MATCH (s:Shloka)-[:BELONGS_TO]->(v:Veda {name:$v}) RETURN s",
            v=veda
        )

    def find_by_branch(self, branch: str) -> list[dict]:
        return self.run(
            "MATCH (s:Shloka)-[:HAS_BRANCH]->(b:Branch {name:$b}) RETURN s",
            b=branch
        )

    def graph_statistics(self) -> dict:
        node_counts = self.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
        )
        rel_counts  = self.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt"
        )
        return {
            "nodes":         {r["label"]: r["cnt"] for r in node_counts},
            "relationships": {r["rel"]:   r["cnt"] for r in rel_counts},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Local JSON fallback backend
# ─────────────────────────────────────────────────────────────────────────────

class _LocalBackend:
    """Wraps Phase 1 LocalKnowledgeGraph for Phase 3 queries."""

    def __init__(self, kg_dir: Path):
        from scripts.module6_knowledge_graph import LocalKnowledgeGraph
        self._kg      = LocalKnowledgeGraph(kg_dir)
        self.available = len(self._kg.nodes) > 0
        if self.available:
            logger.info(f"Local KG loaded: {len(self._kg.nodes)} nodes  "
                        f"{len(self._kg.edges)} edges")
        else:
            logger.warning("Local KG is empty — run Phase 1 first.")

    def find_related_shlokas(self, shloka_id: str) -> list[dict]:
        node_key = f"Shloka:{shloka_id}"
        return [
            {"rel": e["rel"], "target": e["to"]}
            for e in self._kg.edges
            if e["from"] == node_key
        ]

    def find_by_domain(self, domain: str) -> list[str]:
        return self._kg.query_by_domain(domain)

    def find_formulas(self, shloka_id: str) -> list[str]:
        node_key = f"Shloka:{shloka_id}"
        formulas = []
        for e in self._kg.edges:
            if e["from"] == node_key and e["rel"] == "HAS_FORMULA":
                formula_key = e["to"]                 # "Formula:<name>"
                name = formula_key.split(":", 1)[-1]
                formulas.append(name)
        return formulas

    def find_by_veda(self, veda: str) -> list[str]:
        veda_key = f"Veda:{veda}"
        return [
            e["from"].split(":", 1)[-1]
            for e in self._kg.edges
            if e["rel"] == "BELONGS_TO" and e["to"] == veda_key
        ]

    def find_by_branch(self, branch: str) -> list[str]:
        branch_key = f"Branch:{branch}"
        return [
            e["from"].split(":", 1)[-1]
            for e in self._kg.edges
            if e["rel"] == "HAS_BRANCH" and e["to"] == branch_key
        ]

    def get_all_nodes(self) -> dict:
        return self._kg.nodes

    def graph_statistics(self) -> dict:
        return self._kg.statistics()


# ─────────────────────────────────────────────────────────────────────────────
# Unified query engine
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphQueryEngine:
    """
    Unified KG query engine — tries Neo4j first, falls back to local JSON.
    All methods return plain Python dicts/lists for easy JSON serialisation.
    """

    def __init__(self, kg_dir: Path = KG_DIR, use_neo4j: bool = True):
        self._neo4j = _Neo4jBackend() if use_neo4j else None
        self._local = _LocalBackend(Path(kg_dir))
        self._backend_name = (
            "neo4j" if (self._neo4j and self._neo4j.available) else "local"
        )
        logger.info(f"KG backend: {self._backend_name}")

    @property
    def _be(self):
        """Active backend."""
        if self._neo4j and self._neo4j.available:
            return self._neo4j
        return self._local

    # ── Public query methods ───────────────────────────────────────────────

    def get_shloka_relations(self, shloka_id: str) -> list[dict]:
        """Return all relationships for a given shloka node."""
        try:
            return self._be.find_related_shlokas(shloka_id)
        except Exception as exc:
            logger.error(f"get_shloka_relations({shloka_id}): {exc}")
            return []

    def find_by_domain(self, domain: str) -> list:
        """Return shloka IDs/nodes belonging to domain."""
        try:
            return self._be.find_by_domain(domain)
        except Exception as exc:
            logger.error(f"find_by_domain({domain}): {exc}")
            return []

    def find_by_veda(self, veda: str) -> list:
        try:
            return self._be.find_by_veda(veda)
        except Exception as exc:
            logger.error(f"find_by_veda({veda}): {exc}")
            return []

    def find_by_branch(self, branch: str) -> list:
        try:
            return self._be.find_by_branch(branch)
        except Exception as exc:
            logger.error(f"find_by_branch({branch}): {exc}")
            return []

    def find_formulas(self, shloka_id: str) -> list[str]:
        """Return formula names associated with a shloka."""
        try:
            return self._be.find_formulas(shloka_id)
        except Exception as exc:
            logger.error(f"find_formulas({shloka_id}): {exc}")
            return []

    def graph_statistics(self) -> dict:
        try:
            return self._be.graph_statistics()
        except Exception as exc:
            logger.error(f"graph_statistics: {exc}")
            return {}
