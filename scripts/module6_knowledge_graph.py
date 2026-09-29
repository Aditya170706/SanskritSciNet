"""
Module 6 — Knowledge Graph Database
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Creates a Neo4j knowledge graph with node types:
  Shloka, Veda, Domain, Branch, Formula, Source, Keyword

Relationships:
  BELONGS_TO, HAS_DOMAIN, HAS_BRANCH, HAS_FORMULA, FROM_SOURCE, HAS_KEYWORD

Also provides a local JSON-based fallback when Neo4j is unavailable.
"""

import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ANNOTATED_DIR, KNOWLEDGE_GRAPH_DIR, LOGS_DIR,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("knowledge_graph", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "kg_audit.jsonl")


# ---------------------------------------------------------------------------
# Neo4j driver wrapper
# ---------------------------------------------------------------------------

class Neo4jConnector:
    """Thin wrapper around the Neo4j Python driver."""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
            self.available = True
        except Exception as exc:
            logger.warning(f"Neo4j unavailable ({exc}). Will use local JSON fallback.")
            self._driver = None
            self.available = False

    def close(self):
        if self._driver:
            self._driver.close()

    def run(self, query: str, **params):
        if not self.available:
            raise RuntimeError("Neo4j not connected")
        with self._driver.session() as session:
            return session.run(query, **params).data()

    def run_write(self, query: str, **params):
        if not self.available:
            raise RuntimeError("Neo4j not connected")
        with self._driver.session() as session:
            return session.execute_write(lambda tx: tx.run(query, **params).data())


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

SCHEMA_QUERIES = [
    # Uniqueness constraints
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Shloka) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Veda) REQUIRE v.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Branch) REQUIRE b.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Formula) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (src:Source) REQUIRE src.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE k.term IS UNIQUE",
    # Full-text indexes for search
    "CREATE FULLTEXT INDEX shloka_text_index IF NOT EXISTS FOR (s:Shloka) ON EACH [s.text]",
]


def create_schema(connector: Neo4jConnector):
    """Apply schema constraints and indexes to Neo4j."""
    for q in SCHEMA_QUERIES:
        try:
            connector.run(q)
        except Exception as exc:
            logger.warning(f"Schema query skipped ({exc}): {q[:60]}...")
    logger.info("Neo4j schema applied.")


# ---------------------------------------------------------------------------
# Cypher queries for node/relationship creation
# ---------------------------------------------------------------------------

def _upsert_shloka(tx, shloka: dict):
    ann = shloka.get("annotation", {})
    tx.run(
        """
        MERGE (s:Shloka {id: $id})
        SET s.text = $text,
            s.source = $source,
            s.chapter = $chapter,
            s.language = $language,
            s.confidence = $confidence,
            s.annotated_by = $annotated_by
        """,
        id=shloka["id"],
        text=shloka.get("text", ""),
        source=shloka.get("source", ""),
        chapter=shloka.get("chapter", ""),
        language=ann.get("language", "Sanskrit"),
        confidence=ann.get("confidence", 1.0),
        annotated_by=ann.get("annotated_by", ""),
    )


def _upsert_relationships(tx, shloka: dict):
    sid = shloka["id"]
    ann = shloka.get("annotation", {})

    veda = ann.get("veda", "Unknown")
    domain = ann.get("domain", "Unknown")
    branch = ann.get("branch", "Unknown")
    formula = ann.get("formula", "")
    source = ann.get("source", shloka.get("source", "Unknown"))
    keywords = ann.get("keywords", [])

    # BELONGS_TO Veda
    tx.run(
        "MERGE (v:Veda {name: $veda}) "
        "WITH v MATCH (s:Shloka {id: $id}) MERGE (s)-[:BELONGS_TO]->(v)",
        id=sid, veda=veda
    )

    # HAS_DOMAIN
    tx.run(
        "MERGE (d:Domain {name: $domain}) "
        "WITH d MATCH (s:Shloka {id: $id}) MERGE (s)-[:HAS_DOMAIN]->(d)",
        id=sid, domain=domain
    )

    # HAS_BRANCH
    tx.run(
        "MERGE (b:Branch {name: $branch}) "
        "WITH b MATCH (s:Shloka {id: $id}) MERGE (s)-[:HAS_BRANCH]->(b)",
        id=sid, branch=branch
    )

    # HAS_FORMULA
    if formula:
        tx.run(
            "MERGE (f:Formula {name: $formula}) "
            "WITH f MATCH (s:Shloka {id: $id}) MERGE (s)-[:HAS_FORMULA]->(f)",
            id=sid, formula=formula
        )

    # FROM_SOURCE
    if source:
        tx.run(
            "MERGE (src:Source {name: $source}) "
            "WITH src MATCH (s:Shloka {id: $id}) MERGE (s)-[:FROM_SOURCE]->(src)",
            id=sid, source=source
        )

    # HAS_KEYWORD
    for kw in keywords:
        if kw:
            tx.run(
                "MERGE (k:Keyword {term: $kw}) "
                "WITH k MATCH (s:Shloka {id: $id}) MERGE (s)-[:HAS_KEYWORD]->(k)",
                id=sid, kw=kw
            )


def insert_shloka_neo4j(connector: Neo4jConnector, shloka: dict):
    """Insert a single annotated shloka into Neo4j."""
    with connector._driver.session() as session:
        session.execute_write(_upsert_shloka, shloka)
        session.execute_write(_upsert_relationships, shloka)


def bulk_insert_neo4j(connector: Neo4jConnector, shlokas: list[dict]) -> int:
    """Bulk insert annotated shlokas into Neo4j."""
    count = 0
    for i, shloka in enumerate(shlokas):
        try:
            insert_shloka_neo4j(connector, shloka)
            count += 1
            if (i + 1) % 100 == 0:
                logger.info(f"  Inserted {i + 1:,}/{len(shlokas):,} shlokas into Neo4j")
        except Exception as exc:
            logger.error(f"Failed to insert {shloka.get('id')}: {exc}")
    return count


# ---------------------------------------------------------------------------
# Local JSON fallback knowledge graph
# ---------------------------------------------------------------------------

class LocalKnowledgeGraph:
    """
    JSON-based knowledge graph for use when Neo4j is not available.
    Stores nodes and edges in KNOWLEDGE_GRAPH_DIR/graph.json.
    """

    def __init__(self, graph_dir: Path = KNOWLEDGE_GRAPH_DIR):
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.graph_dir / "graph.json"
        self._load()

    def _load(self):
        if self.graph_path.exists():
            with open(self.graph_path, encoding="utf-8") as f:
                data = json.load(f)
            self.nodes = data.get("nodes", {})
            self.edges = data.get("edges", [])
        else:
            self.nodes = {}  # node_type:name → properties dict
            self.edges = []  # list of {from, rel, to}

    def _save(self):
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, ensure_ascii=False, indent=2)

    def _add_node(self, node_type: str, key: str, properties: dict = None):
        node_id = f"{node_type}:{key}"
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, "key": key, **(properties or {})}

    def _add_edge(self, from_key: str, rel: str, to_key: str):
        edge = {"from": from_key, "rel": rel, "to": to_key}
        if edge not in self.edges:
            self.edges.append(edge)

    def insert_shloka(self, shloka: dict):
        ann = shloka.get("annotation", {})
        sid = shloka["id"]

        # Shloka node
        self._add_node("Shloka", sid, {
            "text": shloka.get("text", ""),
            "source": shloka.get("source", ""),
            "chapter": shloka.get("chapter", ""),
            "language": ann.get("language", "Sanskrit"),
        })

        veda = ann.get("veda", "Unknown")
        domain = ann.get("domain", "Unknown")
        branch = ann.get("branch", "Unknown")
        formula = ann.get("formula", "")
        source = ann.get("source", shloka.get("source", ""))
        keywords = ann.get("keywords", [])

        self._add_node("Veda", veda)
        self._add_edge(f"Shloka:{sid}", "BELONGS_TO", f"Veda:{veda}")

        self._add_node("Domain", domain)
        self._add_edge(f"Shloka:{sid}", "HAS_DOMAIN", f"Domain:{domain}")

        self._add_node("Branch", branch)
        self._add_edge(f"Shloka:{sid}", "HAS_BRANCH", f"Branch:{branch}")

        if formula:
            self._add_node("Formula", formula)
            self._add_edge(f"Shloka:{sid}", "HAS_FORMULA", f"Formula:{formula}")

        if source:
            self._add_node("Source", source)
            self._add_edge(f"Shloka:{sid}", "FROM_SOURCE", f"Source:{source}")

        for kw in keywords:
            if kw:
                self._add_node("Keyword", kw)
                self._add_edge(f"Shloka:{sid}", "HAS_KEYWORD", f"Keyword:{kw}")

    def bulk_insert(self, shlokas: list[dict]) -> int:
        for shloka in shlokas:
            self.insert_shloka(shloka)
        self._save()
        logger.info(
            f"Local KG updated: {len(self.nodes):,} nodes, {len(self.edges):,} edges"
        )
        return len(shlokas)

    def query_by_domain(self, domain: str) -> list[str]:
        """Return list of shloka IDs belonging to a given domain."""
        domain_key = f"Domain:{domain}"
        return [
            e["from"].split(":")[-1]
            for e in self.edges
            if e["rel"] == "HAS_DOMAIN" and e["to"] == domain_key
        ]

    def statistics(self) -> dict:
        node_types = {}
        for node in self.nodes.values():
            t = node["type"]
            node_types[t] = node_types.get(t, 0) + 1
        rel_types = {}
        for edge in self.edges:
            r = edge["rel"]
            rel_types[r] = rel_types.get(r, 0) + 1
        return {"nodes": node_types, "relationships": rel_types}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    annotated_dir: str | Path = None,
    graph_dir: str | Path = None,
    use_neo4j: bool = True,
) -> dict:
    """
    Build the knowledge graph from all annotated shlokas.

    Tries Neo4j first; falls back to local JSON graph if unavailable.
    """
    annotated_dir = Path(annotated_dir or ANNOTATED_DIR)
    graph_dir = Path(graph_dir or KNOWLEDGE_GRAPH_DIR)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Load all annotated shlokas
    all_shlokas = []
    for jf in sorted(annotated_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            all_shlokas.extend(json.load(f))

    if not all_shlokas:
        logger.warning("No annotated shlokas found. Run Module 4 first.")
        return {}

    logger.info(f"Building knowledge graph from {len(all_shlokas):,} shlokas")

    if use_neo4j:
        connector = Neo4jConnector()
        if connector.available:
            create_schema(connector)
            count = bulk_insert_neo4j(connector, all_shlokas)
            connector.close()
            audit.log("kg_built", backend="neo4j", inserted=count)
            return {"backend": "neo4j", "inserted": count}

    # Fallback to local JSON
    logger.info("Using local JSON knowledge graph.")
    kg = LocalKnowledgeGraph(graph_dir)
    count = kg.bulk_insert(all_shlokas)
    stats = kg.statistics()
    audit.log("kg_built", backend="local_json", inserted=count, stats=stats)
    logger.info(f"Knowledge graph statistics: {stats}")
    return {"backend": "local_json", "inserted": count, "stats": stats}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build knowledge graph from annotated shlokas")
    parser.add_argument("--annotated-dir", default=None)
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--no-neo4j", action="store_true")
    args = parser.parse_args()

    result = build_knowledge_graph(
        annotated_dir=args.annotated_dir,
        graph_dir=args.graph_dir,
        use_neo4j=not args.no_neo4j,
    )
    print(json.dumps(result, indent=2))
