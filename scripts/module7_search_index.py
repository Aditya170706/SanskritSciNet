"""
Module 7 — Symbolic Search Index
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Builds a vectorless keyword/symbolic search system.
Primary: Elasticsearch
Fallback: Local inverted index stored in database/search_index/
"""

import json
import re
from pathlib import Path
from typing import Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANNOTATED_DIR, SEARCH_INDEX_DIR, LOGS_DIR, ES_HOST, ES_INDEX_NAME
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("search_index", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "search_audit.jsonl")


# ---------------------------------------------------------------------------
# Elasticsearch connector
# ---------------------------------------------------------------------------

SHLOKA_MAPPING = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "shloka_text": {"type": "text", "analyzer": "standard"},
            "source": {"type": "keyword"},
            "chapter": {"type": "keyword"},
            "veda": {"type": "keyword"},
            "domain": {"type": "keyword"},
            "branch": {"type": "keyword"},
            "formula": {"type": "keyword"},
            "language": {"type": "keyword"},
            "keywords": {"type": "keyword"},
            "annotated_by": {"type": "keyword"},
            "confidence": {"type": "float"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
}


class ElasticsearchConnector:
    """Thin wrapper around the Elasticsearch Python client."""

    def __init__(self, host: str = ES_HOST, index: str = ES_INDEX_NAME):
        self.index = index
        try:
            from elasticsearch import Elasticsearch
            self._es = Elasticsearch([host])
            self._es.info()
            logger.info(f"Connected to Elasticsearch at {host}")
            self.available = True
        except Exception as exc:
            logger.warning(f"Elasticsearch unavailable ({exc}). Will use local index fallback.")
            self._es = None
            self.available = False

    def create_index(self):
        """Create index with shloka mapping (idempotent)."""
        if not self.available:
            return
        if not self._es.indices.exists(index=self.index):
            self._es.indices.create(index=self.index, body=SHLOKA_MAPPING)
            logger.info(f"Created Elasticsearch index '{self.index}'")
        else:
            logger.info(f"Elasticsearch index '{self.index}' already exists")

    def index_doc(self, doc: dict):
        """Index a single shloka document."""
        self._es.index(index=self.index, id=doc["id"], document=doc)

    def bulk_index(self, docs: list[dict]) -> int:
        """Bulk index documents using the helpers API."""
        from elasticsearch.helpers import bulk
        actions = [{"_index": self.index, "_id": d["id"], "_source": d} for d in docs]
        success, errors = bulk(self._es, actions, raise_on_error=False)
        if errors:
            logger.warning(f"Bulk index errors: {errors[:3]}")
        return success

    def search(self, query: str, filters: dict = None, size: int = 20) -> list[dict]:
        """Keyword search with optional filters."""
        must = [{"multi_match": {"query": query, "fields": ["shloka_text", "keywords", "domain", "branch", "veda"]}}]
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                filter_clauses.append({"term": {field: value}})

        body = {"query": {"bool": {"must": must, "filter": filter_clauses}}, "size": size}
        result = self._es.search(index=self.index, body=body)
        return [hit["_source"] for hit in result["hits"]["hits"]]

    def exact_search(self, field: str, value: str, size: int = 20) -> list[dict]:
        """Exact match on a keyword field."""
        body = {"query": {"term": {field: value}}, "size": size}
        result = self._es.search(index=self.index, body=body)
        return [hit["_source"] for hit in result["hits"]["hits"]]


# ---------------------------------------------------------------------------
# Local inverted index (fallback)
# ---------------------------------------------------------------------------

class LocalSearchIndex:
    """
    In-memory + disk inverted index for keyword, exact, and filter search.
    Persisted as JSON shards in database/search_index/
    """

    def __init__(self, index_dir: Path = SEARCH_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.docs_path = self.index_dir / "documents.json"
        self.index_path = self.index_dir / "inverted_index.json"
        self._docs: dict[str, dict] = {}         # id → doc
        self._inverted: dict[str, list[str]] = {} # term → [ids]
        self._load()

    # ---- Persistence ----

    def _load(self):
        if self.docs_path.exists():
            with open(self.docs_path, encoding="utf-8") as f:
                self._docs = json.load(f)
        if self.index_path.exists():
            with open(self.index_path, encoding="utf-8") as f:
                self._inverted = json.load(f)
        logger.debug(f"Loaded local index: {len(self._docs):,} docs, {len(self._inverted):,} terms")

    def _save(self):
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._inverted, f, ensure_ascii=False)

    # ---- Indexing ----

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        # Split on whitespace, danda, and standard punctuation
        tokens = re.split(r"[\s।॥,;.!?()\[\]{}<>\"']+", text.lower())
        return [t for t in tokens if t and len(t) > 1]

    def index_doc(self, doc: dict):
        did = doc["id"]
        self._docs[did] = doc

        # Collect all searchable text fields
        fields = ["shloka_text", "keywords", "domain", "branch", "veda", "source"]
        combined = " ".join(str(doc.get(f, "") or "") for f in fields)
        tokens = self._tokenize(combined)

        for token in set(tokens):
            if token not in self._inverted:
                self._inverted[token] = []
            if did not in self._inverted[token]:
                self._inverted[token].append(did)

    def bulk_index(self, docs: list[dict]) -> int:
        for doc in docs:
            self.index_doc(doc)
        self._save()
        logger.info(f"Local index updated: {len(self._docs):,} docs, {len(self._inverted):,} unique terms")
        return len(docs)

    # ---- Search ----

    def keyword_search(self, query: str, filters: dict = None, size: int = 20) -> list[dict]:
        """Return docs matching any query term, optionally filtered."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Score = number of matching tokens
        scores: dict[str, int] = defaultdict(int)
        for token in tokens:
            for did in self._inverted.get(token, []):
                scores[did] += 1

        # Apply filters
        if filters:
            def _matches_filters(doc: dict) -> bool:
                for field, value in filters.items():
                    doc_val = doc.get(field, "")
                    if isinstance(doc_val, list):
                        if value not in doc_val:
                            return False
                    elif str(doc_val).lower() != str(value).lower():
                        return False
                return True
            scored_ids = [did for did in scores if _matches_filters(self._docs.get(did, {}))]
        else:
            scored_ids = list(scores.keys())

        # Sort by score descending
        scored_ids.sort(key=lambda did: scores[did], reverse=True)
        return [self._docs[did] for did in scored_ids[:size] if did in self._docs]

    def exact_search(self, field: str, value: str, size: int = 20) -> list[dict]:
        """Return docs where field exactly matches value."""
        results = []
        for doc in self._docs.values():
            doc_val = doc.get(field, "")
            if isinstance(doc_val, list):
                if value in doc_val:
                    results.append(doc)
            elif str(doc_val).lower() == str(value).lower():
                results.append(doc)
            if len(results) >= size:
                break
        return results

    def filter_search(self, filters: dict, size: int = 20) -> list[dict]:
        """Return docs matching all provided field=value filters."""
        results = []
        for doc in self._docs.values():
            match = True
            for field, value in filters.items():
                doc_val = doc.get(field, "")
                if isinstance(doc_val, list):
                    if value not in doc_val:
                        match = False
                        break
                elif str(doc_val).lower() != str(value).lower():
                    match = False
                    break
            if match:
                results.append(doc)
            if len(results) >= size:
                break
        return results

    def statistics(self) -> dict:
        return {
            "total_documents": len(self._docs),
            "unique_terms": len(self._inverted),
        }


# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------

def shloka_to_index_doc(shloka: dict) -> dict:
    """Convert annotated shloka to an index document."""
    ann = shloka.get("annotation", {})
    keywords = ann.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split("|") if k.strip()]

    return {
        "id": shloka["id"],
        "shloka_text": shloka.get("text", ""),
        "source": shloka.get("source", ""),
        "chapter": shloka.get("chapter", ""),
        "veda": ann.get("veda", "Unknown"),
        "domain": ann.get("domain", "Unknown"),
        "branch": ann.get("branch", "Unknown"),
        "formula": ann.get("formula", ""),
        "language": ann.get("language", "Sanskrit"),
        "keywords": keywords,
        "annotated_by": ann.get("annotated_by", ""),
        "confidence": ann.get("confidence", 1.0),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_search_index(
    annotated_dir: str | Path = None,
    index_dir: str | Path = None,
    use_elasticsearch: bool = True,
) -> dict:
    """
    Build the search index from all annotated shlokas.

    Tries Elasticsearch first; falls back to local inverted index.
    """
    annotated_dir = Path(annotated_dir or ANNOTATED_DIR)
    index_dir = Path(index_dir or SEARCH_INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load all annotated shlokas
    all_shlokas = []
    for jf in sorted(annotated_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            all_shlokas.extend(json.load(f))

    if not all_shlokas:
        logger.warning("No annotated shlokas found. Run Module 4 first.")
        return {}

    docs = [shloka_to_index_doc(s) for s in all_shlokas]
    logger.info(f"Prepared {len(docs):,} documents for indexing")

    if use_elasticsearch:
        es = ElasticsearchConnector()
        if es.available:
            es.create_index()
            count = es.bulk_index(docs)
            audit.log("index_built", backend="elasticsearch", indexed=count)
            return {"backend": "elasticsearch", "indexed": count}

    # Fallback
    logger.info("Using local inverted index.")
    local_idx = LocalSearchIndex(index_dir)
    count = local_idx.bulk_index(docs)
    stats = local_idx.statistics()
    audit.log("index_built", backend="local", indexed=count, stats=stats)
    logger.info(f"Index statistics: {stats}")
    return {"backend": "local", "indexed": count, "stats": stats}


def get_local_index(index_dir: str | Path = None) -> LocalSearchIndex:
    """Return a loaded LocalSearchIndex instance for querying."""
    return LocalSearchIndex(Path(index_dir or SEARCH_INDEX_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build search index from annotated shlokas")
    parser.add_argument("--annotated-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--no-es", action="store_true", help="Disable Elasticsearch")
    parser.add_argument("--search", default=None, help="Test keyword search query")
    parser.add_argument("--exact-field", default=None, help="Field for exact search")
    parser.add_argument("--exact-value", default=None, help="Value for exact search")
    args = parser.parse_args()

    if args.search or args.exact_field:
        idx = get_local_index(args.index_dir)
        if args.exact_field and args.exact_value:
            results = idx.exact_search(args.exact_field, args.exact_value)
        else:
            results = idx.keyword_search(args.search)
        for r in results[:5]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        result = build_search_index(
            annotated_dir=args.annotated_dir,
            index_dir=args.index_dir,
            use_elasticsearch=not args.no_es,
        )
        print(json.dumps(result, indent=2))
