"""
retrieval/semantic_search.py
────────────────────────────
Module 2 — Semantic Retrieval Engine

Encodes shlokas into dense vector representations using a multilingual
SentenceTransformer model, stores them in a local numpy index, and performs
cosine-similarity search.

Storage layout
  database/vector_index/
    embeddings.npy          ← (N, D) float32 matrix
    metadata.json           ← list of {id, text, source, ...}
    index_config.json       ← model name, dim, built_at
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_config import (
    VECTOR_INDEX_DIR, ANNOTATED_DIR, EMBEDDING_MODEL,
    EMBEDDING_DIM, EMBEDDING_BATCH, TOP_K,
)
from retrieval.p3_logger import get_logger, get_audit

logger = get_logger("semantic_search")
audit  = get_audit("semantic_search")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load SentenceTransformer, falling back gracefully."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded: {model_name}")
        return model
    except Exception as exc:
        logger.error(f"Failed to load embedding model '{model_name}': {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_index(
    annotated_dir: Path = ANNOTATED_DIR,
    index_dir: Path = VECTOR_INDEX_DIR,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH,
) -> dict:
    """
    Encode all annotated shlokas and persist the vector index.

    Returns:
        Dict with index statistics
    """
    annotated_dir = Path(annotated_dir)
    index_dir     = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shlokas ──────────────────────────────────────────────────────
    shlokas: list[dict] = []
    for jf in sorted(annotated_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            shlokas.extend(json.load(f))

    if not shlokas:
        logger.warning("No annotated shlokas found — run Phase 1 first.")
        return {}

    texts    = [s.get("text", "") for s in shlokas]
    metadata = [
        {
            "id":      s.get("id", ""),
            "text":    s.get("text", ""),
            "source":  s.get("source", ""),
            "chapter": s.get("chapter", ""),
            "veda":    s.get("annotation", {}).get("veda", "Unknown"),
            "domain":  s.get("annotation", {}).get("domain", "Unknown"),
            "branch":  s.get("annotation", {}).get("branch", "Unknown"),
            "keywords": s.get("annotation", {}).get("keywords", []),
        }
        for s in shlokas
    ]

    # ── Encode ────────────────────────────────────────────────────────────
    logger.info(f"Encoding {len(texts)} shlokas with {model_name}…")
    t0    = time.time()
    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → dot product == cosine sim
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.time() - t0
    logger.info(f"Encoded {len(texts)} shlokas in {elapsed:.1f}s  shape={embeddings.shape}")

    # ── Persist ───────────────────────────────────────────────────────────
    np.save(index_dir / "embeddings.npy", embeddings)
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    cfg = {
        "model_name":  model_name,
        "dim":         int(embeddings.shape[1]),
        "num_vectors": len(embeddings),
        "built_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(index_dir / "index_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    audit.log("vector_index_built", num_vectors=len(embeddings), model=model_name)
    logger.info(f"Vector index saved → {index_dir}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Semantic search
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSearchEngine:
    """
    Cosine-similarity search over the pre-built vector index.
    Embeddings are loaded into memory once and reused.
    """

    def __init__(self, index_dir: Path = VECTOR_INDEX_DIR):
        self.index_dir  = Path(index_dir)
        self._embeddings: Optional[np.ndarray] = None
        self._metadata:   Optional[list[dict]] = None
        self._model       = None
        self._model_name: str = EMBEDDING_MODEL
        self._load()

    def _load(self):
        emb_path  = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "metadata.json"
        cfg_path  = self.index_dir / "index_config.json"

        if not emb_path.exists():
            logger.warning("Vector index not found — call build_vector_index() first.")
            return

        self._embeddings = np.load(emb_path)          # (N, D) float32
        with open(meta_path, encoding="utf-8") as f:
            self._metadata = json.load(f)
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            self._model_name = cfg.get("model_name", EMBEDDING_MODEL)

        logger.info(f"Vector index loaded: {len(self._embeddings)} vectors  "
                    f"dim={self._embeddings.shape[1]}")

    def _get_model(self):
        if self._model is None:
            self._model = load_embedding_model(self._model_name)
        return self._model

    @property
    def ready(self) -> bool:
        return self._embeddings is not None and len(self._embeddings) > 0

    def encode(self, text: str) -> np.ndarray:
        """Encode a single query string into a unit vector."""
        model = self._get_model()
        vec   = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Return top_k most similar shlokas to query.

        Returns:
            List of dicts with shloka metadata + 'semantic_score'
        """
        if not self.ready:
            logger.warning("Vector index empty — returning no semantic results.")
            return []

        q_vec  = self.encode(query)                         # (D,)
        scores = (self._embeddings @ q_vec).tolist()        # cosine sim (normalised)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top    = ranked[:top_k]

        results = []
        for idx, score in top:
            hit = dict(self._metadata[idx])
            hit["semantic_score"] = round(float(score), 4)
            results.append(hit)

        logger.debug(f"Semantic search: top score={top[0][1]:.4f} for '{query[:40]}'")
        return results
