"""
tests/phase3/test_phase3.py
────────────────────────────
Unit + integration tests for Phase 3.
All tests run without external services (Neo4j, Redis, Elasticsearch)
and without internet access (no model downloads).
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


SAMPLE_SHLOKAS_ANNOTATED = [
    {
        "id": f"SHLOKA_{i:06d}",
        "text": t,
        "source": src,
        "chapter": "1",
        "annotation": {
            "veda": veda, "domain": domain, "branch": branch,
            "formula": formula, "source": src, "language": "Sanskrit",
            "keywords": kws, "annotated_by": "test", "confidence": 1.0, "notes": "",
        }
    }
    for i, (t, src, veda, domain, branch, formula, kws) in enumerate([
        ("समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।",
         "Sulba Sutra", "Yajurveda", "Mathematics", "Geometry",
         "square_area", ["mathematics", "geometry"]),
        ("अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।",
         "Rigveda", "Rigveda", "Philosophy", "Vedanta",
         "", ["philosophy", "rigveda"]),
        ("गणितं क्षेत्रमितिः सूत्रं च वर्गघनम्।",
         "Vedic Mathematics", "Unknown", "Mathematics", "Arithmetic",
         "arithmetic_sum", ["mathematics", "arithmetic"]),
        ("ज्योतिष ग्रह नक्षत्र सूर्य।",
         "Atharvaveda", "Atharvaveda", "Astronomy", "Jyotisha",
         "", ["astronomy", "jyotisha"]),
        ("त्रिभुजस्य क्षेत्रफलं भूमिश्चशीर्षलम्बार्धगुणितम्।",
         "Sulba Sutra", "Yajurveda", "Mathematics", "Geometry",
         "triangle_area", ["mathematics", "geometry", "triangle"]),
    ])
]


def _make_annotated_dir(tmp_dir: Path) -> Path:
    ann_dir = tmp_dir / "annotated"
    ann_dir.mkdir()
    with open(ann_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(SAMPLE_SHLOKAS_ANNOTATED, f, ensure_ascii=False)
    return ann_dir


def _make_search_index(tmp_dir: Path) -> Path:
    """Build a local search index in tmp_dir."""
    idx_dir = tmp_dir / "search_index"
    from scripts.module7_search_index import LocalSearchIndex, shloka_to_index_doc
    idx  = LocalSearchIndex(idx_dir)
    docs = [shloka_to_index_doc(s) for s in SAMPLE_SHLOKAS_ANNOTATED]
    idx.bulk_index(docs)
    return idx_dir


def _make_kg(tmp_dir: Path) -> Path:
    """Build a local KG in tmp_dir."""
    kg_dir = tmp_dir / "kg"
    from scripts.module6_knowledge_graph import LocalKnowledgeGraph
    kg = LocalKnowledgeGraph(kg_dir)
    kg.bulk_insert(SAMPLE_SHLOKAS_ANNOTATED)
    return kg_dir


def _make_vector_index(tmp_dir: Path, ann_dir: Path) -> Path:
    """Build a tiny fake vector index (random embeddings, no download)."""
    idx_dir = tmp_dir / "vector_index"
    idx_dir.mkdir()
    n   = len(SAMPLE_SHLOKAS_ANNOTATED)
    dim = 8   # tiny dim for tests
    emb = np.random.randn(n, dim).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(idx_dir / "embeddings.npy", emb)

    metadata = [
        {
            "id":     s["id"],
            "text":   s["text"],
            "source": s["source"],
            "chapter": s.get("chapter", ""),
            "veda":   s["annotation"]["veda"],
            "domain": s["annotation"]["domain"],
            "branch": s["annotation"]["branch"],
            "keywords": s["annotation"]["keywords"],
        }
        for s in SAMPLE_SHLOKAS_ANNOTATED
    ]
    with open(idx_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    cfg = {"model_name": "fake", "dim": dim, "num_vectors": n,
           "built_at": "2024-01-01T00:00:00Z"}
    with open(idx_dir / "index_config.json", "w") as f:
        json.dump(cfg, f)
    return idx_dir


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Search tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticSearch:

    def test_search_returns_top_k(self, tmp_dir):
        """search() must return exactly top_k results (or fewer if index is small)."""
        ann_dir = _make_annotated_dir(tmp_dir)
        idx_dir = _make_vector_index(tmp_dir, ann_dir)

        from retrieval.semantic_search import SemanticSearchEngine
        engine = SemanticSearchEngine(index_dir=idx_dir)

        # Provide a pre-encoded random query vector to skip real tokenisation
        q_vec = np.random.randn(8).astype(np.float32)
        q_vec /= np.linalg.norm(q_vec)
        engine._model = None  # Prevent model load

        # Patch encode to return our vector
        engine.encode = lambda text: q_vec

        results = engine.search("test query", top_k=3)
        assert len(results) == 3

    def test_search_includes_semantic_score(self, tmp_dir):
        """Each result must include a 'semantic_score' field."""
        ann_dir = _make_annotated_dir(tmp_dir)
        idx_dir = _make_vector_index(tmp_dir, ann_dir)

        from retrieval.semantic_search import SemanticSearchEngine
        engine = SemanticSearchEngine(index_dir=idx_dir)
        q_vec  = np.random.randn(8).astype(np.float32)
        q_vec /= np.linalg.norm(q_vec)
        engine.encode = lambda text: q_vec

        results = engine.search("query", top_k=2)
        for r in results:
            assert "semantic_score" in r
            assert -1.0 <= r["semantic_score"] <= 1.0

    def test_search_sorted_by_score(self, tmp_dir):
        """Results must be in descending order of semantic_score."""
        ann_dir = _make_annotated_dir(tmp_dir)
        idx_dir = _make_vector_index(tmp_dir, ann_dir)

        from retrieval.semantic_search import SemanticSearchEngine
        engine = SemanticSearchEngine(index_dir=idx_dir)
        q_vec  = np.ones(8, dtype=np.float32) / np.sqrt(8)
        engine.encode = lambda text: q_vec

        results = engine.search("query", top_k=5)
        scores  = [r["semantic_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_not_ready_returns_empty(self, tmp_dir):
        """If index missing, search must return empty list not raise."""
        from retrieval.semantic_search import SemanticSearchEngine
        engine = SemanticSearchEngine(index_dir=tmp_dir / "nonexistent")
        results = engine.search("test")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic Search tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolicSearch:

    def test_keyword_search_returns_results(self, tmp_dir):
        idx_dir = _make_search_index(tmp_dir)
        from retrieval.symbolic_search import SymbolicSearchEngine
        engine  = SymbolicSearchEngine(index_dir=idx_dir)
        results = engine.keyword_search("गणित", top_k=3)
        assert isinstance(results, list)

    def test_exact_search_domain(self, tmp_dir):
        idx_dir = _make_search_index(tmp_dir)
        from retrieval.symbolic_search import SymbolicSearchEngine
        engine  = SymbolicSearchEngine(index_dir=idx_dir)
        results = engine.exact_search("domain", "Mathematics")
        assert all(r.get("domain") == "Mathematics" for r in results)

    def test_filter_search(self, tmp_dir):
        idx_dir = _make_search_index(tmp_dir)
        from retrieval.symbolic_search import SymbolicSearchEngine
        engine  = SymbolicSearchEngine(index_dir=idx_dir)
        results = engine.filter_search({"veda": "Rigveda"})
        assert all(r.get("veda") == "Rigveda" for r in results)

    def test_keyword_score_in_range(self, tmp_dir):
        idx_dir = _make_search_index(tmp_dir)
        from retrieval.symbolic_search import SymbolicSearchEngine
        engine  = SymbolicSearchEngine(index_dir=idx_dir)
        results = engine.keyword_search("mathematics geometry", top_k=5)
        for r in results:
            assert 0.0 <= r.get("keyword_score", 0) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# KG Query Engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKGQueryEngine:

    def test_find_by_domain(self, tmp_dir):
        kg_dir = _make_kg(tmp_dir)
        from graph.graph_queries import KnowledgeGraphQueryEngine
        kg     = KnowledgeGraphQueryEngine(kg_dir=kg_dir, use_neo4j=False)
        result = kg.find_by_domain("Mathematics")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_find_by_veda(self, tmp_dir):
        kg_dir = _make_kg(tmp_dir)
        from graph.graph_queries import KnowledgeGraphQueryEngine
        kg     = KnowledgeGraphQueryEngine(kg_dir=kg_dir, use_neo4j=False)
        result = kg.find_by_veda("Rigveda")
        assert isinstance(result, list)

    def test_get_shloka_relations(self, tmp_dir):
        kg_dir = _make_kg(tmp_dir)
        from graph.graph_queries import KnowledgeGraphQueryEngine
        kg     = KnowledgeGraphQueryEngine(kg_dir=kg_dir, use_neo4j=False)
        rels   = kg.get_shloka_relations("SHLOKA_000000")
        assert isinstance(rels, list)
        assert len(rels) >= 1
        assert "rel" in rels[0]

    def test_find_formulas(self, tmp_dir):
        kg_dir = _make_kg(tmp_dir)
        from graph.graph_queries import KnowledgeGraphQueryEngine
        kg     = KnowledgeGraphQueryEngine(kg_dir=kg_dir, use_neo4j=False)
        # SHLOKA_000000 has formula "square_area"
        formulas = kg.find_formulas("SHLOKA_000000")
        assert isinstance(formulas, list)

    def test_graph_statistics(self, tmp_dir):
        kg_dir = _make_kg(tmp_dir)
        from graph.graph_queries import KnowledgeGraphQueryEngine
        kg     = KnowledgeGraphQueryEngine(kg_dir=kg_dir, use_neo4j=False)
        stats  = kg.graph_statistics()
        assert "nodes" in stats
        assert "Shloka" in stats["nodes"]


# ─────────────────────────────────────────────────────────────────────────────
# Math Content Detector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMathDetector:

    def test_math_domain_triggers_flag(self):
        from reasoning.math_engine import MathContentDetector
        det    = MathContentDetector()
        result = det.detect("any text", domain="Mathematics")
        assert result["math_flag"] is True
        assert result["confidence"] >= 0.80

    def test_non_math_domain_no_flag(self):
        from reasoning.math_engine import MathContentDetector
        det    = MathContentDetector()
        result = det.detect("अग्निमीळे पुरोहितं।", domain="Philosophy", branch="Vedanta")
        assert result["math_flag"] is False

    def test_devanagari_keyword_triggers(self):
        from reasoning.math_engine import MathContentDetector
        det    = MathContentDetector()
        result = det.detect("क्षेत्रफलं भुजवर्गः।", domain="Unknown")
        assert result["math_flag"] is True
        assert len(result["triggers"]) >= 1

    def test_confidence_in_range(self):
        from reasoning.math_engine import MathContentDetector
        det = MathContentDetector()
        for domain in ("Mathematics", "Philosophy", "Unknown"):
            result = det.detect("test", domain=domain)
            assert 0.0 <= result["confidence"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Formula Extractor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFormulaExtractor:

    def test_keyword_hint_pythagorean(self):
        from reasoning.math_engine import FormulaExtractor
        ext    = FormulaExtractor()
        result = ext.extract("diagonal of a square", domain="Geometry", branch="Geometry")
        assert result is not None
        assert "pythagorean" in result.get("name", "").lower() or \
               "c**2" in result.get("expr", "")

    def test_triangle_area_keyword(self):
        from reasoning.math_engine import FormulaExtractor
        ext    = FormulaExtractor()
        result = ext.extract("त्रिभुजस्य क्षेत्रफलं।", domain="Mathematics", branch="Geometry")
        assert result is not None

    def test_no_match_returns_none(self):
        from reasoning.math_engine import FormulaExtractor
        ext    = FormulaExtractor()
        result = ext.extract("purely ritual philosophical text", domain="Philosophy")
        assert result is None

    def test_domain_heuristic_geometry(self):
        from reasoning.math_engine import FormulaExtractor
        ext    = FormulaExtractor()
        result = ext.extract("some text", domain="Mathematics", branch="Geometry")
        assert result is not None
        assert result.get("match_method") in ("keyword_hint", "domain_heuristic",
                                               "knowledge_graph", "knowledge_graph_raw")


# ─────────────────────────────────────────────────────────────────────────────
# Formula Generator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFormulaGenerator:

    def test_returns_valid_sympy(self):
        from reasoning.math_engine import FormulaGenerator
        gen    = FormulaGenerator()
        result = gen.generate("वर्ग क्षेत्र समान", domain="Mathematics", branch="Geometry")
        assert "expr"  in result
        assert "latex" in result
        assert result["sympy_valid"] is True

    def test_square_pattern_detected(self):
        from reasoning.math_engine import FormulaGenerator
        gen    = FormulaGenerator()
        result = gen.generate("square sum of bhuja", domain="Mathematics", branch="Geometry")
        assert "**2" in result["expr"] or "sqrt" in result["expr"]

    def test_domain_fallback(self):
        from reasoning.math_engine import FormulaGenerator
        gen    = FormulaGenerator()
        result = gen.generate("unknown text", domain="Unknown", branch="Unknown")
        assert result["expr"] is not None
        assert len(result["expr"]) > 0

    def test_trigonometry_branch(self):
        from reasoning.math_engine import FormulaGenerator
        gen    = FormulaGenerator()
        result = gen.generate("sin cos angle", domain="Mathematics", branch="Trigonometry")
        # SymPy simplifies sin²x+cos²x-1 → 0; any valid expr is acceptable
        assert result["expr"] is not None
        assert result["sympy_valid"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Estimator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceEstimator:

    def test_output_in_range(self):
        from reasoning.reasoning_engine import ConfidenceEstimator
        est = ConfidenceEstimator()
        for m, r, g in [(0.9, 0.8, 0.7), (0.1, 0.2, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)]:
            score = est.estimate(m, r, g)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_high_inputs_give_high_score(self):
        from reasoning.reasoning_engine import ConfidenceEstimator
        est   = ConfidenceEstimator()
        score = est.estimate(1.0, 1.0, 1.0)
        assert score >= 0.95

    def test_zero_inputs_give_zero(self):
        from reasoning.reasoning_engine import ConfidenceEstimator
        est   = ConfidenceEstimator()
        score = est.estimate(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_weights_respected(self):
        from reasoning.reasoning_engine import ConfidenceEstimator
        # Extreme weight on model only
        est   = ConfidenceEstimator(model_w=1.0, retrieval_w=0.0, graph_w=0.0)
        score = est.estimate(model_conf=0.75, retrieval_conf=0.0, graph_conf=0.0)
        assert abs(score - 0.75) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Cache tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCache:

    def test_set_and_get(self):
        from cache.redis_cache import RetrievalCache
        cache = RetrievalCache()
        key   = cache.make_key("test shloka")
        cache.set(key, {"result": 42})
        hit = cache.get(key)
        assert hit is not None
        assert hit["result"] == 42

    def test_miss_returns_none(self):
        from cache.redis_cache import RetrievalCache
        cache = RetrievalCache()
        hit   = cache.get("nonexistent_key_xyz_123")
        assert hit is None

    def test_make_key_deterministic(self):
        from cache.redis_cache import RetrievalCache
        k1 = RetrievalCache.make_key("hello")
        k2 = RetrievalCache.make_key("hello")
        assert k1 == k2

    def test_make_key_different_inputs(self):
        from cache.redis_cache import RetrievalCache
        k1 = RetrievalCache.make_key("shlokaA")
        k2 = RetrievalCache.make_key("shlokaB")
        assert k1 != k2

    def test_stats_returned(self):
        from cache.redis_cache import RetrievalCache
        cache = RetrievalCache()
        stats = cache.stats()
        assert "backend" in stats


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridEngine:

    def _make_engine(self, tmp_dir):
        idx_dir = _make_vector_index(tmp_dir, _make_annotated_dir(tmp_dir))
        si_dir  = _make_search_index(tmp_dir)
        kg_dir  = _make_kg(tmp_dir)

        import phase3_config as cfg
        orig_vec = cfg.VECTOR_INDEX_DIR
        orig_si  = cfg.SEARCH_INDEX_DIR
        cfg.VECTOR_INDEX_DIR = idx_dir
        cfg.SEARCH_INDEX_DIR = si_dir

        from retrieval.semantic_search import SemanticSearchEngine
        from retrieval.symbolic_search import SymbolicSearchEngine
        from retrieval.hybrid_engine   import HybridRetrievalEngine
        from graph.graph_queries       import KnowledgeGraphQueryEngine

        sem = SemanticSearchEngine(index_dir=idx_dir)
        q_vec = np.ones(8, dtype=np.float32) / np.sqrt(8)
        sem.encode = lambda text: q_vec

        sym = SymbolicSearchEngine(index_dir=si_dir)
        kg  = KnowledgeGraphQueryEngine(kg_dir=kg_dir, use_neo4j=False)

        engine = HybridRetrievalEngine(top_k=3, kg_engine=kg)
        engine._semantic = sem
        engine._symbolic = sym

        cfg.VECTOR_INDEX_DIR = orig_vec
        cfg.SEARCH_INDEX_DIR = orig_si

        return engine

    def test_retrieve_returns_list(self, tmp_dir):
        engine  = self._make_engine(tmp_dir)
        results = engine.retrieve("गणित सूत्र", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_has_final_score(self, tmp_dir):
        engine  = self._make_engine(tmp_dir)
        results = engine.retrieve("test", top_k=3)
        for r in results:
            assert "final_score" in r
            assert 0.0 <= r["final_score"] <= 1.0

    def test_retrieve_sorted(self, tmp_dir):
        engine  = self._make_engine(tmp_dir)
        results = engine.retrieve("वर्ग भुज", top_k=5)
        scores  = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_top_k(self, tmp_dir):
        engine  = self._make_engine(tmp_dir)
        results = engine.retrieve("text", top_k=2)
        assert len(results) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# Integration — full pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_output_schema(self, tmp_dir):
        """
        End-to-end: build all indices, run pipeline.query(), verify schema.
        Uses stub inference (no Phase 2 models needed).
        """
        ann_dir = _make_annotated_dir(tmp_dir)
        idx_dir = _make_vector_index(tmp_dir, ann_dir)
        si_dir  = _make_search_index(tmp_dir)
        kg_dir  = _make_kg(tmp_dir)

        import phase3_config as cfg
        orig = {
            "VECTOR_INDEX_DIR": cfg.VECTOR_INDEX_DIR,
            "SEARCH_INDEX_DIR": cfg.SEARCH_INDEX_DIR,
            "KG_DIR":           cfg.KG_DIR,
            "ANNOTATED_DIR":    cfg.ANNOTATED_DIR,
        }
        cfg.VECTOR_INDEX_DIR = idx_dir
        cfg.SEARCH_INDEX_DIR = si_dir
        cfg.KG_DIR           = kg_dir
        cfg.ANNOTATED_DIR    = ann_dir

        try:
            from retrieval.semantic_search import SemanticSearchEngine
            from retrieval.symbolic_search import SymbolicSearchEngine
            from retrieval.hybrid_engine   import HybridRetrievalEngine
            from graph.graph_queries       import KnowledgeGraphQueryEngine
            from reasoning.math_engine     import (MathContentDetector,
                                                    FormulaExtractor, FormulaGenerator)
            from reasoning.reasoning_engine import ReasoningEngine, ConfidenceEstimator
            from p3_pipeline import _StubInference

            sem = SemanticSearchEngine(index_dir=idx_dir)
            q   = np.ones(8, dtype=np.float32) / np.sqrt(8)
            sem.encode = lambda t: q

            sym = SymbolicSearchEngine(index_dir=si_dir)
            kg  = KnowledgeGraphQueryEngine(kg_dir=kg_dir, use_neo4j=False)
            ret = HybridRetrievalEngine(top_k=3, kg_engine=kg)
            ret._semantic = sem
            ret._symbolic = sym

            engine = ReasoningEngine(
                inference_pipeline   = _StubInference(),
                hybrid_engine        = ret,
                kg_engine            = kg,
                math_detector        = MathContentDetector(),
                formula_extractor    = FormulaExtractor(kg_engine=kg),
                formula_generator    = FormulaGenerator(),
                confidence_estimator = ConfidenceEstimator(),
            )

            result = engine.reason("समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।")

            # Validate required keys
            for key in ("veda", "domain", "branch", "math_flag", "confidence",
                        "reasoning", "top_retrieved", "elapsed_ms",
                        "confidence_breakdown", "phase2_predictions"):
                assert key in result, f"Missing key: {key}"

            # Validate types
            assert isinstance(result["math_flag"],  bool)
            assert isinstance(result["reasoning"],  list)
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0
            assert len(result["reasoning"]) >= 1

        finally:
            for k, v in orig.items():
                setattr(cfg, k, v)
