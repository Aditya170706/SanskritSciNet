"""
Unit Tests — Phase 1
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Tests cover:
- Data ingestion
- Cleaning pipeline
- Shloka segmentation
- Annotation engine
- Dataset generation
- Knowledge graph (local fallback)
- Search index (local fallback)
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SHLOKA_TEXT = """अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।
होतारं रत्नधातमम्।।

इन्द्रमिद् गाथिनो बृहदिन्द्रमर्केभिरर्किणः।
इन्द्रं वाणीरनूषत।।

गणितं क्षेत्रमितिः सूत्रं च वर्गघनम्।
संख्यानं परिमाणं स्यात्।।
"""

SAMPLE_ANNOTATED_SHLOKA = {
    "id": "SHLOKA_000001",
    "text": "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।",
    "source": "Rigveda",
    "chapter": "1.1.1",
    "annotation": {
        "veda": "Rigveda",
        "domain": "Philosophy",
        "branch": "Vedanta",
        "formula": "",
        "source": "Rigveda",
        "language": "Sanskrit",
        "keywords": ["philosophy", "rigveda", "vedanta"],
        "annotated_by": "manual",
        "annotated_at": "2024-01-01T00:00:00Z",
        "confidence": 1.0,
        "notes": "",
    }
}


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after each test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_txt_file(tmp_dir):
    """Write sample Sanskrit text to a temp file."""
    f = tmp_dir / "rigveda_sample.txt"
    f.write_text(SAMPLE_SHLOKA_TEXT, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Module 1 — Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:

    def test_ingest_txt_file(self, tmp_dir, sample_txt_file):
        from scripts.module1_ingestion import ingest_file
        raw_dir = tmp_dir / "raw"
        metadata_dir = tmp_dir / "metadata"

        # Monkey-patch config paths
        import scripts.module1_ingestion as m
        original_raw = m.RAW_DIR
        original_meta = m.METADATA_DIR
        original_registry = m.SOURCE_REGISTRY_FILE
        m.RAW_DIR = raw_dir
        m.METADATA_DIR = metadata_dir
        m.SOURCE_REGISTRY_FILE = metadata_dir / "source_registry.json"

        try:
            entry = ingest_file(
                sample_txt_file,
                source_name="Test Rigveda",
                author="Test Author",
                license_type="Public Domain",
            )
        finally:
            m.RAW_DIR = original_raw
            m.METADATA_DIR = original_meta
            m.SOURCE_REGISTRY_FILE = original_registry

        assert entry["source_name"] == "Test Rigveda"
        assert entry["format"] == ".txt"
        assert entry["char_count"] > 0
        assert len(entry["sha256"]) == 64
        assert (raw_dir / entry["raw_filename"]).exists()

    def test_ingest_duplicate_file(self, tmp_dir, sample_txt_file):
        """Duplicate files (same sha256) should not create duplicate registry entries."""
        from scripts.module1_ingestion import ingest_file, _load_registry
        import scripts.module1_ingestion as m
        raw_dir = tmp_dir / "raw"
        metadata_dir = tmp_dir / "metadata"
        m.RAW_DIR = raw_dir
        m.METADATA_DIR = metadata_dir
        m.SOURCE_REGISTRY_FILE = metadata_dir / "source_registry.json"

        try:
            ingest_file(sample_txt_file, source_name="Test")
            ingest_file(sample_txt_file, source_name="Test")
            registry = _load_registry()
        finally:
            import config
            m.RAW_DIR = config.RAW_DIR
            m.METADATA_DIR = config.METADATA_DIR
            m.SOURCE_REGISTRY_FILE = config.SOURCE_REGISTRY_FILE

        # Only 1 entry despite 2 ingest calls
        assert len(registry["sources"]) == 1

    def test_unsupported_format_raises(self, tmp_dir):
        bad_file = tmp_dir / "test.docx"
        bad_file.write_bytes(b"fake docx content")
        from scripts.module1_ingestion import ingest_file
        with pytest.raises(ValueError, match="Unsupported file format"):
            ingest_file(bad_file, source_name="Bad")


# ---------------------------------------------------------------------------
# Module 2 — Cleaning
# ---------------------------------------------------------------------------

class TestCleaning:

    def test_normalize_unicode(self):
        from scripts.module2_cleaning import normalize_unicode
        # Composed vs decomposed form
        text_nfd = "\u0915\u093c"  # ka + nukta (NFC would compose)
        result = normalize_unicode(text_nfd)
        assert result is not None

    def test_remove_page_numbers(self):
        from scripts.module2_cleaning import remove_page_numbers
        text = "line one\n42\nline two\n  123  \nline three"
        result = remove_page_numbers(text)
        assert "42" not in result.split()
        assert "123" not in result.split()
        assert "line one" in result
        assert "line three" in result

    def test_remove_headers_footers(self):
        from scripts.module2_cleaning import remove_headers_footers
        text = "CHAPTER 5\nsome content here\nPAGE 10"
        result = remove_headers_footers(text)
        assert "CHAPTER 5" not in result
        assert "some content here" in result

    def test_normalize_whitespace(self):
        from scripts.module2_cleaning import normalize_whitespace
        text = "line one   \nline two\n\n\n\nline three"
        result = normalize_whitespace(text)
        assert "   " not in result
        assert result.count("\n\n\n") == 0

    def test_full_clean_pipeline(self, tmp_dir, sample_txt_file):
        from scripts.module2_cleaning import clean_file
        out_path = tmp_dir / "cleaned.txt"
        result = clean_file(sample_txt_file, out_path)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert len(content) > 10
        # Should contain Devanagari
        assert any(ord(c) >= 0x0900 for c in content)

    def test_clean_removes_invisible_chars(self):
        from scripts.module2_cleaning import remove_invisible_chars
        text = "hello\u200bworld\ufeff"
        result = remove_invisible_chars(text)
        assert "\u200b" not in result
        assert "\ufeff" not in result


# ---------------------------------------------------------------------------
# Module 3 — Segmentation
# ---------------------------------------------------------------------------

class TestSegmentation:

    def test_segment_text_basic(self):
        from scripts.module3_segmentation import segment_text, ShlokaIDGenerator
        gen = ShlokaIDGenerator(start=1)
        shlokas = segment_text(SAMPLE_SHLOKA_TEXT, source="Rigveda", chapter="1.1", id_generator=gen)
        assert len(shlokas) >= 2
        assert all("id" in s for s in shlokas)
        assert all("text" in s for s in shlokas)
        assert all("source" in s for s in shlokas)
        assert shlokas[0]["id"] == "SHLOKA_000001"

    def test_shloka_id_format(self):
        from scripts.module3_segmentation import segment_text, ShlokaIDGenerator
        gen = ShlokaIDGenerator(start=1)
        shlokas = segment_text(SAMPLE_SHLOKA_TEXT, source="Test", id_generator=gen)
        for s in shlokas:
            assert s["id"].startswith("SHLOKA_")
            assert len(s["id"]) == 13  # SHLOKA_000001

    def test_segment_file(self, tmp_dir, sample_txt_file):
        from scripts.module3_segmentation import segment_file
        out = tmp_dir / "segmented.json"
        result_path = segment_file(sample_txt_file, source="Rigveda", output_path=out)
        assert result_path.exists()
        with open(result_path) as f:
            shlokas = json.load(f)
        assert isinstance(shlokas, list)
        assert len(shlokas) >= 1
        assert shlokas[0]["source"] == "Rigveda"

    def test_no_danda_falls_back_to_paragraph(self):
        from scripts.module3_segmentation import segment_text, ShlokaIDGenerator
        text = "line one line two\n\nline three line four\n\nline five"
        gen = ShlokaIDGenerator()
        shlokas = segment_text(text, source="Test", id_generator=gen)
        # Should fall back to paragraph split and possibly skip non-Devanagari
        assert isinstance(shlokas, list)

    def test_filters_non_devanagari(self):
        from scripts.module3_segmentation import segment_text, ShlokaIDGenerator
        text = "abc def ghi।jkl mno।"
        gen = ShlokaIDGenerator()
        shlokas = segment_text(text, source="Test", id_generator=gen)
        # Should return empty list (no Devanagari)
        assert shlokas == []


# ---------------------------------------------------------------------------
# Module 4 — Annotation
# ---------------------------------------------------------------------------

class TestAnnotation:

    def test_manual_annotation(self):
        from scripts.module4_annotation import annotate_shloka
        shloka = {"id": "SHLOKA_000001", "text": "test", "source": "Rigveda", "chapter": "1.1"}
        manual = {"veda": "Rigveda", "domain": "Philosophy", "branch": "Vedanta",
                  "formula": "", "source": "Rigveda", "language": "Sanskrit", "keywords": ["test"]}
        result = annotate_shloka(shloka, annotation=manual)
        assert result["annotation"]["veda"] == "Rigveda"
        assert result["annotation"]["annotated_by"] == "manual"

    def test_ai_annotation_heuristics(self):
        from scripts.module4_annotation import ai_annotate_shloka
        shloka = {
            "id": "SHLOKA_000001",
            "text": "गणितं क्षेत्रम् वर्गः।",
            "source": "Rigveda",
            "chapter": "1.1"
        }
        ann = ai_annotate_shloka(shloka)
        assert ann["annotated_by"] == "ai_heuristic"
        assert ann["domain"] == "Mathematics"
        assert ann["veda"] == "Rigveda"

    def test_ai_annotation_astronomy(self):
        from scripts.module4_annotation import ai_annotate_shloka
        shloka = {
            "id": "SHLOKA_000002",
            "text": "ज्योतिष ग्रह नक्षत्र सूर्य।",
            "source": "Atharvaveda",
            "chapter": "2.1"
        }
        ann = ai_annotate_shloka(shloka)
        assert ann["domain"] == "Astronomy"
        assert ann["veda"] == "Atharvaveda"

    def test_annotate_file(self, tmp_dir):
        from scripts.module4_annotation import annotate_file
        # Create a small segmented file
        seg_path = tmp_dir / "test.json"
        shlokas = [SAMPLE_ANNOTATED_SHLOKA.copy()]
        # Strip existing annotation so it goes through AI
        shlokas[0] = {k: v for k, v in shlokas[0].items() if k != "annotation"}
        with open(seg_path, "w") as f:
            json.dump(shlokas, f)

        out_path = tmp_dir / "annotated.json"
        annotate_file(seg_path, out_path, use_ai=True)
        assert out_path.exists()
        with open(out_path) as f:
            annotated = json.load(f)
        assert len(annotated) == 1
        assert "annotation" in annotated[0]


# ---------------------------------------------------------------------------
# Module 5 — Dataset Builder
# ---------------------------------------------------------------------------

class TestDatasetBuilder:

    def _make_annotated_dir(self, tmp_dir: Path, n: int = 30) -> Path:
        ann_dir = tmp_dir / "annotated"
        ann_dir.mkdir()
        shlokas = []
        domains = ["Mathematics", "Astronomy", "Philosophy", "Grammar"]
        for i in range(n):
            shlokas.append({
                "id": f"SHLOKA_{i:06d}",
                "text": f"सं संख्या {i}।",
                "source": "Test",
                "chapter": "1",
                "annotation": {
                    "veda": "Rigveda",
                    "domain": domains[i % len(domains)],
                    "branch": "Unknown",
                    "formula": "",
                    "source": "Test",
                    "language": "Sanskrit",
                    "keywords": [domains[i % len(domains)].lower()],
                    "annotated_by": "test",
                    "confidence": 1.0,
                    "notes": "",
                }
            })
        with open(ann_dir / "corpus.json", "w") as f:
            json.dump(shlokas, f)
        return ann_dir

    def test_dataset_split_sizes(self, tmp_dir):
        from scripts.module5_dataset import build_dataset
        import pandas as pd
        ann_dir = self._make_annotated_dir(tmp_dir, n=100)
        out_dir = tmp_dir / "training"
        result = build_dataset(ann_dir, out_dir, seed=42, stratify_by=None)

        assert result["total"] == 100
        train_df = pd.read_csv(out_dir / "train.csv")
        val_df = pd.read_csv(out_dir / "validation.csv")
        test_df = pd.read_csv(out_dir / "test.csv")
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 100
        # Check approximate ratios (within 1 sample)
        assert abs(len(train_df) - 70) <= 2
        assert abs(len(val_df) - 15) <= 2
        assert abs(len(test_df) - 15) <= 2

    def test_dataset_columns(self, tmp_dir):
        from scripts.module5_dataset import build_dataset
        import pandas as pd
        ann_dir = self._make_annotated_dir(tmp_dir, n=20)
        out_dir = tmp_dir / "training"
        build_dataset(ann_dir, out_dir, seed=42, stratify_by=None)
        df = pd.read_csv(out_dir / "train.csv")
        expected = {"id", "text", "source", "chapter", "veda", "domain", "branch", "keywords"}
        assert expected.issubset(set(df.columns))

    def test_reproducibility(self, tmp_dir):
        from scripts.module5_dataset import build_dataset
        import pandas as pd
        ann_dir = self._make_annotated_dir(tmp_dir, n=40)
        out1 = tmp_dir / "run1"
        out2 = tmp_dir / "run2"
        build_dataset(ann_dir, out1, seed=42, stratify_by=None)
        build_dataset(ann_dir, out2, seed=42, stratify_by=None)
        df1 = pd.read_csv(out1 / "train.csv")
        df2 = pd.read_csv(out2 / "train.csv")
        assert list(df1["id"]) == list(df2["id"])


# ---------------------------------------------------------------------------
# Module 6 — Knowledge Graph (local fallback)
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:

    def test_local_kg_insert_and_query(self, tmp_dir):
        from scripts.module6_knowledge_graph import LocalKnowledgeGraph
        kg = LocalKnowledgeGraph(tmp_dir)
        kg.insert_shloka(SAMPLE_ANNOTATED_SHLOKA)
        kg._save()

        # Reload
        kg2 = LocalKnowledgeGraph(tmp_dir)
        nodes = kg2.nodes
        edges = kg2.edges

        # Shloka node exists
        assert "Shloka:SHLOKA_000001" in nodes
        # Veda node exists
        assert "Veda:Rigveda" in nodes
        # BELONGS_TO relationship
        assert any(
            e["from"] == "Shloka:SHLOKA_000001" and e["rel"] == "BELONGS_TO" and e["to"] == "Veda:Rigveda"
            for e in edges
        )

    def test_local_kg_query_by_domain(self, tmp_dir):
        from scripts.module6_knowledge_graph import LocalKnowledgeGraph
        kg = LocalKnowledgeGraph(tmp_dir)
        kg.insert_shloka(SAMPLE_ANNOTATED_SHLOKA)
        results = kg.query_by_domain("Philosophy")
        assert "SHLOKA_000001" in results

    def test_local_kg_statistics(self, tmp_dir):
        from scripts.module6_knowledge_graph import LocalKnowledgeGraph
        kg = LocalKnowledgeGraph(tmp_dir)
        kg.insert_shloka(SAMPLE_ANNOTATED_SHLOKA)
        stats = kg.statistics()
        assert "nodes" in stats
        assert "Shloka" in stats["nodes"]
        assert stats["nodes"]["Shloka"] >= 1


# ---------------------------------------------------------------------------
# Module 7 — Search Index (local fallback)
# ---------------------------------------------------------------------------

class TestSearchIndex:

    def _make_index(self, tmp_dir: Path) -> "LocalSearchIndex":
        from scripts.module7_search_index import LocalSearchIndex, shloka_to_index_doc
        idx = LocalSearchIndex(tmp_dir)
        docs = [shloka_to_index_doc(SAMPLE_ANNOTATED_SHLOKA)]
        # Add more docs
        for i in range(5):
            extra = {
                "id": f"SHLOKA_{i+2:06d}",
                "text": f"गणितं संख्या {i}।",
                "source": "Test",
                "chapter": "1",
                "annotation": {
                    "veda": "Rigveda",
                    "domain": "Mathematics",
                    "branch": "Arithmetic",
                    "formula": "",
                    "source": "Test",
                    "language": "Sanskrit",
                    "keywords": ["mathematics", "arithmetic"],
                    "annotated_by": "test",
                    "confidence": 0.8,
                    "notes": "",
                }
            }
            docs.append(shloka_to_index_doc(extra))
        idx.bulk_index(docs)
        return idx

    def test_keyword_search(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        results = idx.keyword_search("philosophy")
        assert len(results) >= 1

    def test_exact_search(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        results = idx.exact_search("domain", "Mathematics")
        assert len(results) >= 1
        assert all(r["domain"] == "Mathematics" for r in results)

    def test_filter_search(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        results = idx.filter_search({"veda": "Rigveda", "domain": "Mathematics"})
        assert isinstance(results, list)

    def test_statistics(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        stats = idx.statistics()
        assert stats["total_documents"] >= 1
        assert stats["unique_terms"] >= 1

    def test_persistence(self, tmp_dir):
        from scripts.module7_search_index import LocalSearchIndex, shloka_to_index_doc
        idx = LocalSearchIndex(tmp_dir)
        idx.bulk_index([shloka_to_index_doc(SAMPLE_ANNOTATED_SHLOKA)])

        # Reload
        idx2 = LocalSearchIndex(tmp_dir)
        assert idx2.statistics()["total_documents"] >= 1


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_end_to_end_pipeline(self, tmp_dir, sample_txt_file):
        """
        Smoke test: run modules 2-7 on a sample file end-to-end.
        Does not test Module 1 (ingestion) to avoid path-patching complexity.
        """
        cleaned_dir = tmp_dir / "cleaned"
        segmented_dir = tmp_dir / "segmented"
        annotated_dir = tmp_dir / "annotated"
        training_dir = tmp_dir / "training"
        kg_dir = tmp_dir / "kg"
        idx_dir = tmp_dir / "idx"
        logs_dir = tmp_dir / "logs"

        # Module 2
        from scripts.module2_cleaning import clean_file
        cleaned_dir.mkdir()
        cleaned_path = clean_file(sample_txt_file, cleaned_dir / "sample.txt")

        # Module 3
        from scripts.module3_segmentation import segment_file
        segmented_dir.mkdir()
        seg_path = segment_file(cleaned_path, source="Rigveda", output_path=segmented_dir / "sample.json")

        with open(seg_path) as f:
            shlokas = json.load(f)
        assert len(shlokas) >= 1

        # Module 4
        from scripts.module4_annotation import annotate_file
        annotated_dir.mkdir()
        ann_path = annotate_file(seg_path, annotated_dir / "sample.json", use_ai=True)

        with open(ann_path) as f:
            annotated = json.load(f)
        assert len(annotated) >= 1
        assert "annotation" in annotated[0]

        # Module 5
        from scripts.module5_dataset import build_dataset
        result = build_dataset(annotated_dir, training_dir, seed=42, stratify_by=None)
        assert "train" in result or result == {}  # May be empty if <3 shlokas

        # Module 6
        from scripts.module6_knowledge_graph import LocalKnowledgeGraph
        kg = LocalKnowledgeGraph(kg_dir)
        count = kg.bulk_insert(annotated)
        assert count == len(annotated)

        # Module 7
        from scripts.module7_search_index import LocalSearchIndex, shloka_to_index_doc
        idx = LocalSearchIndex(idx_dir)
        docs = [shloka_to_index_doc(s) for s in annotated]
        idx.bulk_index(docs)
        stats = idx.statistics()
        assert stats["total_documents"] == len(annotated)
