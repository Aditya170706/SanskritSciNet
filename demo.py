#!/usr/bin/env python3
"""
Quick-start demo — runs the full Phase 1 pipeline on sample data.
No external services required (uses local JSON fallbacks for Neo4j / Elasticsearch).

Usage:
    python demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.seed_sample_data import seed_sample_data
from config import RAW_DIR, LOGS_DIR
from scripts.logger import setup_logger

logger = setup_logger("demo", LOGS_DIR)


def main():
    print("=" * 70)
    print("  Vedic Shloka Intelligence System — Phase 1 Demo")
    print("=" * 70)

    print("\n[STEP 0] Seeding sample Sanskrit corpus...")
    seed_sample_data(RAW_DIR)

    print("\n[STEP 1-7] Running full pipeline on sample data...")
    from run_pipeline import run_pipeline

    result = run_pipeline(
        input_path=None,       # Raw files already seeded
        use_ai_annotation=True,
        use_neo4j=False,       # Use local JSON fallback
        use_elasticsearch=False,  # Use local index fallback
        seed=42,
    )

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — PHASE 1 RESULTS")
    print("=" * 70)

    seg = result.get("segmentation", {})
    ann = result.get("annotation", {})
    ds = result.get("dataset", {})
    kg = result.get("knowledge_graph", {})
    si = result.get("search_index", {})

    print(f"\n  Segmentation : {seg.get('total_shlokas', 0):>6,} shlokas extracted")
    print(f"  Annotation   : {ann.get('total_annotated', 0):>6,} shlokas annotated")

    if ds:
        print(f"  Dataset      : train={ds.get('train',{}).get('count',0):,} | "
              f"val={ds.get('validation',{}).get('count',0):,} | "
              f"test={ds.get('test',{}).get('count',0):,}")

    if kg:
        stats = kg.get("stats", {})
        n = stats.get("nodes", {})
        r = stats.get("relationships", {})
        print(f"  Knowledge Graph ({kg.get('backend','?')}):")
        for k, v in n.items():
            print(f"    {k:15s}: {v:>4} nodes")
        for k, v in r.items():
            print(f"    {k:15s}: {v:>4} edges")

    if si:
        idx_stats = si.get("stats", {})
        print(f"  Search Index ({si.get('backend','?')}):")
        print(f"    Documents  : {idx_stats.get('total_documents', si.get('indexed',0)):>6,}")
        print(f"    Terms      : {idx_stats.get('unique_terms', 0):>6,}")

    print(f"\n  Total time   : {result.get('elapsed_seconds', 0):.1f}s")
    print("\n  Output locations:")
    from config import (CLEANED_DIR, SEGMENTED_DIR, ANNOTATED_DIR,
                        TRAINING_DIR, KNOWLEDGE_GRAPH_DIR, SEARCH_INDEX_DIR)
    for label, path in [
        ("Cleaned text", CLEANED_DIR),
        ("Segmented JSON", SEGMENTED_DIR),
        ("Annotated JSON", ANNOTATED_DIR),
        ("Training CSVs", TRAINING_DIR),
        ("Knowledge Graph", KNOWLEDGE_GRAPH_DIR),
        ("Search Index", SEARCH_INDEX_DIR),
    ]:
        print(f"    {label:<16}: {path}")

    print("\n  Run search demo:")
    print("    python -m scripts.module7_search_index --search 'गणित' --no-es")
    print("\n  Run tests:")
    print("    pytest tests/ -v")
    print("=" * 70)


if __name__ == "__main__":
    main()
