"""
Master Pipeline Orchestrator — Phase 1
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Runs all 7 modules end-to-end in the correct order.
"""

import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, RAW_DIR, CLEANED_DIR, SEGMENTED_DIR,
    ANNOTATED_DIR, TRAINING_DIR, KNOWLEDGE_GRAPH_DIR,
    SEARCH_INDEX_DIR, METADATA_DIR, LOGS_DIR
)
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("pipeline", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "pipeline_audit.jsonl")


def ensure_directories():
    """Create all required project directories."""
    dirs = [
        RAW_DIR, CLEANED_DIR, SEGMENTED_DIR, ANNOTATED_DIR,
        TRAINING_DIR, KNOWLEDGE_GRAPH_DIR, SEARCH_INDEX_DIR,
        METADATA_DIR, LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("All directories verified/created.")


def run_pipeline(
    input_path: str | Path = None,
    source_name: str = "Sanskrit Corpus",
    author: str = "Unknown",
    publication_year: int = None,
    license_type: str = "Public Domain",
    citation: str = "",
    use_ai_annotation: bool = True,
    use_neo4j: bool = True,
    use_elasticsearch: bool = True,
    seed: int = 42,
) -> dict:
    """
    Execute Phase 1 pipeline end-to-end.

    Args:
        input_path: File or directory to ingest (skip ingestion if None)
        source_name: Source text name
        author: Author of source text
        publication_year: Year of publication
        license_type: License type string
        citation: Full bibliographic citation
        use_ai_annotation: Use AI heuristic annotation
        use_neo4j: Attempt Neo4j connection
        use_elasticsearch: Attempt Elasticsearch connection
        seed: Random seed for dataset split

    Returns:
        Dict summarizing pipeline results
    """
    t0 = time.time()
    summary = {}

    ensure_directories()

    # ── Module 1: Ingestion ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODULE 1 — Data Ingestion")
    logger.info("=" * 60)

    ingested = []
    if input_path:
        from scripts.module1_ingestion import ingest_file, ingest_directory
        p = Path(input_path)
        try:
            if p.is_dir():
                ingested = ingest_directory(
                    p, source_name_prefix=source_name,
                    author=author, publication_year=publication_year,
                    license_type=license_type, citation_reference=citation,
                )
            else:
                entry = ingest_file(
                    p, source_name=source_name,
                    author=author, publication_year=publication_year,
                    license_type=license_type, citation_reference=citation,
                )
                ingested = [entry]
            summary["ingestion"] = {"files_ingested": len(ingested)}
            logger.info(f"Ingestion complete: {len(ingested)} file(s)")
        except Exception as exc:
            logger.error(f"Ingestion failed: {exc}")
            summary["ingestion"] = {"error": str(exc)}
    else:
        logger.info("No input path provided — skipping ingestion (use existing raw files)")
        summary["ingestion"] = {"skipped": True}

    # ── Module 2: Cleaning ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODULE 2 — Data Cleaning")
    logger.info("=" * 60)

    from scripts.module2_cleaning import clean_directory
    cleaned_files = clean_directory(RAW_DIR, CLEANED_DIR, extensions=(".txt", ".xml", ".html"))
    summary["cleaning"] = {"files_cleaned": len(cleaned_files)}
    logger.info(f"Cleaning complete: {len(cleaned_files)} file(s)")

    # ── Module 3: Segmentation ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODULE 3 — Shloka Segmentation")
    logger.info("=" * 60)

    from scripts.module3_segmentation import segment_directory, load_segmented_shlokas
    seg_summaries = segment_directory(CLEANED_DIR, SEGMENTED_DIR)
    all_shlokas = load_segmented_shlokas(SEGMENTED_DIR)
    summary["segmentation"] = {
        "files_segmented": len(seg_summaries),
        "total_shlokas": len(all_shlokas),
    }
    logger.info(f"Segmentation complete: {len(all_shlokas):,} shlokas")

    # ── Module 4: Annotation ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODULE 4 — Annotation")
    logger.info("=" * 60)

    from scripts.module4_annotation import annotate_directory, load_annotated_shlokas
    ann_files = annotate_directory(SEGMENTED_DIR, ANNOTATED_DIR, use_ai=use_ai_annotation)
    annotated = load_annotated_shlokas(ANNOTATED_DIR)
    summary["annotation"] = {
        "files_annotated": len(ann_files),
        "total_annotated": len(annotated),
        "ai_assisted": use_ai_annotation,
    }
    logger.info(f"Annotation complete: {len(annotated):,} shlokas annotated")

    # ── Module 5: Dataset ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODULE 5 — Training Dataset")
    logger.info("=" * 60)

    from scripts.module5_dataset import build_dataset
    ds_result = build_dataset(ANNOTATED_DIR, TRAINING_DIR, seed=seed)
    summary["dataset"] = ds_result
    if ds_result:
        logger.info(
            f"Dataset: train={ds_result.get('train',{}).get('count',0):,} | "
            f"val={ds_result.get('validation',{}).get('count',0):,} | "
            f"test={ds_result.get('test',{}).get('count',0):,}"
        )

    # ── Module 6: Knowledge Graph ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODULE 6 — Knowledge Graph")
    logger.info("=" * 60)

    from scripts.module6_knowledge_graph import build_knowledge_graph
    kg_result = build_knowledge_graph(ANNOTATED_DIR, KNOWLEDGE_GRAPH_DIR, use_neo4j=use_neo4j)
    summary["knowledge_graph"] = kg_result
    logger.info(f"Knowledge graph: {kg_result}")

    # ── Module 7: Search Index ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODULE 7 — Search Index")
    logger.info("=" * 60)

    from scripts.module7_search_index import build_search_index
    si_result = build_search_index(ANNOTATED_DIR, SEARCH_INDEX_DIR, use_elasticsearch=use_elasticsearch)
    summary["search_index"] = si_result
    logger.info(f"Search index: {si_result}")

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 2)

    logger.info("=" * 60)
    logger.info(f"PHASE 1 COMPLETE in {elapsed:.1f}s")
    logger.info("=" * 60)

    audit.log("pipeline_complete", **summary)

    # Save summary report
    summary_path = LOGS_DIR / "phase1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase 1 pipeline: Vedic Shloka Intelligence System"
    )
    parser.add_argument("--input", default=None, help="Input file or directory to ingest")
    parser.add_argument("--source-name", default="Sanskrit Corpus")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--license", default="Public Domain")
    parser.add_argument("--citation", default="")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI annotation")
    parser.add_argument("--no-neo4j", action="store_true")
    parser.add_argument("--no-es", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_pipeline(
        input_path=args.input,
        source_name=args.source_name,
        author=args.author,
        publication_year=args.year,
        license_type=args.license,
        citation=args.citation,
        use_ai_annotation=not args.no_ai,
        use_neo4j=not args.no_neo4j,
        use_elasticsearch=not args.no_es,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(json.dumps(result, indent=2))
