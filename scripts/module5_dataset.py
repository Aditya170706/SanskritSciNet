"""
Module 5 — Training Dataset Builder
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Converts annotated shlokas into train/validation/test CSV splits.
Split ratio: 70% train / 15% validation / 15% test
Saves to data/training/
"""

import json
import random
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANNOTATED_DIR, TRAINING_DIR, LOGS_DIR, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("dataset_builder", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "dataset_audit.jsonl")


# ---------------------------------------------------------------------------
# Schema flattening
# ---------------------------------------------------------------------------

def flatten_shloka(shloka: dict) -> dict:
    """
    Flatten a nested annotated shloka dict into a tabular row.

    Returns:
        Flat dict suitable for CSV export
    """
    ann = shloka.get("annotation", {})
    keywords = ann.get("keywords", [])
    if isinstance(keywords, list):
        keywords_str = "|".join(keywords)
    else:
        keywords_str = str(keywords)

    return {
        "id": shloka.get("id", ""),
        "text": shloka.get("text", ""),
        "source": shloka.get("source", ""),
        "chapter": shloka.get("chapter", ""),
        "veda": ann.get("veda", "Unknown"),
        "domain": ann.get("domain", "Unknown"),
        "branch": ann.get("branch", "Unknown"),
        "formula": ann.get("formula", ""),
        "language": ann.get("language", "Sanskrit"),
        "keywords": keywords_str,
        "annotated_by": ann.get("annotated_by", ""),
        "confidence": ann.get("confidence", 1.0),
        "notes": ann.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def build_dataset(
    annotated_dir: str | Path = None,
    output_dir: str | Path = None,
    seed: int = 42,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
    stratify_by: Optional[str] = "domain",
) -> dict:
    """
    Build train/validation/test CSV splits from annotated shlokas.

    Args:
        annotated_dir: Directory containing annotated JSON files
        output_dir: Directory to save CSV files
        seed: Random seed for reproducibility
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify_by: Field to stratify split on (None for random split)

    Returns:
        Dict with split sizes and file paths
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    annotated_dir = Path(annotated_dir or ANNOTATED_DIR)
    output_dir = Path(output_dir or TRAINING_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all annotated shlokas
    all_shlokas = []
    for jf in sorted(annotated_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            shlokas = json.load(f)
        all_shlokas.extend(shlokas)

    if not all_shlokas:
        logger.warning("No annotated shlokas found. Run Module 4 first.")
        return {}

    logger.info(f"Loaded {len(all_shlokas):,} annotated shlokas for dataset building")

    # Flatten
    rows = [flatten_shloka(s) for s in all_shlokas]
    df = pd.DataFrame(rows)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    if stratify_by and stratify_by in df.columns:
        logger.info(f"Using stratified split on '{stratify_by}'")
        train_frames, val_frames, test_frames = [], [], []

        for _, group in df.groupby(stratify_by):
            n = len(group)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val
            if n_test < 0:
                n_val = n - n_train
                n_test = 0

            train_frames.append(group.iloc[:n_train])
            val_frames.append(group.iloc[n_train:n_train + n_val])
            test_frames.append(group.iloc[n_train + n_val:])

        train_df = pd.concat(train_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df = pd.concat(val_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df = pd.concat(test_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

    # Save
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    result = {
        "total": len(df),
        "train": {"count": len(train_df), "path": str(train_path)},
        "validation": {"count": len(val_df), "path": str(val_path)},
        "test": {"count": len(test_df), "path": str(test_path)},
        "columns": list(df.columns),
        "seed": seed,
    }

    # Save split manifest
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    audit.log("dataset_built", **result)
    logger.info(
        f"Dataset ready: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}"
    )
    logger.info(f"Files saved to {output_dir}")
    return result


def dataset_statistics(output_dir: str | Path = None) -> dict:
    """
    Print and return basic statistics about the generated dataset.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required")

    output_dir = Path(output_dir or TRAINING_DIR)
    stats = {}
    for split in ("train", "validation", "test"):
        csv_path = output_dir / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            stats[split] = {
                "count": len(df),
                "domain_distribution": df["domain"].value_counts().to_dict() if "domain" in df else {},
                "veda_distribution": df["veda"].value_counts().to_dict() if "veda" in df else {},
            }
            logger.info(f"{split}: {len(df):,} rows")
            if "domain" in df:
                logger.info(f"  Domain distribution: {dict(df['domain'].value_counts())}")
        else:
            logger.warning(f"File not found: {csv_path}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build training dataset from annotated shlokas")
    parser.add_argument("--annotated-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args()

    result = build_dataset(
        annotated_dir=args.annotated_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        stratify_by=None if args.no_stratify else "domain",
    )
    print(json.dumps(result, indent=2))

    dataset_statistics(args.output_dir)
