"""
p2_run_training.py
──────────────────
Master script: trains all three Vedic Shloka classifiers end-to-end.

Usage
  # Train all three models
  python p2_run_training.py

  # Train a single task
  python p2_run_training.py --tasks veda

  # Override hyperparameters
  python p2_run_training.py --epochs 3 --batch-size 8 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase2_config import (
    TASKS, PRETRAINED_MODEL_NAME,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED, LOGS_DIR,
)
from scripts.phase2.p2_logger import get_logger, get_audit
from scripts.phase2.p2_dataset import load_tokenizer, make_dataloaders
from scripts.phase2.p2_model import ShlokaClassifier
from scripts.phase2.p2_trainer import train_classifier
from scripts.phase2.p2_evaluator import evaluate_model
from scripts.phase2.p2_dataset import load_label_encoder

logger = get_logger("p2_runner")
audit  = get_audit("runner")


def run_all_tasks(
    tasks: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    pretrained: str,
    version: str,
) -> dict:
    """Train + evaluate all specified tasks."""

    t0 = time.time()
    summary: dict = {}

    # ── Tokenizer (shared across all tasks) ────────────────────────────────
    logger.info("Loading tokenizer…")
    tokenizer = load_tokenizer(pretrained)

    for task in tasks:
        logger.info("=" * 64)
        logger.info(f"TASK: {task.upper()}")
        logger.info("=" * 64)

        cfg       = TASKS[task]
        model_dir = Path(cfg["model_dir"])

        # ── Data ──────────────────────────────────────────────────────────
        logger.info(f"[{task}] Building dataloaders…")
        train_dl, val_dl, test_dl, le, data_info = make_dataloaders(
            task, tokenizer, batch_size=batch_size, seed=seed
        )
        logger.info(f"[{task}] {data_info['num_labels']} classes | "
                    f"train={data_info['train_size']} "
                    f"val={data_info['val_size']} "
                    f"test={data_info['test_size']}")

        # ── Model ─────────────────────────────────────────────────────────
        logger.info(f"[{task}] Initialising ShlokaClassifier…")
        model = ShlokaClassifier(
            pretrained_name=pretrained,
            num_labels=data_info["num_labels"],
        )

        # ── Train ─────────────────────────────────────────────────────────
        train_result = train_classifier(
            task=task,
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            model_dir=model_dir,
            data_info=data_info,
            epochs=epochs,
            lr=lr,
            seed=seed,
            class_weights=data_info.get("class_weights"),
            version=version,
        )

        # ── Save tokenizer alongside model ─────────────────────────────────
        tokenizer.save_pretrained(model_dir / "tokenizer")
        logger.info(f"[{task}] Tokenizer saved → {model_dir / 'tokenizer'}")

        # ── Evaluate ──────────────────────────────────────────────────────
        logger.info(f"[{task}] Evaluating on test set…")
        # Reload best checkpoint for evaluation
        best_model = ShlokaClassifier.load(model_dir)
        eval_results = evaluate_model(best_model, test_dl, le, task)

        summary[task] = {
            "best_val_metrics": train_result["best_metrics"],
            "test_metrics":     eval_results,
            "version":          version,
            "model_dir":        str(model_dir),
        }

    elapsed = time.time() - t0
    logger.info(f"All tasks complete in {elapsed:.1f}s")

    # ── Save master summary ────────────────────────────────────────────────
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["version"]     = version
    summary_path = LOGS_DIR / "phase2_summary.json"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Phase 2 summary → {summary_path}")

    audit.log("phase2_complete", version=version, tasks=tasks,
              elapsed_sec=round(elapsed, 1))
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("  PHASE 2 — RESULTS SUMMARY")
    print("=" * 64)
    for task in ("veda", "domain", "branch"):
        if task not in summary:
            continue
        s = summary[task]
        vm = s.get("best_val_metrics", {})
        tm = s.get("test_metrics", {})
        print(f"\n  {task.upper()} CLASSIFIER")
        print(f"    Best val  F1 : {vm.get('f1', 0):.4f}")
        print(f"    Test  acc    : {tm.get('accuracy', 0):.4f}")
        print(f"    Test  wF1   : {tm.get('weighted_f1', 0):.4f}")
        print(f"    Test  mF1   : {tm.get('macro_f1', 0):.4f}")
        print(f"    Mean conf.  : {tm.get('mean_confidence', 0):.3f}")
    print(f"\n  Version   : {summary.get('version', '—')}")
    print(f"  Elapsed   : {summary.get('elapsed_sec', 0):.1f}s")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 — Train Vedic Shloka classifiers"
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()),
                        help="Which tasks to train (default: all)")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--seed",       type=int,   default=RANDOM_SEED)
    parser.add_argument("--model",      default=PRETRAINED_MODEL_NAME,
                        help="HuggingFace model identifier")
    parser.add_argument("--version",    default=None,
                        help="Version tag (auto-generated if omitted)")
    args = parser.parse_args()

    version = args.version or datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

    summary = run_all_tasks(
        tasks=args.tasks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        pretrained=args.model,
        version=version,
    )
    print_summary(summary)
