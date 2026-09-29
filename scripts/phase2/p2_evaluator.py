"""
scripts/phase2/p2_evaluator.py
──────────────────────────────
Evaluation module: loads trained model, runs test-set predictions,
computes full metrics and saves reports + confusion-matrix plots.

Outputs to reports/<task>/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from phase2_config import REPORTS_DIR, TASKS
from scripts.phase2.p2_logger import get_logger, get_audit
from scripts.phase2.p2_dataset import load_label_encoder
from scripts.phase2.p2_trainer import get_device

logger = get_logger("p2_evaluator")
audit  = get_audit("evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    test_dl: DataLoader,
    label_encoder,
    task: str,
    report_dir: Path = None,
) -> dict:
    """
    Run full evaluation on test_dl.

    Args:
        model:         Loaded ShlokaClassifier
        test_dl:       Test DataLoader
        label_encoder: Fitted sklearn LabelEncoder
        task:          Task name (for logging / filenames)
        report_dir:    Directory to save reports (defaults to reports/<task>)

    Returns:
        Dict with accuracy, precision, recall, f1, per-class metrics
    """
    device  = get_device()
    use_amp = device.type == "cuda"
    model.to(device).eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dl:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=use_amp):
                logits = model(**batch)
            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits  = np.vstack(all_logits)
    all_labels  = np.concatenate(all_labels)
    all_probs   = _softmax(all_logits)
    all_preds   = np.argmax(all_logits, axis=1)
    all_conf    = all_probs.max(axis=1)

    class_names = list(label_encoder.classes_)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # ── Per-class metrics ──────────────────────────────────────────────────
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))),
        zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": round(float(per_class_p[i]),  4),
            "recall":    round(float(per_class_r[i]),  4),
            "f1":        round(float(per_class_f1[i]), 4),
            "support":   int(support[i]),
        }
        for i in range(len(class_names))
    }

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    results = {
        "task":               task,
        "test_size":          len(all_labels),
        "accuracy":           round(float(acc), 4),
        "weighted_precision": round(float(p),   4),
        "weighted_recall":    round(float(r),   4),
        "weighted_f1":        round(float(f1),  4),
        "macro_precision":    round(float(p_m), 4),
        "macro_recall":       round(float(r_m), 4),
        "macro_f1":           round(float(f1_m),4),
        "mean_confidence":    round(float(all_conf.mean()), 4),
        "per_class":          per_class,
        "confusion_matrix":   cm.tolist(),
        "class_names":        class_names,
    }

    # ── Save reports ───────────────────────────────────────────────────────
    report_dir = Path(report_dir or (REPORTS_DIR / task))
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # sklearn classification report (text)
    # Pass labels= to guard against missing classes in small test sets
    observed_labels = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    safe_names = [class_names[i] for i in observed_labels if i < len(class_names)]
    txt_report = classification_report(
        all_labels, all_preds,
        labels=observed_labels,
        target_names=safe_names,
        zero_division=0,
    )
    (report_dir / "classification_report.txt").write_text(txt_report)

    # Confusion matrix plot
    _plot_confusion_matrix(cm, class_names, task, report_dir)

    # Training curves (if history available)
    _plot_training_curves(task, report_dir)

    logger.info(
        f"[{task}] Test — acc={acc:.4f}  wF1={f1:.4f}  "
        f"mF1={f1_m:.4f}  mean_conf={all_conf.mean():.3f}"
    )
    audit.log("evaluation_complete", task=task, accuracy=acc, weighted_f1=f1, macro_f1=f1_m)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _plot_confusion_matrix(cm: np.ndarray, class_names: list, task: str, out_dir: Path) -> None:
    n = len(class_names)
    fig_size = max(8, n * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Normalize for colour scale but annotate with raw counts
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="grey",
        ax=ax, cbar_kws={"label": "Normalised frequency"},
    )
    ax.set_title(f"{task.capitalize()} Classifier — Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Confusion matrix → {path}")


def _plot_training_curves(task: str, report_dir: Path) -> None:
    """Plot loss + accuracy curves if training_history.json is available."""
    task_cfg = TASKS.get(task, {})
    model_dir = Path(task_cfg.get("model_dir", ""))
    hist_path = model_dir / "training_history.json"
    if not hist_path.exists():
        return

    with open(hist_path) as f:
        history = json.load(f)

    epochs      = [r["epoch"]      for r in history]
    train_loss  = [r["train_loss"] for r in history]
    val_loss    = [r["val_loss"]   for r in history]
    train_acc   = [r["train_acc"]  for r in history]
    val_acc     = [r["val_acc"]    for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, "o-", label="Train", color="#2196F3")
    ax1.plot(epochs, val_loss,   "s--", label="Val",  color="#F44336")
    ax1.set_title(f"{task.capitalize()} — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, "o-", label="Train", color="#4CAF50")
    ax2.plot(epochs, val_acc,   "s--", label="Val",  color="#FF9800")
    ax2.set_title(f"{task.capitalize()} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = report_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Training curves → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_task(task: str, test_dl: DataLoader) -> dict:
    """
    Load saved model for `task` and evaluate on test_dl.
    Intended for use after training.
    """
    from scripts.phase2.p2_model import ShlokaClassifier
    task_cfg  = TASKS[task]
    model_dir = Path(task_cfg["model_dir"])

    model = ShlokaClassifier.load(model_dir)
    le    = load_label_encoder(model_dir / "label_encoder.pkl")
    return evaluate_model(model, test_dl, le, task)
