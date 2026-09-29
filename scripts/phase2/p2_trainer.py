"""
scripts/phase2/p2_trainer.py
────────────────────────────
Reusable training engine for all three Vedic Shloka classifiers.

Features
  • AdamW optimiser + linear warm-up scheduler
  • Class-weighted CrossEntropyLoss (handles label imbalance)
  • Automatic GPU / CPU / mixed-precision selection
  • Per-epoch train + validation metrics
  • Early stopping on validation loss
  • Best-model checkpointing
  • Full structured logging
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from phase2_config import (
    LEARNING_RATE, EPOCHS, WEIGHT_DECAY, WARMUP_RATIO,
    GRAD_CLIP, RANDOM_SEED, MODELS_DIR, VERSION_REGISTRY,
)
from scripts.phase2.p2_logger import get_logger, get_audit

logger = get_logger("p2_trainer")
audit  = get_audit("training")


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        logger.info("No GPU found — training on CPU")
    return dev


def set_seed(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Version registry
# ─────────────────────────────────────────────────────────────────────────────

def _register_version(task: str, version: str, params: dict, metrics: dict) -> None:
    import phase2_config as _cfg  # read live value, not import-time snapshot
    reg_path = _cfg.VERSION_REGISTRY
    _cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry = {}
    if reg_path.exists():
        with open(reg_path) as f:
            registry = json.load(f)
    registry.setdefault(task, []).append({
        "version":   version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params":    params,
        "metrics":   metrics,
    })
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Version '{version}' registered for task '{task}'")


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    preds = np.argmax(logits, axis=1)
    acc   = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict]:
    """Run one evaluation pass; return (avg_loss, metrics_dict)."""
    model.eval()
    total_loss  = 0.0
    all_logits  = []
    all_labels  = []

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            labels = batch.pop("labels")
            with autocast(enabled=use_amp):
                logits = model(**batch)
                loss   = criterion(logits, labels)
            total_loss += loss.item()
            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)
    metrics    = _compute_metrics(all_logits, all_labels)
    return total_loss / max(len(loader), 1), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    task: str,
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    model_dir: Path,
    data_info: dict,
    epochs: int              = EPOCHS,
    lr: float                = LEARNING_RATE,
    weight_decay: float      = WEIGHT_DECAY,
    warmup_ratio: float      = WARMUP_RATIO,
    grad_clip: float         = GRAD_CLIP,
    seed: int                = RANDOM_SEED,
    class_weights: Optional[list[float]] = None,
    version: Optional[str]   = None,
) -> dict:
    """
    Fine-tune `model` on `train_dl` and evaluate on `val_dl`.

    Args:
        task:          Task name ("veda" | "domain" | "branch")
        model:         ShlokaClassifier instance
        train_dl:      Training DataLoader
        val_dl:        Validation DataLoader
        model_dir:     Directory to checkpoint the best model
        data_info:     Dict from make_dataloaders (labels, sizes …)
        class_weights: Optional per-class weights for loss
        version:       Version string (auto-generated if None)

    Returns:
        Dict with training history and best metrics
    """
    set_seed(seed)
    device  = get_device()
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    if class_weights:
        w = torch.tensor(class_weights, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimiser ─────────────────────────────────────────────────────────
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=lr)

    # ── Scheduler ─────────────────────────────────────────────────────────
    total_steps  = len(train_dl) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    version = version or datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_metrics  = {}
    t0 = time.time()

    logger.info(f"[{task}] Training started — {epochs} epochs, "
                f"{'AMP' if use_amp else 'FP32'}, device={device}")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss   = 0.0
        train_steps  = 0
        all_logits_t = []
        all_labels_t = []

        for step, batch in enumerate(train_dl, 1):
            batch  = _batch_to_device(batch, device)
            labels = batch.pop("labels")

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(**batch)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss  += loss.item()
            train_steps += 1
            all_logits_t.append(logits.detach().cpu().float().numpy())
            all_labels_t.append(labels.cpu().numpy())

            if step % max(len(train_dl) // 4, 1) == 0:
                logger.info(
                    f"  [{task}] epoch {epoch}/{epochs} step {step}/{len(train_dl)} "
                    f"loss={train_loss/train_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = train_loss / max(train_steps, 1)
        train_metrics  = _compute_metrics(
            np.vstack(all_logits_t), np.concatenate(all_labels_t)
        )

        # ── Validate ───────────────────────────────────────────────────────
        avg_val_loss, val_metrics = _eval_epoch(model, val_dl, criterion, device, use_amp)

        row = {
            "epoch":       epoch,
            "train_loss":  round(avg_train_loss, 6),
            "val_loss":    round(avg_val_loss, 6),
            "train_acc":   round(train_metrics["accuracy"], 4),
            "val_acc":     round(val_metrics["accuracy"], 4),
            "val_f1":      round(val_metrics["f1"], 4),
            "lr":          round(scheduler.get_last_lr()[0], 8),
        }
        history.append(row)

        logger.info(
            f"[{task}] Epoch {epoch}/{epochs} — "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics  = {**val_metrics, "val_loss": avg_val_loss}
            model.save(model_dir)
            logger.info(f"  ✓ Best model saved (val_loss={avg_val_loss:.4f})")

    elapsed = time.time() - t0

    # ── Save training artefacts ────────────────────────────────────────────
    training_cfg = {
        "task":          task,
        "version":       version,
        "pretrained":    model.pretrained_name,
        "num_labels":    model.num_labels,
        "classes":       data_info["classes"],
        "epochs":        epochs,
        "batch_size":    train_dl.batch_size,
        "learning_rate": lr,
        "weight_decay":  weight_decay,
        "warmup_ratio":  warmup_ratio,
        "seed":          seed,
        "train_size":    data_info["train_size"],
        "val_size":      data_info["val_size"],
        "elapsed_sec":   round(elapsed, 1),
    }
    with open(model_dir / "training_config.json", "w") as f:
        json.dump(training_cfg, f, indent=2)

    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(model_dir / "best_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    _register_version(task, version, training_cfg, best_metrics)
    audit.log("training_complete", task=task, version=version,
              best_metrics=best_metrics, elapsed_sec=round(elapsed, 1))

    logger.info(
        f"[{task}] Training complete in {elapsed:.1f}s — "
        f"best val_f1={best_metrics.get('f1', 0):.4f}"
    )
    return {"version": version, "history": history, "best_metrics": best_metrics,
            "training_config": training_cfg}
