"""
scripts/phase2/p2_inference.py
──────────────────────────────
Inference pipeline — accepts raw Sanskrit shloka text, runs all three
classifiers, and returns a structured prediction with confidence scores.

Example
-------
>>> from scripts.phase2.p2_inference import ShlokaInferencePipeline
>>> pipe = ShlokaInferencePipeline()
>>> pipe.predict("अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।")
{
  "veda":       "Rigveda",
  "domain":     "Philosophy",
  "branch":     "Vedanta",
  "confidence": {"veda": 0.91, "domain": 0.78, "branch": 0.65, "overall": 0.78},
  "probabilities": {
      "veda":   {"Rigveda": 0.91, "Unknown": 0.05, ...},
      "domain": {...},
      "branch": {...}
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from phase2_config import MAX_SEQ_LENGTH, TRUNCATION, TASKS
from scripts.phase2.p2_logger import get_logger
from scripts.phase2.p2_dataset import load_label_encoder
from scripts.phase2.p2_model import ShlokaClassifier
from scripts.phase2.p2_trainer import get_device

logger = get_logger("p2_inference")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _tokenize_single(text: str, tokenizer, max_length: int, device: torch.device) -> dict:
    enc = tokenizer(
        str(text),
        max_length=max_length,
        padding="max_length",
        truncation=TRUNCATION,
        return_tensors="pt",
    )
    out = {
        "input_ids":      enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }
    if "token_type_ids" in enc:
        out["token_type_ids"] = enc["token_type_ids"].to(device)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class ShlokaInferencePipeline:
    """
    Loads all three trained classifiers and exposes a unified predict() method.

    Args:
        tasks:        List of tasks to load (default: all three)
        models_root:  Root directory where model folders live
        max_length:   Tokenizer max sequence length
    """

    def __init__(
        self,
        tasks: Optional[list[str]] = None,
        models_root: Optional[Path] = None,
        max_length: int = MAX_SEQ_LENGTH,
        _task_dirs: Optional[dict] = None,   # test injection: {task: Path}
    ):
        self.tasks      = tasks or list(TASKS.keys())
        self.max_length = max_length
        self.device     = get_device()
        self.use_amp    = self.device.type == "cuda"
        self._task_dirs = _task_dirs  # optional override

        self._models    : dict[str, ShlokaClassifier] = {}
        self._encoders  : dict[str, object]            = {}
        self._tokenizer : Optional[AutoTokenizer]      = None

        self._load_all(models_root)

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_all(self, models_root: Optional[Path]) -> None:
        for task in self.tasks:
            cfg = TASKS[task]
            if self._task_dirs and task in self._task_dirs:
                model_dir = Path(self._task_dirs[task])
            else:
                model_dir = Path(models_root or cfg["model_dir"])

            # Skip gracefully if model not yet trained
            ckpt_path = model_dir / "classifier_head.pt"
            if not ckpt_path.exists():
                logger.warning(f"[{task}] No trained model found at {model_dir} — skipping")
                continue

            try:
                model = ShlokaClassifier.load(model_dir)
                model.to(self.device).eval()
                self._models[task] = model

                le = load_label_encoder(model_dir / "label_encoder.pkl")
                self._encoders[task] = le

                # Load tokenizer from first successfully loaded model
                if self._tokenizer is None:
                    for tok_path in [
                        model_dir / "tokenizer",
                        model_dir / "backbone",
                    ]:
                        if tok_path.exists():
                            try:
                                self._tokenizer = AutoTokenizer.from_pretrained(
                                    str(tok_path), local_files_only=True
                                )
                                logger.info(f"Tokenizer loaded from {tok_path}")
                                break
                            except Exception as e:
                                logger.warning(f"Could not load from {tok_path}: {e}")
                    if self._tokenizer is None:
                        from scripts.phase2.p2_dataset import load_tokenizer
                        self._tokenizer = load_tokenizer()

                logger.info(f"[{task}] Model + encoder loaded")
            except Exception as exc:
                logger.error(f"[{task}] Failed to load model: {exc}")

        if not self._models:
            raise RuntimeError(
                "No trained models found. Run p2_run_training.py first."
            )

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Predict veda, domain, and branch for a single shloka.

        Args:
            text: Sanskrit shloka string

        Returns:
            Structured prediction dict
        """
        inputs  = _tokenize_single(text, self._tokenizer, self.max_length, self.device)
        results = {}

        with torch.no_grad():
            for task, model in self._models.items():
                le = self._encoders[task]
                with autocast(enabled=self.use_amp):
                    logits = model(**inputs).squeeze(0).cpu().float().numpy()

                probs = _softmax(logits)
                pred_idx  = int(np.argmax(probs))
                pred_label = le.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])

                class_probs = {
                    str(le.inverse_transform([i])[0]): round(float(probs[i]), 4)
                    for i in range(len(probs))
                }
                results[task] = {
                    "label":        pred_label,
                    "confidence":   round(confidence, 4),
                    "probabilities": class_probs,
                }

        output = self._format_output(results)
        logger.debug(f"Prediction: {json.dumps(output)}")
        return output

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run predict() over a list of shlokas."""
        return [self.predict(t) for t in texts]

    # ── Output formatting ──────────────────────────────────────────────────

    @staticmethod
    def _format_output(results: dict) -> dict:
        """Flatten nested results into the canonical output schema."""
        out: dict = {}
        confidences: dict[str, float] = {}

        for task, r in results.items():
            out[task]               = r["label"]
            confidences[task]       = r["confidence"]

        # Overall confidence = geometric mean of individual confidences
        if confidences:
            geo_mean = float(np.exp(np.mean(np.log(
                [max(v, 1e-9) for v in confidences.values()]
            ))))
            confidences["overall"] = round(geo_mean, 4)

        out["confidence"]    = confidences
        out["probabilities"] = {t: results[t]["probabilities"] for t in results}
        return out

    # ── Available tasks ────────────────────────────────────────────────────

    @property
    def loaded_tasks(self) -> list[str]:
        return list(self._models.keys())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Vedic Shloka inference")
    parser.add_argument("text", nargs="?",
                        default="अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।",
                        help="Sanskrit shloka text to classify")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to run (default: all)")
    args = parser.parse_args()

    pipe   = ShlokaInferencePipeline(tasks=args.tasks)
    result = pipe.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
