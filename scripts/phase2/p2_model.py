"""
scripts/phase2/p2_model.py
──────────────────────────
Transformer-based classification head for Vedic Shloka tasks.

Architecture
  ┌─────────────────────────────────────────────┐
  │  Pretrained backbone (IndicBERT / mBERT)    │
  │  → [CLS] pooled output (hidden_size)        │
  │  → LayerNorm → Dropout                      │
  │  → Linear (hidden_size → num_labels)        │
  └─────────────────────────────────────────────┘

The backbone weights are fine-tuned end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from phase2_config import DROPOUT, PRETRAINED_MODEL_NAME, FALLBACK_MODEL_NAME
from scripts.phase2.p2_logger import get_logger

logger = get_logger("p2_model")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ShlokaClassifier(nn.Module):
    """
    Fine-tuned transformer classifier for single-label text classification.

    Args:
        pretrained_name: HuggingFace model identifier
        num_labels:      Number of output classes
        dropout:         Dropout probability on the CLS representation
    """

    def __init__(
        self,
        pretrained_name: str = PRETRAINED_MODEL_NAME,
        num_labels: int = 2,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.num_labels      = num_labels

        # ── Load backbone ──────────────────────────────────────────────────
        config = self._load_config(pretrained_name)
        self.backbone = self._load_backbone(pretrained_name, config)
        hidden_size = config.hidden_size

        # ── Classification head ────────────────────────────────────────────
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Weight init for the new head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_config(name: str):
        for n in (name, FALLBACK_MODEL_NAME):
            try:
                cfg = AutoConfig.from_pretrained(n)
                logger.info(f"Loaded config: {n}")
                return cfg
            except Exception as exc:
                logger.warning(f"Config load failed for {n}: {exc}")
        raise RuntimeError("Cannot load model config.")

    @staticmethod
    def _load_backbone(name: str, config) -> AutoModel:
        for n in (name, FALLBACK_MODEL_NAME):
            try:
                m = AutoModel.from_pretrained(n, config=config)
                logger.info(f"Loaded backbone: {n}")
                return m
            except Exception as exc:
                logger.warning(f"Backbone load failed for {n}: {exc}")
        raise RuntimeError("Cannot load backbone model.")

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L) token IDs
            attention_mask: (B, L) mask
            token_type_ids: (B, L) optional segment IDs

        Returns:
            logits: (B, num_labels)
        """
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)

        # Use [CLS] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]   # (B, H)
        cls_repr = self.norm(cls_repr)
        cls_repr = self.dropout(cls_repr)
        logits   = self.classifier(cls_repr)            # (B, num_labels)
        return logits

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, model_dir: Path) -> None:
        """Save backbone + head to model_dir."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.backbone.save_pretrained(model_dir / "backbone")
        torch.save(
            {
                "head_state_dict":  {
                    "norm":       self.norm.state_dict(),
                    "dropout":    self.dropout.state_dict(),
                    "classifier": self.classifier.state_dict(),
                },
                "num_labels":       self.num_labels,
                "pretrained_name":  self.pretrained_name,
            },
            model_dir / "classifier_head.pt",
        )
        logger.info(f"Model saved → {model_dir}")

    @classmethod
    def load(cls, model_dir: Path, dropout: float = DROPOUT) -> "ShlokaClassifier":
        """Reconstruct model from saved files."""
        model_dir = Path(model_dir)
        ckpt = torch.load(model_dir / "classifier_head.pt", map_location="cpu",
                          weights_only=False)
        num_labels      = ckpt["num_labels"]
        pretrained_name = ckpt["pretrained_name"]

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.pretrained_name = pretrained_name
        model.num_labels      = num_labels

        # Reload backbone from saved weights
        model.backbone = AutoModel.from_pretrained(model_dir / "backbone")

        hidden_size = model.backbone.config.hidden_size
        model.norm       = nn.LayerNorm(hidden_size)
        model.dropout    = nn.Dropout(dropout)
        model.classifier = nn.Linear(hidden_size, num_labels)

        head = ckpt["head_state_dict"]
        model.norm.load_state_dict(head["norm"])
        model.dropout.load_state_dict(head["dropout"])
        model.classifier.load_state_dict(head["classifier"])

        logger.info(f"Model loaded ← {model_dir}")
        return model
