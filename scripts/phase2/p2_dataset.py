"""
scripts/phase2/p2_dataset.py
────────────────────────────
Dataset loading, label encoding, and PyTorch Dataset / DataLoader construction
for the Vedic Shloka transfer-learning pipeline.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from phase2_config import (
    DATA_DIR, MAX_SEQ_LENGTH, PADDING, TRUNCATION,
    BATCH_SIZE, RANDOM_SEED, TASKS,
    PRETRAINED_MODEL_NAME, FALLBACK_MODEL_NAME,
)
from scripts.phase2.p2_logger import get_logger

logger = get_logger("p2_dataset")


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser helper
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str = PRETRAINED_MODEL_NAME) -> AutoTokenizer:
    """Load tokenizer, falling back to multilingual BERT if needed."""
    for name in (model_name, FALLBACK_MODEL_NAME):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            logger.info(f"Loaded tokenizer: {name}")
            return tok
        except Exception as exc:
            logger.warning(f"Could not load tokenizer '{name}': {exc}")
    raise RuntimeError("No tokenizer could be loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Label encoder helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_label_encoder(series: pd.Series, known_labels: list[str]) -> LabelEncoder:
    """
    Fit a LabelEncoder on the union of known_labels + observed values.
    This ensures unseen labels in test data don't cause KeyErrors.
    """
    all_values = list(known_labels) + series.dropna().astype(str).tolist()
    le = LabelEncoder()
    le.fit(all_values)
    return le


def save_label_encoder(le: LabelEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(le, f)
    logger.info(f"Label encoder saved → {path}")


def load_label_encoder(path: Path) -> LabelEncoder:
    with open(path, "rb") as f:
        le = pickle.load(f)
    logger.info(f"Label encoder loaded ← {path}")
    return le


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ShlokaDataset(Dataset):
    """
    Maps a DataFrame of (text, label_id) pairs to PyTorch tensors.

    Args:
        texts:       List of raw Sanskrit shloka strings
        labels:      Integer-encoded label array (or None for inference)
        tokenizer:   HuggingFace tokenizer
        max_length:  Maximum token sequence length
    """

    def __init__(
        self,
        texts: list[str],
        labels: Optional[list[int]],
        tokenizer: AutoTokenizer,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=TRUNCATION,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    task: str,
    tokenizer: AutoTokenizer,
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder, dict]:
    """
    Build train / validation / test DataLoaders for a given task.

    Args:
        task:       One of "veda", "domain", "branch"
        tokenizer:  Loaded HuggingFace tokenizer
        data_dir:   Directory containing train/validation/test.csv
        batch_size: Mini-batch size
        seed:       RNG seed for reproducibility

    Returns:
        (train_dl, val_dl, test_dl, label_encoder, data_info_dict)
    """
    cfg = TASKS[task]
    col = cfg["column"]
    known_labels = cfg["labels"]
    model_dir = Path(cfg["model_dir"])

    # ── Load CSVs ──────────────────────────────────────────────────────────
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "validation.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    # Fill NaN in text and label columns
    for df in (train_df, val_df, test_df):
        df["text"] = df["text"].fillna("").astype(str)
        df[col]    = df[col].fillna("Unknown").astype(str)

    # ── Label encoding ─────────────────────────────────────────────────────
    le = build_label_encoder(train_df[col], known_labels)
    save_label_encoder(le, model_dir / "label_encoder.pkl")

    def encode(series: pd.Series) -> list[int]:
        # Map unseen labels → "Unknown" (always in known_labels)
        return le.transform(
            series.apply(lambda v: v if v in le.classes_ else "Unknown")
        ).tolist()

    y_train = encode(train_df[col])
    y_val   = encode(val_df[col])
    y_test  = encode(test_df[col])

    num_labels = len(le.classes_)
    logger.info(f"[{task}] {num_labels} classes: {list(le.classes_)}")
    logger.info(f"[{task}] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ── Class weights for imbalanced data ─────────────────────────────────
    counts = np.bincount(y_train, minlength=num_labels).astype(float)
    class_weights = (counts.sum() / (num_labels * np.maximum(counts, 1))).tolist()

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds = ShlokaDataset(train_df["text"].tolist(), y_train, tokenizer)
    val_ds   = ShlokaDataset(val_df["text"].tolist(),   y_val,   tokenizer)
    test_ds  = ShlokaDataset(test_df["text"].tolist(),  y_test,  tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, generator=g)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    data_info = {
        "task":          task,
        "target_column": col,
        "num_labels":    num_labels,
        "classes":       list(le.classes_),
        "class_weights": class_weights,
        "train_size":    len(train_ds),
        "val_size":      len(val_ds),
        "test_size":     len(test_ds),
    }
    return train_dl, val_dl, test_dl, le, data_info
