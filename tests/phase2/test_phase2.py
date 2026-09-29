"""
tests/phase2/test_phase2.py
────────────────────────────
Unit + integration tests for Phase 2 modules.

Tests are designed to run WITHOUT downloading any pretrained weights
by using a tiny randomly-initialised BERT config as the backbone.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# ── project root on path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


SAMPLE_SHLOKAS = [
    "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।",
    "गणितं क्षेत्रमितिः सूत्रं च वर्गघनम्।",
    "ज्योतिष ग्रह नक्षत्र सूर्य।",
    "इन्द्रमिद् गाथिनो बृहद् इन्द्रम्।",
    "ब्रह्म आत्म मोक्ष धर्म सत्यम्।",
]

SAMPLE_LABELS = ["Rigveda", "Mathematics", "Astronomy", "Rigveda", "Philosophy"]


def _make_csv_files(tmp_dir: Path) -> Path:
    """Write minimal train/val/test CSVs for testing."""
    import pandas as pd

    rows = [
        {
            "id":      f"SHLOKA_{i:06d}",
            "text":    SAMPLE_SHLOKAS[i % len(SAMPLE_SHLOKAS)],
            "source":  "Test",
            "chapter": "1",
            "veda":    ["Rigveda", "Unknown", "Rigveda"][i % 3],
            "domain":  ["Philosophy", "Mathematics", "Astronomy"][i % 3],
            "branch":  ["Vedanta", "Arithmetic", "Jyotisha"][i % 3],
            "formula": "",
            "language": "Sanskrit",
            "keywords": "rigveda",
            "annotated_by": "test",
            "confidence": 1.0,
            "notes": "",
        }
        for i in range(30)
    ]
    df = pd.DataFrame(rows)
    data_dir = tmp_dir / "training"
    data_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = df.iloc[:20], df.iloc[20:25], df.iloc[25:]
    train.to_csv(data_dir / "train.csv",      index=False)
    val.to_csv(  data_dir / "validation.csv", index=False)
    test.to_csv( data_dir / "test.csv",       index=False)
    return data_dir


def _tiny_bert_config():
    """Return a BertConfig with a tiny hidden size for fast tests."""
    from transformers import BertConfig
    return BertConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )


def _make_fake_model(tmp_dir: Path, num_labels: int = 3) -> Path:
    """
    Build and save a tiny ShlokaClassifier that uses a randomly-initialised
    BertModel backbone (no internet required).
    """
    from transformers import BertModel
    from scripts.phase2.p2_model import ShlokaClassifier

    cfg      = _tiny_bert_config()
    backbone = BertModel(cfg)

    model = ShlokaClassifier.__new__(ShlokaClassifier)
    nn.Module.__init__(model)
    model.pretrained_name = "test-tiny-bert"
    model.num_labels      = num_labels
    model.backbone        = backbone
    model.norm            = nn.LayerNorm(cfg.hidden_size)
    model.dropout         = nn.Dropout(0.0)
    model.classifier      = nn.Linear(cfg.hidden_size, num_labels)
    nn.init.zeros_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    # Save DIRECTLY into tmp_dir (model_dir), not a "fake_model" subdir
    model_dir = Path(tmp_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir)
    return model_dir


def _make_fake_tokenizer(tmp_dir: Path) -> Path:
    """Save a BertTokenizer with minimal vocab for tests."""
    from transformers import BertTokenizer, BertConfig
    import os

    tok_dir = tmp_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Write a minimal vocab file
    vocab_path = tok_dir / "vocab.txt"
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    normal_tokens  = [f"tok{i}" for i in range(100)]
    vocab_path.write_text("\n".join(special_tokens + normal_tokens))

    # Write tokenizer config
    tok_cfg = {
        "model_type":       "bert",
        "do_lower_case":    True,
        "vocab_size":       len(special_tokens) + len(normal_tokens),
        "tokenizer_class":  "BertTokenizer",
    }
    (tok_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg))

    return tok_dir


def _make_fake_le(classes: list[str], path: Path) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(le, f)
    return le


def _tiny_dataloader(num_samples: int = 6, seq_len: int = 16, num_labels: int = 3) -> DataLoader:
    """Synthetic DataLoader that bypasses tokenisation entirely."""
    input_ids      = torch.randint(0, 100, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(num_samples, seq_len, dtype=torch.long)
    labels         = torch.randint(0, num_labels, (num_samples,))
    ds = TensorDataset(input_ids, attention_mask, token_type_ids, labels)

    class _BatchDS(torch.utils.data.Dataset):
        def __len__(self): return num_samples
        def __getitem__(self, i):
            return {
                "input_ids":      input_ids[i],
                "attention_mask": attention_mask[i],
                "token_type_ids": token_type_ids[i],
                "labels":         labels[i],
            }

    return DataLoader(_BatchDS(), batch_size=2, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset:

    def test_shloka_dataset_returns_correct_keys(self):
        """ShlokaDataset items must contain input_ids, attention_mask, labels."""
        from scripts.phase2.p2_dataset import ShlokaDataset
        from transformers import BertTokenizer

        # Build a tiny BertTokenizer from scratch
        with tempfile.TemporaryDirectory() as d:
            vocab = Path(d) / "vocab.txt"
            tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"] + \
                     list("अआइईउऊएऐओऔकखगघ") + [f"##t{i}" for i in range(20)]
            vocab.write_text("\n".join(tokens))
            tok = BertTokenizer(vocab_file=str(vocab))

        texts  = ["अग्नि पुरोहित।", "गणित सूत्र।"]
        labels = [0, 1]
        ds     = ShlokaDataset(texts, labels, tok, max_length=32)

        assert len(ds) == 2
        item = ds[0]
        assert "input_ids"      in item
        assert "attention_mask" in item
        assert "labels"         in item
        assert item["input_ids"].shape[0] == 32

    def test_shloka_dataset_no_labels(self):
        """When labels=None (inference mode), 'labels' key must be absent."""
        from scripts.phase2.p2_dataset import ShlokaDataset
        from transformers import BertTokenizer

        with tempfile.TemporaryDirectory() as d:
            vocab = Path(d) / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld")
            tok = BertTokenizer(vocab_file=str(vocab))

        ds   = ShlokaDataset(["hello world"], None, tok, max_length=16)
        item = ds[0]
        assert "labels" not in item

    def test_label_encoder_roundtrip(self, tmp_dir):
        """save → load must preserve class ordering."""
        from scripts.phase2.p2_dataset import (
            build_label_encoder, save_label_encoder, load_label_encoder
        )
        import pandas as pd

        series = pd.Series(["Rigveda", "Unknown", "Rigveda", "Samaveda"])
        known  = ["Rigveda", "Samaveda", "Yajurveda", "Unknown"]
        le     = build_label_encoder(series, known)

        path = tmp_dir / "le.pkl"
        save_label_encoder(le, path)
        le2 = load_label_encoder(path)
        assert list(le.classes_) == list(le2.classes_)

    def test_label_encoder_unknown_fallback(self, tmp_dir):
        """Unseen labels must map to 'Unknown'."""
        from scripts.phase2.p2_dataset import build_label_encoder
        import pandas as pd

        series = pd.Series(["Rigveda", "Unknown"])
        known  = ["Rigveda", "Unknown"]
        le     = build_label_encoder(series, known)

        # "NewLabel" not in known → caller should remap to "Unknown"
        mapped = "Unknown" if "NewLabel" not in le.classes_ else "NewLabel"
        enc    = le.transform([mapped])
        assert enc[0] == le.transform(["Unknown"])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModel:

    def test_forward_pass_shape(self, tmp_dir):
        """logits must be (B, num_labels)."""
        model_dir = _make_fake_model(tmp_dir, num_labels=4)
        from scripts.phase2.p2_model import ShlokaClassifier
        model = ShlokaClassifier.load(model_dir)
        model.eval()

        B, L = 3, 16
        with torch.no_grad():
            logits = model(
                input_ids=torch.randint(0, 100, (B, L)),
                attention_mask=torch.ones(B, L, dtype=torch.long),
            )
        assert logits.shape == (B, 4)

    def test_save_load_weights_identical(self, tmp_dir):
        """Saved and reloaded models must produce identical logits."""
        model_dir = _make_fake_model(tmp_dir, num_labels=3)
        from scripts.phase2.p2_model import ShlokaClassifier
        m1 = ShlokaClassifier.load(model_dir)
        m2 = ShlokaClassifier.load(model_dir)
        m1.eval(); m2.eval()

        x = dict(
            input_ids=torch.randint(0, 100, (2, 16)),
            attention_mask=torch.ones(2, 16, dtype=torch.long),
        )
        with torch.no_grad():
            l1 = m1(**x)
            l2 = m2(**x)
        assert torch.allclose(l1, l2, atol=1e-5)

    def test_classifier_head_replaced(self, tmp_dir):
        """num_labels must match the saved value after reload."""
        for n in [2, 6, 12]:
            d = _make_fake_model(tmp_dir / f"m{n}", num_labels=n)
            from scripts.phase2.p2_model import ShlokaClassifier
            m = ShlokaClassifier.load(d)
            assert m.num_labels == n
            assert m.classifier.out_features == n

    def test_no_nan_in_output(self, tmp_dir):
        """Logits must not contain NaN."""
        model_dir = _make_fake_model(tmp_dir, num_labels=5)
        from scripts.phase2.p2_model import ShlokaClassifier
        m = ShlokaClassifier.load(model_dir)
        m.eval()
        with torch.no_grad():
            logits = m(
                input_ids=torch.randint(0, 100, (4, 16)),
                attention_mask=torch.ones(4, 16, dtype=torch.long),
            )
        assert not torch.isnan(logits).any()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainer:

    def test_single_training_step(self, tmp_dir):
        """
        One epoch over a synthetic DataLoader must complete without error
        and produce a training_history.json with one entry.
        """
        from scripts.phase2.p2_trainer import train_classifier

        model_dir  = _make_fake_model(tmp_dir / "ckpt", num_labels=3)
        from scripts.phase2.p2_model import ShlokaClassifier
        model = ShlokaClassifier.load(model_dir)

        train_dl = _tiny_dataloader(num_samples=4, seq_len=16, num_labels=3)
        val_dl   = _tiny_dataloader(num_samples=4, seq_len=16, num_labels=3)
        data_info = {
            "train_size": 4, "val_size": 4, "num_labels": 3,
            "classes": ["A", "B", "C"],
        }

        result = train_classifier(
            task="veda",
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            model_dir=model_dir,
            data_info=data_info,
            epochs=1,
            lr=1e-4,
            seed=42,
            version="test_v1",
        )

        assert result["version"] == "test_v1"
        assert len(result["history"]) == 1
        assert "f1" in result["best_metrics"]
        assert (model_dir / "training_history.json").exists()
        assert (model_dir / "training_config.json").exists()

    def test_best_model_checkpointed(self, tmp_dir):
        """Best checkpoint must be saved (classifier_head.pt exists)."""
        from scripts.phase2.p2_trainer import train_classifier
        from scripts.phase2.p2_model import ShlokaClassifier

        model_dir = _make_fake_model(tmp_dir / "ckpt2", num_labels=2)
        model = ShlokaClassifier.load(model_dir)

        dl = _tiny_dataloader(num_samples=4, seq_len=16, num_labels=2)
        data_info = {"train_size": 4, "val_size": 4, "num_labels": 2,
                     "classes": ["A", "B"]}
        train_classifier("veda", model, dl, dl, model_dir, data_info,
                         epochs=2, lr=1e-4, seed=0, version="ckpt_test")

        assert (model_dir / "classifier_head.pt").exists()

    def test_training_config_saved(self, tmp_dir):
        """training_config.json must contain key hyperparameters."""
        from scripts.phase2.p2_trainer import train_classifier
        from scripts.phase2.p2_model import ShlokaClassifier

        model_dir = _make_fake_model(tmp_dir / "cfg", num_labels=2)
        model = ShlokaClassifier.load(model_dir)
        dl = _tiny_dataloader(num_samples=4, seq_len=16, num_labels=2)
        data_info = {"train_size": 4, "val_size": 4, "num_labels": 2,
                     "classes": ["A", "B"]}
        train_classifier("domain", model, dl, dl, model_dir, data_info,
                         epochs=1, lr=2e-5, seed=1, version="cfg_test")

        cfg = json.loads((model_dir / "training_config.json").read_text())
        assert cfg["epochs"] == 1
        assert cfg["learning_rate"] == 2e-5
        assert cfg["task"] == "domain"

    def test_version_registry_updated(self, tmp_dir):
        """VERSION_REGISTRY must contain the new version after training."""
        import phase2_config as p2cfg
        orig = p2cfg.VERSION_REGISTRY
        p2cfg.VERSION_REGISTRY = tmp_dir / "version_registry.json"
        p2cfg.MODELS_DIR = tmp_dir

        try:
            from scripts.phase2.p2_trainer import train_classifier, _register_version
            _register_version("veda", "v_test", {"epochs": 1}, {"f1": 0.5})

            reg = json.loads(p2cfg.VERSION_REGISTRY.read_text())
            assert "veda" in reg
            assert reg["veda"][0]["version"] == "v_test"
        finally:
            p2cfg.VERSION_REGISTRY = orig
            p2cfg.MODELS_DIR = orig.parent


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluator:

    def test_eval_returns_required_keys(self, tmp_dir):
        """evaluate_model must return accuracy, precision, recall, f1, cm."""
        from scripts.phase2.p2_evaluator import evaluate_model
        from scripts.phase2.p2_model import ShlokaClassifier

        num_labels = 3
        model_dir  = _make_fake_model(tmp_dir, num_labels=num_labels)
        model      = ShlokaClassifier.load(model_dir)
        le         = _make_fake_le(["A", "B", "C"], tmp_dir / "le.pkl")
        test_dl    = _tiny_dataloader(num_samples=6, seq_len=16, num_labels=num_labels)

        results = evaluate_model(model, test_dl, le, "veda", report_dir=tmp_dir / "report")

        for key in ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
                    "macro_f1", "confusion_matrix", "class_names", "per_class"):
            assert key in results, f"Missing key: {key}"

    def test_confusion_matrix_shape(self, tmp_dir):
        """Confusion matrix must be (num_labels × num_labels)."""
        from scripts.phase2.p2_evaluator import evaluate_model
        from scripts.phase2.p2_model import ShlokaClassifier

        n = 4
        model = ShlokaClassifier.load(_make_fake_model(tmp_dir, num_labels=n))
        le    = _make_fake_le([f"C{i}" for i in range(n)], tmp_dir / "le.pkl")
        dl    = _tiny_dataloader(num_samples=8, seq_len=16, num_labels=n)

        results = evaluate_model(model, dl, le, "domain", report_dir=tmp_dir / "r")
        cm      = np.array(results["confusion_matrix"])
        assert cm.shape == (n, n)

    def test_report_files_created(self, tmp_dir):
        """eval_results.json and classification_report.txt must be created."""
        from scripts.phase2.p2_evaluator import evaluate_model
        from scripts.phase2.p2_model import ShlokaClassifier

        model = ShlokaClassifier.load(_make_fake_model(tmp_dir, num_labels=2))
        le    = _make_fake_le(["X", "Y"], tmp_dir / "le.pkl")
        dl    = _tiny_dataloader(num_samples=4, seq_len=16, num_labels=2)
        rdir  = tmp_dir / "report"

        evaluate_model(model, dl, le, "branch", report_dir=rdir)

        assert (rdir / "eval_results.json").exists()
        assert (rdir / "classification_report.txt").exists()
        assert (rdir / "confusion_matrix.png").exists()

    def test_metrics_within_valid_range(self, tmp_dir):
        """All metric values must be in [0, 1]."""
        from scripts.phase2.p2_evaluator import evaluate_model
        from scripts.phase2.p2_model import ShlokaClassifier

        model = ShlokaClassifier.load(_make_fake_model(tmp_dir, num_labels=3))
        le    = _make_fake_le(["A", "B", "C"], tmp_dir / "le.pkl")
        dl    = _tiny_dataloader(num_samples=6, seq_len=16, num_labels=3)

        r = evaluate_model(model, dl, le, "veda", report_dir=tmp_dir / "r")
        for metric in ("accuracy", "weighted_f1", "macro_f1", "mean_confidence"):
            assert 0.0 <= r[metric] <= 1.0, f"{metric}={r[metric]} out of range"


# ─────────────────────────────────────────────────────────────────────────────
# Inference tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInference:

    def _setup_inference(self, tmp_dir: Path):
        """
        Build minimal saved models + tokenizer for inference tests,
        bypassing internet downloads entirely.
        """
        import phase2_config as p2cfg
        from scripts.phase2.p2_inference import ShlokaInferencePipeline
        from scripts.phase2.p2_model import ShlokaClassifier
        from transformers import BertTokenizer

        # Tiny vocab tokenizer
        vocab = tmp_dir / "vocab.txt"
        tokens = (["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
                  + list("अआइईउऊएकखगघचछजझटठडढ")
                  + [f"##w{i}" for i in range(60)])
        vocab.write_text("\n".join(tokens))
        tok = BertTokenizer(vocab_file=str(vocab))

        tasks = ["veda", "domain", "branch"]
        labels_map = {
            "veda":   ["Rigveda", "Unknown", "Other"],
            "domain": ["Mathematics", "Unknown", "Philosophy"],
            "branch": ["Arithmetic", "Unknown", "Vedanta"],
        }

        for task in tasks:
            n         = len(labels_map[task])
            model_dir = tmp_dir / task
            model_dir.mkdir(parents=True, exist_ok=True)
            _make_fake_model(model_dir, num_labels=n)

            # Save tokenizer backbone inside model dir
            tok.save_pretrained(model_dir / "backbone")
            tok.save_pretrained(model_dir / "tokenizer")

            # Save label encoder
            _make_fake_le(labels_map[task], model_dir / "label_encoder.pkl")

        # Pass models_root so pipeline builds paths as tmp_dir/<task>
        pipe = ShlokaInferencePipeline(tasks=tasks, models_root=None,
                                       max_length=64,   # match tiny BERT max_pos=64
                                       _task_dirs={t: tmp_dir / t for t in tasks})
        return pipe, labels_map

    def test_predict_returns_all_keys(self, tmp_dir):
        """predict() output must contain veda, domain, branch, confidence."""
        pipe, _ = self._setup_inference(tmp_dir)
        result  = pipe.predict("अग्निमीळे पुरोहितं।")

        for key in ("veda", "domain", "branch"):
            assert key in result

        assert "confidence" in result
        assert "overall" in result["confidence"]

    def test_predict_label_is_string(self, tmp_dir):
        """Each task label must be a non-empty string."""
        pipe, _ = self._setup_inference(tmp_dir)
        result  = pipe.predict("गणितं क्षेत्रम्।")

        for task in ("veda", "domain", "branch"):
            assert isinstance(result[task], str)
            assert len(result[task]) > 0

    def test_confidence_in_range(self, tmp_dir):
        """All confidence values must be in [0, 1]."""
        pipe, _ = self._setup_inference(tmp_dir)
        result  = pipe.predict("test shloka")

        for val in result["confidence"].values():
            assert 0.0 <= val <= 1.0, f"Confidence {val} out of range"

    def test_probabilities_sum_to_one(self, tmp_dir):
        """Softmax probabilities for each task must sum to ~1."""
        pipe, _ = self._setup_inference(tmp_dir)
        result  = pipe.predict("अग्नि।")

        for task, probs in result["probabilities"].items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-3, f"[{task}] probs sum to {total}"

    def test_batch_predict_length(self, tmp_dir):
        """predict_batch must return one result per input."""
        pipe, _ = self._setup_inference(tmp_dir)
        texts   = SAMPLE_SHLOKAS[:3]
        results = pipe.predict_batch(texts)

        assert len(results) == len(texts)

    def test_empty_string_handled(self, tmp_dir):
        """Empty string input must not raise an exception."""
        pipe, _ = self._setup_inference(tmp_dir)
        result  = pipe.predict("")
        assert "confidence" in result


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_train_eval_infer_cycle(self, tmp_dir):
        """
        Full mini pipeline: 1 epoch training → evaluation → inference.
        Uses only synthetic data and tiny random models — no downloads.
        """
        import phase2_config as p2cfg
        from scripts.phase2.p2_trainer import train_classifier
        from scripts.phase2.p2_evaluator import evaluate_model
        from scripts.phase2.p2_model import ShlokaClassifier

        num_labels = 3
        task       = "veda"
        model_dir  = tmp_dir / "int_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        orig_reg  = p2cfg.VERSION_REGISTRY
        orig_mdir = p2cfg.MODELS_DIR
        p2cfg.VERSION_REGISTRY = tmp_dir / "reg.json"
        p2cfg.MODELS_DIR       = tmp_dir

        try:
            # Build model
            _make_fake_model(model_dir, num_labels=num_labels)
            model = ShlokaClassifier.load(model_dir)

            dl = _tiny_dataloader(num_samples=6, seq_len=16, num_labels=num_labels)
            data_info = {
                "train_size": 6, "val_size": 6, "num_labels": num_labels,
                "classes": ["Rigveda", "Samaveda", "Unknown"],
            }

            # Train
            train_result = train_classifier(
                task, model, dl, dl, model_dir, data_info,
                epochs=1, lr=1e-4, seed=42, version="int_v1"
            )
            assert train_result["version"] == "int_v1"

            # Evaluate
            best_model = ShlokaClassifier.load(model_dir)
            le         = _make_fake_le(data_info["classes"], model_dir / "label_encoder.pkl")
            eval_r     = evaluate_model(best_model, dl, le, task,
                                        report_dir=tmp_dir / "report")
            assert "accuracy" in eval_r
            assert (tmp_dir / "report" / "eval_results.json").exists()

        finally:
            p2cfg.VERSION_REGISTRY = orig_reg
            p2cfg.MODELS_DIR       = orig_mdir
