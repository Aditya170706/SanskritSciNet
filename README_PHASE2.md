# Phase 2 — Model Development and Transfer Learning Pipeline

## Overview

Fine-tunes three independent transformer classifiers (IndicBERT / mBERT) on the structured dataset from Phase 1, producing production-ready models for Vedic shloka classification.

---

## Architecture

```
Input shloka text
       │
  AutoTokenizer (IndicBERT / mBERT)
  max_length=256, padding, truncation
       │
  Pretrained Backbone (frozen → fine-tuned)
  last_hidden_state[:, 0, :]  ← [CLS] token
       │
  LayerNorm  →  Dropout(0.1)
       │
  Linear(hidden_size → num_labels)
       │
  CrossEntropyLoss (class-weighted)
       │
  Predicted label + softmax confidence
```

Three independent models share the same architecture with different output heads:

| Model | Task | Labels |
|-------|------|--------|
| `veda_classifier` | Which Veda? | Rigveda, Samaveda, Yajurveda, Atharvaveda, Other, Unknown |
| `domain_classifier` | Knowledge domain | Mathematics, Astronomy, Philosophy, Grammar … |
| `branch_classifier` | Mathematical branch | Geometry, Algebra, Arithmetic, Trigonometry … |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all 3 models (downloads IndicBERT on first run, ~500MB)
python p2_run_training.py

# Train with custom hyperparameters
python p2_run_training.py --epochs 3 --batch-size 8 --lr 3e-5

# Train a single task
python p2_run_training.py --tasks veda --epochs 5

# Run inference on a shloka
python -m scripts.phase2.p2_inference "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।"

# Run all tests
pytest tests/ -v
```

---

## Module Reference

### `phase2_config.py`
Central configuration: label sets, task registry, hyperparameters, directory paths.

### `scripts/phase2/p2_dataset.py`
- `ShlokaDataset` — PyTorch Dataset over tokenised shloka text
- `load_tokenizer()` — loads IndicBERT tokenizer with mBERT fallback
- `build_label_encoder()` — fits sklearn `LabelEncoder` over all known + observed labels
- `make_dataloaders()` — returns `(train_dl, val_dl, test_dl, le, data_info)`

### `scripts/phase2/p2_model.py`
- `ShlokaClassifier` — pretrained backbone + LayerNorm + Dropout + Linear head
- `.save(model_dir)` — saves backbone (HuggingFace format) + head state dict
- `.load(model_dir)` — reconstructs model from disk
- Supports IndicBERT with automatic fallback to `bert-base-multilingual-cased`

### `scripts/phase2/p2_trainer.py`
- `train_classifier()` — full training loop with:
  - AdamW + linear warm-up scheduler
  - Class-weighted `CrossEntropyLoss`
  - Automatic AMP (mixed precision) on GPU
  - Per-epoch validation + best-model checkpointing
  - Structured logging + version registry update

### `scripts/phase2/p2_evaluator.py`
- `evaluate_model()` — runs test-set inference and computes:
  - Accuracy, weighted P/R/F1, macro P/R/F1, per-class metrics
  - Confusion matrix (PNG plot)
  - Training curves (loss + accuracy, PNG plot)
  - `reports/<task>/eval_results.json`
  - `reports/<task>/classification_report.txt`

### `scripts/phase2/p2_inference.py`
- `ShlokaInferencePipeline` — loads all three models, predicts all tasks in one call
- `predict(text)` → structured dict with labels, per-task confidence, class probabilities
- `predict_batch(texts)` → list of prediction dicts
- Overall confidence = geometric mean of per-task softmax max probabilities

### `p2_run_training.py`
CLI master runner. Trains + evaluates all tasks end-to-end. Saves `logs/phase2_summary.json`.

---

## Output Schema

```json
{
  "veda":   "Rigveda",
  "domain": "Mathematics",
  "branch": "Arithmetic",
  "confidence": {
    "veda":    0.91,
    "domain":  0.84,
    "branch":  0.73,
    "overall": 0.82
  },
  "probabilities": {
    "veda":   { "Rigveda": 0.91, "Unknown": 0.05, "Samaveda": 0.02, ... },
    "domain": { "Mathematics": 0.84, "Unknown": 0.10, ... },
    "branch": { "Arithmetic": 0.73, "Unknown": 0.15, ... }
  }
}
```

---

## Saved Artefacts

```
models/
├── version_registry.json          ← all training runs with metrics
├── veda_classifier/
│   ├── backbone/                  ← HuggingFace model weights
│   ├── tokenizer/                 ← saved tokenizer
│   ├── classifier_head.pt         ← LayerNorm + dropout + linear head
│   ├── label_encoder.pkl          ← fitted sklearn LabelEncoder
│   ├── training_config.json       ← hyperparameters + dataset info
│   ├── training_history.json      ← per-epoch loss/accuracy/F1
│   └── best_metrics.json          ← best validation metrics
├── domain_classifier/             ← same structure
└── branch_classifier/             ← same structure

reports/
├── veda/
│   ├── eval_results.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_curves.png
├── domain/
└── branch/
```

---

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRETRAINED_MODEL` | `ai4bharat/indic-bert` | HuggingFace model ID |
| `MAX_SEQ_LENGTH` | 256 | Tokenizer max length |
| `BATCH_SIZE` | 16 | Mini-batch size |
| `LEARNING_RATE` | 2e-5 | AdamW learning rate |
| `EPOCHS` | 5 | Training epochs |
| `DROPOUT` | 0.1 | Classifier head dropout |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `WARMUP_RATIO` | 0.1 | Fraction of steps for LR warm-up |
| `RANDOM_SEED` | 42 | Global reproducibility seed |

Override via CLI flags or the `PRETRAINED_MODEL` environment variable.

---

## Hardware

- **CPU**: Works out-of-the-box (slower)
- **GPU**: Auto-detected; enables mixed-precision (AMP) training automatically
- **Multi-GPU**: Use `torchrun` or HuggingFace Accelerate

---

## Reproducibility

- `RANDOM_SEED` seeds Python `random`, NumPy, PyTorch, and CUDA
- `torch.backends.cudnn.deterministic = True`
- All hyperparameters saved in `training_config.json`
- Dataset version tracked via Phase 1 `metadata/source_registry.json`
- Every training run versioned in `models/version_registry.json`

---

## Phase 3 Compatibility

The inference pipeline exposes a stable dict schema and can be imported directly:

```python
from scripts.phase2.p2_inference import ShlokaInferencePipeline
pipe = ShlokaInferencePipeline()
result = pipe.predict(shloka_text)
# result["veda"], result["domain"], result["branch"], result["confidence"]
```
