# `code/` — Model, Training & Inference

This package contains both stages of the project: the **binary fallacy
detector** and the **multi-class explainable TACEI pipeline**, plus shared
utilities and hyperparameter tuning.

## Modules

### Multi-class TACEI

| File | Role |
|------|------|
| `main.py` | TACEI entry point. Handles `train`, `eval` (cross-validation), and `prediction` modes; data loading & preprocessing; W&B logging. |
| `mtlPredictor.py` | **Phase 1 (MTL).** Multi-task trainer that jointly learns fallacy classification and token-level rationale extraction. |
| `classPredictor.py` | **Phase 2 (CLS).** Classifier trained on rationale-masked text from Phase 1 to produce the final label. |
| `optuna_tuning.py` | Hyperparameter search over the two-phase pipeline, optimizing test macro-F1. |
| `utils.py` | Tokenization, rationale-label alignment, evaluation metrics, and loss functions. |
| `tweet_preprocessing.py` | Cyrillic-safe text normalization (`normalizeTweet`). |

### Binary detection

| File | Role |
|------|------|
| `train_contrastive.py` | Contrastively fine-tunes the `deepvk/USER-bge-m3` sentence encoder on `(fallacious, neutral)` pairs using a cached multiple-negatives ranking loss; saves the fine-tuned encoder to `user-bge-contrastive-finetuned/`. |
| `train_binary_classifier.py` | Trains a binary `fallacy` / `neutral` classifier on top of the (fine-tuned or base) encoder, with AMP and gradient accumulation; reports accuracy and F1 on a held-out test split. |

## Binary detection stage

Run the two scripts in order. They read their data from the working directory by
default (see paths inside each script / `data/binary_detection_data/`).

```bash
# 1) Contrastively fine-tune the encoder
uv run python ./code/train_contrastive.py
#    → writes ./user-bge-contrastive-finetuned/

# 2) Train the binary classifier on top of that encoder
uv run python ./code/train_binary_classifier.py
#    → writes ./best_binary_classifier_full.pt
```

Inputs expected by these scripts:

- `all_pairs.json` — list of `{original_text, neutral_text, original_label}`
- `extra_fallacies.json`, `extra_neutrals.json` — optional extra examples
  (skipped automatically if missing)

The encoder used by the classifier is set via `encoder_name` in
`train_binary_classifier.py` (`"user-bge-contrastive-finetuned"` by default,
or `"deepvk/USER-bge-m3"` to skip contrastive fine-tuning).

<!-- TODO: if these scripts should read directly from data/binary_detection_data/,
     update the hardcoded paths or expose them as CLI arguments. -->

## Multi-class TACEI stage

### `main.py` modes

```bash
# Train using pre-split files
uv run python ./code/main.py -mode train \
  -train_file ./data/multiclass_TACEI_data/train_ru.tsv \
  -valid_file ./data/multiclass_TACEI_data/validate_ru.tsv \
  -test_file  ./data/multiclass_TACEI_data/test_ru.tsv \
  -model_config deepvk/USER-bge-m3 \
  -saved_model_path ./data/saved_models/USER-bge-m3/

# Cross-validated evaluation on a single file
uv run python ./code/main.py -mode eval \
  -input_path ./data/multiclass_TACEI_data/train_ru.tsv -n_folds 5

# Inference on new, unlabeled text (one example per line)
uv run python ./code/main.py -mode prediction \
  -input_new_data_path ./data/test_ru.txt \
  -output_new_data_path ./data/output.csv \
  -saved_model_path ./data/saved_models/USER-bge-m3/
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-mode` | `eval` | `train`, `eval`, or `prediction` |
| `-model_config` | `vinai/bertweet-base` | HuggingFace encoder name |
| `-lr` | `2e-5` | Learning rate (AdamW) |
| `-weight_decay` | `0.01` | Weight decay |
| `-exp_weight` | `0.07` | Weight of the rationale (explanation) loss |
| `-n_epochs` | `20` | Max epochs |
| `-patience` | `5` | Early-stopping patience |
| `-max_len` | `128` | Max sequence length (tokens) |
| `-train_batch_size` / `-test_batch_size` | `16` / `64` | Batch sizes |
| `-cls_hidden_size` / `-exp_hidden_size` | `128` / `128` | Head hidden sizes |
| `-class_weights` | per-class defaults | Space-separated per-class loss weights (12 values, order = label map) |
| `-saved_model_path` | `data/saved_models/` | Where checkpoints are written |

Training outputs (under `-saved_model_path`): `phase1.pt`, `phase2.pt`, and full
checkpoints `phase1_best_checkpoint.pt`, `phase2_best_checkpoint.pt`.

### Hyperparameter tuning (`optuna_tuning.py`)

```bash
uv run python ./code/optuna_tuning.py \
  -train_file ./data/multiclass_TACEI_data/train_ru.tsv \
  -valid_file ./data/multiclass_TACEI_data/validate_ru.tsv \
  -test_file  ./data/multiclass_TACEI_data/test_ru.tsv \
  -model_config deepvk/USER-bge-m3 \
  -n_trials 30 -study_name tacei_optuna_ru \
  -storage sqlite:///./data/optuna/ru/optuna.db
```

## Label map

```
0  ad hominem            6  fallacy of extension
1  ad populum            7  fallacy of relevance
2  appeal to emotion     8  false causality
3  circular reasoning    9  false dilemma
4  equivocation         10  faulty generalization
5  fallacy of credibility 11 intentional
```

## Notes

- `tweet_preprocessing.normalizeTweet` is Cyrillic-safe — it does **not** strip
  non-ASCII characters, unlike English-only tweet normalizers.
- Random seeds are fixed for reproducibility; data splits use `random_state=42`.