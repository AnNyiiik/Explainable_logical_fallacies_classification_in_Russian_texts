# Explainable Logical Fallacies Classification in Russian Texts

An explainable pipeline for detecting and classifying logical fallacies in
Russian (and English) texts. The project combines two complementary stages:

1. **Binary detection** — does a text contain a logical fallacy at all?
2. **Multi-class, explainable classification (TACEI)** — which of the 12
   fallacy types is it, together with the **rationale** (the span of text
   responsible for the fallacy).

This repository is a research codebase oriented toward **reproducibility**: all
experiments are driven by parameterized shell scripts, and all data-generation
utilities are configurable via command-line arguments and environment variables.

> **Note:** the multi-class stage adapts the
> [TACEI](https://github.com/thi-huyennguyen/TACEI) architecture to
> Russian-language logical-fallacy detection.
> <!-- TODO: cite the original paper and your own report/thesis here -->

## Task

| Stage | Question | Output |
|-------|----------|--------|
| Binary detection | Is there a fallacy? | `fallacy` / `neutral` |
| Multi-class (TACEI) | Which fallacy, and why? | 1 of 12 labels + rationale span |

### Fallacy classes (multi-class stage)

| Idx | Label | Idx | Label |
|----:|-------|----:|-------|
| 0 | ad hominem | 6 | fallacy of extension |
| 1 | ad populum | 7 | fallacy of relevance |
| 2 | appeal to emotion | 8 | false causality |
| 3 | circular reasoning | 9 | false dilemma |
| 4 | equivocation | 10 | faulty generalization |
| 5 | fallacy of credibility | 11 | intentional |

## Architecture

### Binary detection

A sentence-embedding model (`deepvk/USER-bge-m3`) is first **contrastively
fine-tuned** on `(fallacious, neutral)` pairs, then used as the encoder for a
lightweight binary classifier (`fallacy` vs `neutral`).

```
pairs ──► [contrastive fine-tune encoder] ──► [binary classifier head] ──► fallacy / neutral
```

### Multi-class (TACEI), two phases

- **Phase 1 — MTL (`mtlPredictor.py`)**: a multi-task model on top of a
  transformer encoder that jointly learns classification and token-level
  rationale extraction.
- **Phase 2 — CLS (`classPredictor.py`)**: a classifier trained on the
  rationale-masked text from Phase 1, producing the final fallacy label.

```
text ──► [Phase 1: MTL] ──► rationale span ──► [Phase 2: CLS] ──► fallacy label
```

Both phases support gradient accumulation, mixed precision (AMP), per-class loss
weighting, early stopping, and full-checkpoint saving.

## Repository structure

```
.
├── code/                                # All source code
│   ├── main.py                          #   TACEI entry point (train / eval / prediction)
│   ├── mtlPredictor.py                  #   TACEI Phase 1 (multi-task)
│   ├── classPredictor.py                #   TACEI Phase 2 (classifier)
│   ├── optuna_tuning.py                 #   Hyperparameter search for TACEI
│   ├── utils.py                         #   Tokenization, metrics, losses
│   ├── tweet_preprocessing.py           #   Cyrillic-safe text normalization
│   ├── train_contrastive.py             #   Contrastive fine-tuning of the encoder
│   ├── train_binary_classifier.py       #   Binary fallacy / neutral classifier
│   ├── few-shot.py                      #   Few-shot / zero-shot LLM baseline
│   ├── data_generation/                 #   Synthetic data generation utilities
│   │   ├── data_generation.py           #     Fallacy example generator
│   │   ├── generate_neutral_examples.py #     Neutral example generator
│   │   ├── neutralize_examples.py       #     Build contrastive (fallacy, neutral) pairs
│   │   ├── translate_data_with_gpt.py   #     DeepL + LLM span-aligned translation
│   │   └── README.md
│   ├── experiments/                     #   Reproducible run scripts
│   │   ├── train.sh                     #     TACEI training (multilingual default)
│   │   ├── train_USER-bge.sh            #     TACEI training, deepvk/USER-bge-m3
│   │   ├── train_roberta_large.sh       #     TACEI training, roberta-large (EN)
│   │   ├── tune_ru.sh / tune_en.sh      #     Optuna hyperparameter search
│   │   ├── prediction.sh                #     Inference
│   │   ├── test_roberta_large.sh        #     Extract text from TSV → run inference
│   │   ├── train_contrastive.sh         #     Contrastive encoder fine-tuning
│   │   ├── train_binary_classifier.sh   #     Binary classifier training
│   │   ├── few-shot.sh                  #     Few-shot / zero-shot baseline
│   │   └── README.md
│   └── README.md
├── data/
│   ├── binary_detection_data/           #   JSON pairs & extras for the binary stage
│   └── multiclass_TACEI_data/           #   TSV splits for the TACEI stage
├── requirements.txt                     # Pinned Python dependencies
└── README.md                            # This file
```

## Installation

The project uses [**uv**](https://docs.astral.sh/uv/) for dependency management
(the run scripts call `uv run python ...`).

### Option A — uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/AnNyiiik/Explainable_logical_fallacies_classification_in_Russian_texts.git
cd Explainable_logical_fallacies_classification_in_Russian_texts

# Create the environment and install dependencies from pyproject.toml / uv.lock
uv sync
```

After `uv sync`, every script can be run with `uv run python ...` or the
provided `.sh` wrappers, and uv resolves the environment automatically.

### Option B — pip + venv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # TODO: export requirements if not present
```

<!-- TODO: confirm the Python version. The scripts target Python >= 3.10. -->

### GPU / hardware notes

Training large encoders (e.g. `roberta-large`, `xlm-roberta-large`,
`deepvk/USER-bge-m3`) at `max_len=512` requires a CUDA GPU. The scripts use a
small physical batch size with gradient accumulation and mixed precision to fit
long sequences in memory.

## Data

### Binary detection (`data/binary_detection_data/`)

Contrastive pairs and auxiliary examples used by `train_contrastive.py` and
`train_binary_classifier.py`:

| File | Schema | Used for |
|------|--------|----------|
| `all_pairs.json` | `{original_text, neutral_text, original_label}` | contrastive pairs + binary labels |
| `extra_fallacies.json` | `[{text}, ...]` or `[str, ...]` (optional) | extra positive (fallacy) examples |
| `extra_neutrals.json` | `[{text}, ...]` or `[str, ...]` (optional) | extra negative (neutral) examples |

<!-- TODO: confirm exact filenames/paths inside binary_detection_data/. -->

### Multi-class TACEI (`data/multiclass_TACEI_data/`)

Tab-separated (`.tsv`) splits with at least these columns:

| Column | Description |
|--------|-------------|
| `text` | The input text to classify |
| `label` | One of the 12 fallacy labels above |
| `rationale` | The fallacious span(s), `[SEP]`-separated if multiple |

Expected splits: `train_{ru,en}.tsv`, `validate_{ru,en}.tsv`, `test_{ru,en}.tsv`.

## Quick start

```bash
# --- Binary detection stage ---
# 1) contrastively fine-tune the encoder, then 2) train the binary classifier
uv run python ./code/train_contrastive.py
uv run python ./code/train_binary_classifier.py

# --- Multi-class TACEI stage ---
# Train on Russian data with the default multilingual encoder
./experiments/train.sh

# Override the model / hyperparameters without editing the script
MODEL="deepvk/USER-bge-m3" LR=1e-5 N_EPOCHS=10 ./experiments/train.sh

# Hyperparameter search
N_TRIALS=50 ./experiments/tune_ru.sh

# Inference with a trained TACEI model
MODEL="roberta-large" ./experiments/prediction.sh
```

## Reproducibility

- Random seeds are fixed in `code/main.py` and `code/optuna_tuning.py`
  (`random`, `numpy`, `torch`; cuDNN deterministic). Data splits use a fixed
  `random_state=42`.
- Optuna studies persist to SQLite, so searches are resumable and inspectable.
- Experiment configuration is captured via Weights & Biases when enabled.