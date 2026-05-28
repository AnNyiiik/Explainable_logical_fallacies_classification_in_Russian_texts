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

The multi-class stage adapts the [TACEI](https://github.com/thi-huyennguyen/TACEI)
architecture to Russian-language logical-fallacy detection.

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
│   ├── data_generation/                 #   Synthetic data generation utilities
│   │   ├── data_generation.py           #     Fallacy example generator
│   │   ├── generate_neutral_examples.py #     Neutral example generator
│   │   ├── neutralize_examples.py       #     Build contrastive (fallacy, neutral) pairs
│   │   ├── translate_data_with_gpt.py   #     DeepL + LLM span-aligned translation
│   │   ├── prompts_for_data_generation/ #     Prompts for fallacy generation (per class)
│   │   ├── prompts_for_contrastive_pairs/ #   Prompts for neutralization / contrastive pairs
│   │   └── README.md
│   ├── experiments/                     #   Experiment entry points
│   │   ├── few-shot.py                  #     Few-shot / zero-shot LLM baseline
│   │   ├── launch_scripts/              #     Parameterized run scripts (.sh)
│   │   │   ├── train.sh                 #       TACEI training (multilingual default)
│   │   │   ├── train_USER-bge.sh        #       TACEI training, deepvk/USER-bge-m3
│   │   │   ├── train_roberta_large.sh   #       TACEI training, roberta-large (EN)
│   │   │   ├── tune_USER-bge.sh         #       Optuna search (RU)
│   │   │   ├── tune_roberta.sh          #       Optuna search (EN)
│   │   │   ├── prediction.sh            #       Inference
│   │   │   ├── test_roberta_large.sh    #       Extract text from TSV → run inference
│   │   │   ├── pretrain_encoder_contrastive.sh  # Contrastive encoder fine-tuning
│   │   │   ├── train_binary.sh          #       Binary classifier training
│   │   │   └── few-shot.sh              #       Few-shot / zero-shot baseline
│   │   └── README.md
│   └── README.md
├── data/
│   ├── binary_detection_data/           #   JSON pairs & extras for the binary stage
│   └── multiclass_TACEI_data/           #   TSV splits for the TACEI stage
├── requirements.txt                     # Pinned Python dependencies
└── README.md                            # This file
```

> **Note.** Trained checkpoints are not committed to the repository. Their
> location is controlled per-script by the `MODEL_PATH` environment variable
> (e.g. `MODEL_PATH=./data/saved_models/$MODEL_THING/`). See
> `code/experiments/README.md` for all overridable variables.

## Installation

The project uses [**uv**](https://docs.astral.sh/uv/) as the primary
environment manager (the run scripts call `uv run python ...`), but a plain
`pip` workflow is also supported.

### Requirements

- Python ≥ 3.10
- For training: a CUDA-capable GPU (≥ 12 GB VRAM recommended for
  `roberta-large`, `xlm-roberta-large`, `deepvk/USER-bge-m3` at
  `max_len=512`).
- For data generation / few-shot baselines: an OpenAI-compatible API key
  (ProxyAPI) and optionally a DeepL key (translation only).

### Option A — uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone
git clone https://github.com/AnNyiiik/Explainable_logical_fallacies_classification_in_Russian_texts.git
cd Explainable_logical_fallacies_classification_in_Russian_texts

# Resolve & install all dependencies into a project-local .venv
uv venv
uv pip install -r requirements.txt
```

If `pyproject.toml` / `uv.lock` are present, `uv sync` does the same in one
step.

### Option B — pip + venv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch with CUDA

`pip install torch` installs the CPU build by default. For GPU training,
install the CUDA build that matches your driver — for example, CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

See the [PyTorch install matrix](https://pytorch.org/get-started/locally/) for
the right index URL for your system.

### API keys (only for data generation / few-shot)

These scripts read secrets from the environment — never commit keys to git:

```bash
export PROXYAPI_KEY="sk-..."     # OpenAI-compatible endpoint (ProxyAPI)
export DEEPL_AUTH_KEY="..."      # only for translate_data_with_gpt.py
```

The shell scripts in `code/experiments/` do **not** pass the key as a flag
(it would leak into `set -x` logs and shell history), so exporting the
variable is required.

## Data

### Binary detection (`data/binary_detection_data/`)

Inputs used by `train_contrastive.py` and `train_binary_classifier.py`:

| File | Schema | Used for |
|------|--------|----------|
| `all_pairs.json` | `{original_text, neutral_text, original_label}` | contrastive pairs + binary labels |
| `extra_fallacies.json` | `[{text}, ...]` or `[str, ...]` (optional) | extra positive (fallacy) examples |
| `extra_neutrals.json` | `[{text}, ...]` or `[str, ...]` (optional) | extra negative (neutral) examples |

### Multi-class TACEI (`data/multiclass_TACEI_data/`)

Tab-separated (`.tsv`) splits with at least these columns:

| Column | Description |
|--------|-------------|
| `text` | The input text to classify |
| `label` | One of the 12 fallacy labels above |
| `rationale` | The fallacious span(s), `[SEP]`-separated if multiple |

Expected splits: `train_{ru,en}.tsv`, `validate_{ru,en}.tsv`, `test_{ru,en}.tsv`.

## Pretrained models

Trained model checkpoints are distributed as a single archive (they are not
committed to the repository). Download and extract them into the directory that
the scripts expect (`./data/saved_models/` by default):

```bash
# Download (~ 22 GB)
curl -L -o models.tar.zstd https://timafrolov.me/models.tar.zstd

# Extract (requires zstd: `apt install zstd` / `brew install zstd`)
mkdir -p ./data/saved_models
tar --use-compress-program=unzstd -xf models.tar.zstd -C ./data/saved_models

# Clean up the archive afterwards (optional)
rm models.tar.zstd
```

After extraction, point the relevant script at the model with `MODEL_PATH`
(or `SAVED_MODELS_DIR`) if your layout differs from the default — for example:

```bash
MODEL="deepvk/USER-bge-m3"   MODEL_PATH=./data/saved_models/deepvk_USER-bge-m3/   ./code/experiments/launch_scripts/prediction.sh
```

> If `tar` on your system does not support `--use-compress-program`, extract in
> two steps: `unzstd models.tar.zstd && tar -xf models.tar`.

## Quick start

```bash
# --- Binary detection stage ---
./code/experiments/launch_scripts/pretrain_encoder_contrastive.sh   # 1) fine-tune encoder
./code/experiments/launch_scripts/train_binary.sh                   # 2) train binary head

# --- Multi-class TACEI stage ---
./code/experiments/launch_scripts/train.sh                          # train with defaults (RU, xlm-roberta-large)
MODEL="deepvk/USER-bge-m3" LR=1e-5 N_EPOCHS=10 ./code/experiments/launch_scripts/train.sh

# Hyperparameter search
N_TRIALS=50 ./code/experiments/launch_scripts/tune_USER-bge.sh

# Inference with a trained TACEI model
MODEL="roberta-large" ./code/experiments/launch_scripts/prediction.sh

# --- LLM baseline ---
export PROXYAPI_KEY="..."
MODE=both ./code/experiments/launch_scripts/few-shot.sh
```

All scripts accept any parameter as an environment variable; see
`code/experiments/README.md` for the full list per script.

## Reproducibility

- Random seeds are fixed in `code/main.py` and `code/optuna_tuning.py`
  (`random`, `numpy`, `torch`; cuDNN deterministic). Data splits use a fixed
  `random_state=42`.
- Optuna studies persist to SQLite, so searches are resumable and inspectable.
- Experiment configuration is captured via Weights & Biases when enabled.
- Trained model paths are controlled per-script by `MODEL_PATH`, so runs of
  different configurations don't collide.