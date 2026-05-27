# `experiments/` — Reproducible Run Scripts

Shell wrappers around `code/main.py` and `code/optuna_tuning.py`, plus few-shot
LLM baselines. Every script is **fully parameterized through environment
variables**: defaults reproduce the reference runs, and any value can be
overridden without editing the file.

## Convention

Each variable uses `${VAR:-default}`, so:

```bash
# Run with defaults
./train.sh

# Override any subset inline
MODEL="deepvk/USER-bge-m3" LR=1e-5 N_EPOCHS=10 MAX_LEN=384 ./train.sh
```

`MODEL` is slugified (`/` → `_`) to build output paths automatically.

## Training scripts

| Script | Default model | Default data | Notes |
|--------|---------------|--------------|-------|
| `train.sh` | `xlm-roberta-large` | Russian | train batch size 8 |
| `train_USER-bge.sh` | `deepvk/USER-bge-m3` | Russian | train batch size 4 |
| `train_roberta_large.sh` | `roberta-large` | English | writes to `…-2/` path |

Overridable variables: `MODEL`, `MODEL_PATH`, `TRAIN_FILE`, `VALID_FILE`,
`TEST_FILE`, `N_EPOCHS`, `PATIENCE`, `LR`, `EXP_WEIGHT`, `TRAIN_BATCH_SIZE`,
`TEST_BATCH_SIZE`, `MAX_LEN`, `CLS_HIDDEN_SIZE`, `EXP_HIDDEN_SIZE`.

```bash
./train.sh
./train_USER-bge.sh
MODEL="FacebookAI/roberta-large" ./train_roberta_large.sh
```

## Tuning scripts

| Script | Default model | Default data |
|--------|---------------|--------------|
| `tune_ru.sh` | `deepvk/USER-bge-m3` | Russian |
| `tune_en.sh` | `FacebookAI/roberta-large` | English |

Overridable variables: the training set above, plus `N_TRIALS`, `STORAGE`,
`STUDY_NAME`, `OPTUNA_DIR`. Studies persist to SQLite and are resumable.

```bash
N_TRIALS=50 ./tune_ru.sh
MODEL="FacebookAI/roberta-large" N_TRIALS=30 ./tune_en.sh
```

## Inference / test scripts

| Script | Purpose |
|--------|---------|
| `prediction.sh` | Run a trained model on `INPUT_DATA`, write predictions to `OUTPUT_DATA`. |
| `test_roberta_large.sh` | Extract the text column from a test TSV (`cut -f$TEXT_COL`) and run prediction. |

```bash
MODEL="roberta-large" ./prediction.sh
TEXT_COL=2 ./test_roberta_large.sh
```

## Few-shot LLM baseline

<!-- TODO: confirm the filename of the few-shot evaluation script in this folder. -->
`few_shot_eval.py` evaluates zero-shot / few-shot classification against the
OpenAI-compatible endpoint, with per-example checkpointing and token-level
rationale F1.

```bash
export PROXYAPI_KEY="..."
python few_shot_eval.py \
  -train_path ./data/train_ru.tsv \
  -test_path  ./data/test_ru.tsv \
  -output_dir ./few_shot_results/claude-haiku/ \
  -model claude-haiku-4-5 -mode both
```

## Reproducibility checklist

- Seeds are fixed in the underlying Python entry points.
- Output directories are derived from the model name, so parallel runs of
  different models don't collide.
- Optuna SQLite storage makes searches resumable and auditable.