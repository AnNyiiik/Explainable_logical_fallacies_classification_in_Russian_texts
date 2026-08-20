#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-deepvk/USER-bge-m3}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"

SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-./data/saved_models}"
# Override MODEL_PATH when training the same model on different data (e.g. the
# English translation), otherwise the second run overwrites the first.
MODEL_PATH="${MODEL_PATH:-$SAVED_MODELS_DIR/$MODEL_THING/}"

DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train_ru.tsv}"
VALID_FILE="${VALID_FILE:-$DATA_DIR/validate_ru.tsv}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test_ru.tsv}"

# Values selected by the Optuna search for USER-bge-m3. The learning rate was
# tuned for an effective batch of 8 (4 x 2 accumulation steps), so changing the
# batch size means the search has to be repeated.
N_EPOCHS="${N_EPOCHS:-5}"
PATIENCE="${PATIENCE:-5}"
LR="${LR:-1.23e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0012}"
EXP_WEIGHT="${EXP_WEIGHT:-0.25}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-512}"
CLS_HIDDEN_SIZE="${CLS_HIDDEN_SIZE:-128}"
EXP_HIDDEN_SIZE="${EXP_HIDDEN_SIZE:-128}"
SEED="${SEED:-42}"

mkdir -p "$MODEL_PATH"

uv run python ./code/main.py -mode train \
   -train_file "$TRAIN_FILE" \
   -valid_file "$VALID_FILE" \
   -test_file "$TEST_FILE" \
   -model_config "$MODEL" \
   -saved_model_path "$MODEL_PATH" \
   -n_epochs "$N_EPOCHS" \
   -patience "$PATIENCE" \
   -lr "$LR" \
   -weight_decay "$WEIGHT_DECAY" \
   -exp_weight "$EXP_WEIGHT" \
   -train_batch_size "$TRAIN_BATCH_SIZE" \
   -accumulation_steps "$ACCUMULATION_STEPS" \
   -test_batch_size "$TEST_BATCH_SIZE" \
   -max_len "$MAX_LEN" \
   -cls_hidden_size "$CLS_HIDDEN_SIZE" \
   -exp_hidden_size "$EXP_HIDDEN_SIZE" \
   -seed "$SEED"