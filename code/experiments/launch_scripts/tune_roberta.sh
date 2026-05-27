#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-roberta-large}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"

SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-./data/saved_models}"
MODEL_PATH="${MODEL_PATH:-$SAVED_MODELS_DIR/$MODEL_THING/}"

DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train_en.tsv}"
VALID_FILE="${VALID_FILE:-$DATA_DIR/validate_en.tsv}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test_en.tsv}"

N_EPOCHS="${N_EPOCHS:-5}"
PATIENCE="${PATIENCE:-3}"
LR="${LR:-2e-5}"
EXP_WEIGHT="${EXP_WEIGHT:-0.2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-512}"
CLS_HIDDEN_SIZE="${CLS_HIDDEN_SIZE:-128}"
EXP_HIDDEN_SIZE="${EXP_HIDDEN_SIZE:-128}"

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
   -exp_weight "$EXP_WEIGHT" \
   -train_batch_size "$TRAIN_BATCH_SIZE" \
   -test_batch_size "$TEST_BATCH_SIZE" \
   -max_len "$MAX_LEN" \
   -cls_hidden_size "$CLS_HIDDEN_SIZE" \
   -exp_hidden_size "$EXP_HIDDEN_SIZE"