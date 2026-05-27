#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-deepvk/USER-bge-m3}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"
OPTUNA_DIR="${OPTUNA_DIR:-./data/optuna/$MODEL_THING/}"
mkdir -p "$OPTUNA_DIR"

TRAIN_FILE="${TRAIN_FILE:-./data/train_ru.tsv}"
VALID_FILE="${VALID_FILE:-./data/validate_ru.tsv}"
TEST_FILE="${TEST_FILE:-./data/test_ru.tsv}"

MAX_LEN="${MAX_LEN:-256}"
CLS_HIDDEN_SIZE="${CLS_HIDDEN_SIZE:-128}"
EXP_HIDDEN_SIZE="${EXP_HIDDEN_SIZE:-128}"
N_TRIALS="${N_TRIALS:-30}"
STORAGE="${STORAGE:-sqlite:///${OPTUNA_DIR}/optuna.db}"
STUDY_NAME="${STUDY_NAME:-tacei_optuna_${MODEL_THING}}"

uv run python ./code/optuna_tuning.py \
    -train_file "$TRAIN_FILE" \
    -valid_file "$VALID_FILE" \
    -test_file "$TEST_FILE" \
    -model_config "$MODEL" \
    -max_len "$MAX_LEN" \
    -cls_hidden_size "$CLS_HIDDEN_SIZE" \
    -exp_hidden_size "$EXP_HIDDEN_SIZE" \
    -n_trials "$N_TRIALS" \
    -study_name "$STUDY_NAME" \
    -storage "$STORAGE"