#!/usr/bin/env bash
set -euxo pipefail

DATA_DIR="${DATA_DIR:-./data/binary_detection_data}"
PAIRS_FILE="${PAIRS_FILE:-$DATA_DIR/all_pairs.json}"

MODEL_NAME="${MODEL_NAME:-deepvk/USER-bge-m3}"

SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-./data/saved_models}"
BINARY_DIR="${BINARY_DIR:-$SAVED_MODELS_DIR/binary}"
OUTPUT_DIR="${OUTPUT_DIR:-$BINARY_DIR/user-bge-contrastive/}"
SAVE_DIR="${SAVE_DIR:-$BINARY_DIR/user-bge-contrastive-finetuned/}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SCALE="${SCALE:-20.0}"
EVAL_STEPS="${EVAL_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
TEST_SIZE="${TEST_SIZE:-0.1}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_DIR" "$SAVE_DIR"

uv run python ./code/train_contrastive.py \
    -pairs_file "$PAIRS_FILE" \
    -model_name "$MODEL_NAME" \
    -output_dir "$OUTPUT_DIR" \
    -save_dir "$SAVE_DIR" \
    -max_seq_length "$MAX_SEQ_LENGTH" \
    -num_train_epochs "$NUM_TRAIN_EPOCHS" \
    -per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    -gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    -warmup_ratio "$WARMUP_RATIO" \
    -scale "$SCALE" \
    -eval_steps "$EVAL_STEPS" \
    -logging_steps "$LOGGING_STEPS" \
    -test_size "$TEST_SIZE" \
    -seed "$SEED"