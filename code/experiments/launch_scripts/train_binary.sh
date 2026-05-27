#!/usr/bin/env bash
set -euxo pipefail

DATA_DIR="${DATA_DIR:-./data/binary_detection_data}"
PAIRS_FILE="${PAIRS_FILE:-$DATA_DIR/all_pairs.json}"
EXTRA_FALLACIES_FILE="${EXTRA_FALLACIES_FILE:-$DATA_DIR/extra_fallacies.json}"
EXTRA_NEUTRALS_FILE="${EXTRA_NEUTRALS_FILE:-$DATA_DIR/extra_neutrals.json}"

SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-./data/saved_models}"
BINARY_DIR="${BINARY_DIR:-$SAVED_MODELS_DIR/binary}"
ENCODER_NAME="${ENCODER_NAME:-$BINARY_DIR/user-bge-contrastive-finetuned/}"
BEST_MODEL_PATH="${BEST_MODEL_PATH:-$BINARY_DIR/best_binary_classifier_full.pt}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$BEST_MODEL_PATH")"

uv run python ./code/train_binary_classifier.py \
    -pairs_file "$PAIRS_FILE" \
    -extra_fallacies_file "$EXTRA_FALLACIES_FILE" \
    -extra_neutrals_file "$EXTRA_NEUTRALS_FILE" \
    -encoder_name "$ENCODER_NAME" \
    -best_model_path "$BEST_MODEL_PATH" \
    -max_seq_length "$MAX_SEQ_LENGTH" \
    -batch_size "$BATCH_SIZE" \
    -accumulation_steps "$ACCUMULATION_STEPS" \
    -epochs "$EPOCHS" \
    -lr "$LR" \
    -seed "$SEED"