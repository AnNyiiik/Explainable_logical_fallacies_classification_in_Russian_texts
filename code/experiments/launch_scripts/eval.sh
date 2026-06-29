#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-deepvk/USER-bge-m3}"
DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
INPUT_FILE="${INPUT_FILE:-$DATA_DIR/train_ru.tsv}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-12345}"

N_EPOCHS="${N_EPOCHS:-5}"
PATIENCE="${PATIENCE:-3}"
LR="${LR:-2e-5}"
EXP_WEIGHT="${EXP_WEIGHT:-0.2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-512}"
CLS_HIDDEN_SIZE="${CLS_HIDDEN_SIZE:-128}"
EXP_HIDDEN_SIZE="${EXP_HIDDEN_SIZE:-128}"

ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"

export WANDB_MODE=disabled

mkdir -p results

echo "========================================="
echo "Running cross-validation with seed=$SEED"
echo "========================================="
uv run python ./code/main.py -mode eval \
    -seed "$SEED" \
    -input_path "$INPUT_FILE" \
    -n_folds "$N_FOLDS" \
    -model_config "$MODEL" \
    -n_epochs "$N_EPOCHS" \
    -patience "$PATIENCE" \
    -lr "$LR" \
    -exp_weight "$EXP_WEIGHT" \
    -train_batch_size "$TRAIN_BATCH_SIZE" \
    -test_batch_size "$TEST_BATCH_SIZE" \
    -max_len "$MAX_LEN" \
    -cls_hidden_size "$CLS_HIDDEN_SIZE" \
    -exp_hidden_size "$EXP_HIDDEN_SIZE" \
    -accumulation_steps "$ACCUMULATION_STEPS"

echo "Experiment completed. Results saved in ./results/"