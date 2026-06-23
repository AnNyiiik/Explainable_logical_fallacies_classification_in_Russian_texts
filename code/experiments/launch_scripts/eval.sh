#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-deepvk/USER-bge-m3}"
DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
INPUT_FILE="${INPUT_FILE:-$DATA_DIR/train_ru.tsv}"
N_FOLDS="${N_FOLDS:-5}"
SEEDS="${SEEDS:-12345 67891 54321 999 777}"

MAX_LEN="${MAX_LEN:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
CLS_HIDDEN_SIZE="${CLS_HIDDEN_SIZE:-128}"
EXP_HIDDEN_SIZE="${EXP_HIDDEN_SIZE:-128}"
N_EPOCHS="${N_EPOCHS:-15}"
PATIENCE="${PATIENCE:-5}"
LR="${LR:-1.23e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0012}"
EXP_WEIGHT="${EXP_WEIGHT:-0.25}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
USE_AMP="${USE_AMP:-false}"

CLASS_WEIGHTS="${CLASS_WEIGHTS:-1.0 1.6 1.0 1.4 1.0 1.1 1.0 1.0 1.6 1.0 1.0 1.6}"

export WANDB_MODE=disabled

echo "=== DEBUG ==="
echo "MODEL: $MODEL"
echo "INPUT_FILE: $INPUT_FILE"
echo "CLASS_WEIGHTS: $CLASS_WEIGHTS"
echo "Current directory: $(pwd)"
ls -la "$INPUT_FILE" || { echo "❌ Файл $INPUT_FILE не найден!"; exit 1; }
echo "============="

mkdir -p results

for SEED in $SEEDS; do
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
        -weight_decay "$WEIGHT_DECAY" \
        -exp_weight "$EXP_WEIGHT" \
        -train_batch_size "$TRAIN_BATCH_SIZE" \
        -test_batch_size "$TEST_BATCH_SIZE" \
        -max_len "$MAX_LEN" \
        -cls_hidden_size "$CLS_HIDDEN_SIZE" \
        -exp_hidden_size "$EXP_HIDDEN_SIZE" \
        -class_weights $CLASS_WEIGHTS \
        -accumulation_steps "$ACCUMULATION_STEPS" \
        $( [ "$USE_AMP" = "true" ] && echo "-use_amp" )
done

echo "All experiments completed. Results saved in ./results/"