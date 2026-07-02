#!/usr/bin/env bash
set -e

SEEDS=(12345 67891 54321 999 777)

for SEED in "${SEEDS[@]}"; do
    echo "========================================="
    echo "Running roberta-large with seed $SEED"
    echo "========================================="
    SEED=$SEED bash ./code/experiments/launch_scripts/run_eval_parallel.sh
done

echo "All roberta-large experiments completed."