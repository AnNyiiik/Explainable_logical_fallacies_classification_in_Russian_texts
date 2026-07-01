#!/usr/bin/env bash
set -e

SEEDS=(12345 67891 54321 999 777)

for SEED in "${SEEDS[@]}"; do
    echo "========================================="
    echo "Running seed $SEED"
    echo "========================================="
    SEED=$SEED bash ./code/experiments/launch_scripts/eval.sh
done

echo "All seeds completed. Results in ./results/"