#!/usr/bin/env bash
set -e

SEEDS=(12345 67891 54321 999 777)
GPUS=(0 1 2 3)

echo "=== Launching first 4 seeds on 4 GPUs ==="
for i in 0 1 2 3; do
    seed=${SEEDS[$i]}
    gpu=${GPUS[$i]}
    echo "Starting seed $seed on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu SEED=$seed bash ./code/experiments/launch_scripts/eval.sh &
done

wait
echo "First 4 seeds completed."

LAST_SEED=${SEEDS[4]}
echo "=== Starting seed $LAST_SEED: parallel folds ==="

run_fold() {
    local fold=$1
    local gpu=$2
    echo "  Starting fold $fold of seed $LAST_SEED on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python ./code/main.py -mode eval \
        -seed "$LAST_SEED" \
        -input_path ./data/multiclass_TACEI_data/train_ru.tsv \
        -n_folds 5 \
        -model_config deepvk/USER-bge-m3 \
        -n_epochs 5 \
        -patience 3 \
        -lr 2e-5 \
        -exp_weight 0.2 \
        -train_batch_size 4 \
        -test_batch_size 32 \
        -max_len 512 \
        -cls_hidden_size 128 \
        -exp_hidden_size 128 \
        -accumulation_steps 2 \
        -fold_index "$fold" > "logs_fold_${fold}.log" 2>&1
}
export -f run_fold

echo "Launching folds 0..3 on GPUs 0..3"
for fold in 0 1 2 3; do
    gpu=$((fold))
    run_fold "$fold" "$gpu" &
done
wait

echo "Launching fold 4 on GPU 0"
run_fold 4 0

echo "All folds for seed $LAST_SEED completed."
echo "All experiments finished."