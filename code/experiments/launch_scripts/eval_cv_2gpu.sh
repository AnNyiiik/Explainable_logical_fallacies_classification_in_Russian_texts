#!/usr/bin/env bash
# Run one seed's cross-validation with the folds spread across two GPUs.
#
# Folds are independent, so this is exact parallelism: each fold trains alone on
# one card with the batch size unchanged, and the result is identical to running
# the folds one after another. (DataParallel would not be -- it splits the batch
# across cards and changes the effective batch size.)
#
#   SEED=777 bash eval_cv_2gpu.sh
#
set -euxo pipefail

SEED="${SEED:-42}"
N_FOLDS="${N_FOLDS:-5}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
INPUT_FILE="${INPUT_FILE:-$DATA_DIR/train_ru_cv.tsv}"
MODEL="${MODEL:-deepvk/USER-bge-m3}"

# Fixed at the values Optuna selected -- do not change these without re-running
# the search, since the learning rate was tuned for this effective batch size.
LR="${LR:-1.23e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0012}"
EXP_WEIGHT="${EXP_WEIGHT:-0.25}"
N_EPOCHS="${N_EPOCHS:-5}"
PATIENCE="${PATIENCE:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-512}"
CLS_HIDDEN_SIZE="${CLS_HIDDEN_SIZE:-128}"
EXP_HIDDEN_SIZE="${EXP_HIDDEN_SIZE:-128}"
TEST_SIZE="${TEST_SIZE:-15}"

# Split the folds between the cards: even indices on A, odd on B.
FOLDS_A=""
FOLDS_B=""
for ((f = 0; f < N_FOLDS; f++)); do
    if (( f % 2 == 0 )); then FOLDS_A="${FOLDS_A}${FOLDS_A:+,}$f"
    else                     FOLDS_B="${FOLDS_B}${FOLDS_B:+,}$f"; fi
done

run_folds () {
    local gpu="$1" folds="$2" tag="$3"
    [ -z "$folds" ] && return 0
    CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled \
    uv run python ./code/main.py -mode eval \
        -input_path "$INPUT_FILE" \
        -model_config "$MODEL" \
        -seed "$SEED" \
        -folds "$folds" \
        -n_folds "$N_FOLDS" \
        -test_size "$TEST_SIZE" \
        -lr "$LR" \
        -weight_decay "$WEIGHT_DECAY" \
        -exp_weight "$EXP_WEIGHT" \
        -n_epochs "$N_EPOCHS" \
        -patience "$PATIENCE" \
        -train_batch_size "$TRAIN_BATCH_SIZE" \
        -accumulation_steps "$ACCUMULATION_STEPS" \
        -test_batch_size "$TEST_BATCH_SIZE" \
        -max_len "$MAX_LEN" \
        -cls_hidden_size "$CLS_HIDDEN_SIZE" \
        -exp_hidden_size "$EXP_HIDDEN_SIZE" \
        > "cv_seed${SEED}_${tag}.log" 2>&1
}

echo "seed $SEED: folds [$FOLDS_A] on GPU $GPU_A, folds [$FOLDS_B] on GPU $GPU_B"
run_folds "$GPU_A" "$FOLDS_A" "gpuA" &
PID_A=$!
run_folds "$GPU_B" "$FOLDS_B" "gpuB" &
PID_B=$!

FAILED=0
wait $PID_A || FAILED=1
wait $PID_B || FAILED=1
if [ "$FAILED" -ne 0 ]; then
    echo "at least one process failed -- see cv_seed${SEED}_gpu*.log" >&2
    exit 1
fi

# Merge the per-fold metrics into the single file the analysis scripts expect.
python - "$SEED" "$N_FOLDS" <<'PYEOF'
import glob, os, sys
import pandas as pd
seed, n_folds = sys.argv[1], int(sys.argv[2])
d = os.path.join('results', f'seed_{seed}', 'metrics')
files = sorted(glob.glob(os.path.join(d, 'fold*.csv')),
               key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
if len(files) != n_folds:
    sys.exit(f"expected {n_folds} fold metric files in {d}, found {len(files)}")
out = f'results/fold_metrics_seed_{seed}.csv'
pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values('fold') \
  .to_csv(out, index=False)
print(f"merged {len(files)} folds into {out}")
PYEOF
