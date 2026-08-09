#!/usr/bin/env bash
set -euxo pipefail

# Пути к данным
DATA_DIR="${DATA_DIR:-./data/binary_detection_data}"
PAIRS_FILE="${PAIRS_FILE:-$DATA_DIR/all_pairs.json}"
EXTRA_FALLACIES_FILE="${EXTRA_FALLACIES_FILE:-$DATA_DIR/extra_fallacies.json}"
EXTRA_NEUTRALS_FILE="${EXTRA_NEUTRALS_FILE:-$DATA_DIR/extra_neutrals.json}"

# Папка для сохранения моделей
SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-./data/saved_models}"
BINARY_DIR="${BINARY_DIR:-$SAVED_MODELS_DIR/binary}"

# Выбор энкодера: если есть дообученный (контрастивный) – используем его, иначе базовый
FINETUNED_ENCODER="$BINARY_DIR/user-bge-contrastive-finetuned"
BASE_ENCODER="deepvk/USER-bge-m3"

if [ -z "${ENCODER_NAME:-}" ]; then
    if [ -f "$FINETUNED_ENCODER/config.json" ]; then
        ENCODER_NAME="$FINETUNED_ENCODER/"
        echo "[train_binary_cv] Using fine-tuned encoder: $ENCODER_NAME"
    else
        ENCODER_NAME="$BASE_ENCODER"
        echo "[train_binary_cv] Fine-tuned encoder not found at $FINETUNED_ENCODER"
        echo "[train_binary_cv] Falling back to base encoder: $ENCODER_NAME"
        echo "[train_binary_cv] (Run pretrain_encoder_contrastive.sh first to use a fine-tuned encoder.)"
    fi
else
    echo "[train_binary_cv] Using ENCODER_NAME override: $ENCODER_NAME"
fi

# Путь для сохранения финальной модели (обученной на всех train данных)
BEST_MODEL_PATH="${BEST_MODEL_PATH:-$BINARY_DIR/best_binary_classifier_full.pt}"

# Гиперпараметры
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
SEED="${SEED:-42}"

# Параметры кросс-валидации
CV_FOLDS="${CV_FOLDS:-5}"
TEST_SIZE="${TEST_SIZE:-0.15}"          # 0 = без отдельного теста
SAVE_FOLD_MODELS="${SAVE_FOLD_MODELS:-false}"   # true/false

mkdir -p "$(dirname "$BEST_MODEL_PATH")"

# Формируем команду
CMD="uv run python ./code/train_binary_cv.py"
CMD="$CMD -pairs_file \"$PAIRS_FILE\""
CMD="$CMD -extra_fallacies_file \"$EXTRA_FALLACIES_FILE\""
CMD="$CMD -extra_neutrals_file \"$EXTRA_NEUTRALS_FILE\""
CMD="$CMD -encoder_name \"$ENCODER_NAME\""
CMD="$CMD -best_model_path \"$BEST_MODEL_PATH\""
CMD="$CMD -max_seq_length \"$MAX_SEQ_LENGTH\""
CMD="$CMD -batch_size \"$BATCH_SIZE\""
CMD="$CMD -accumulation_steps \"$ACCUMULATION_STEPS\""
CMD="$CMD -epochs \"$EPOCHS\""
CMD="$CMD -lr \"$LR\""
CMD="$CMD -seed \"$SEED\""
CMD="$CMD -cv_folds \"$CV_FOLDS\""
CMD="$CMD -test_size \"$TEST_SIZE\""
[ "$SAVE_FOLD_MODELS" = "true" ] && CMD="$CMD -save_fold_models"

echo "========================================="
echo "Launching binary classifier cross-validation"
echo "========================================="
echo "Data:      $PAIRS_FILE"
echo "Encoder:   $ENCODER_NAME"
echo "Folds:     $CV_FOLDS"
echo "Test size: $TEST_SIZE"
echo "Batch:     $BATCH_SIZE (accum: $ACCUMULATION_STEPS)"
echo "Epochs:    $EPOCHS"
echo "LR:        $LR"
echo "Seed:      $SEED"
echo "Save fold models: $SAVE_FOLD_MODELS"
echo "Output:    $BEST_MODEL_PATH"
echo "Command:   $CMD"
echo "========================================="

eval $CMD

echo "Cross-validation finished. Results saved to cv_results.json"