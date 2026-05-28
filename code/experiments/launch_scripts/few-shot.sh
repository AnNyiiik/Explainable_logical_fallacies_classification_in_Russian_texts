#!/usr/bin/env bash
set -euxo pipefail

# API key must be provided via the environment (do not hardcode):
#   export PROXYAPI_KEY="..."
#
# Run mode is controlled by the MODE variable (default: few_shot):
#   MODE=zero_shot ./few-shot.sh   # only zero-shot
#   MODE=few_shot  ./few-shot.sh   # only few-shot (default)
#   MODE=both      ./few-shot.sh   # both, sequentially

API_BASE="${API_BASE:-https://openai.api.proxyapi.ru/v1}"
MODEL="${MODEL:-claude-haiku-4-5}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"

DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
TRAIN_PATH="${TRAIN_PATH:-$DATA_DIR/train_ru.tsv}"
TEST_PATH="${TEST_PATH:-$DATA_DIR/test_ru.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-./code/experiments/few_shot_results/$MODEL_THING/}"

MODE="${MODE:-few_shot}"
case "$MODE" in
    zero_shot|few_shot|both) ;;
    *)
        echo "Error: MODE must be one of: zero_shot, few_shot, both (got: '$MODE')" >&2
        exit 1
        ;;
esac

FEW_SHOT_EXAMPLES_PER_CLASS="${FEW_SHOT_EXAMPLES_PER_CLASS:-3}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-1202}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60.0}"
SLEEP="${SLEEP:-0.3}"
MAX_TOKENS="${MAX_TOKENS:-800}"

mkdir -p "$OUTPUT_DIR"

uv run python ./code/experiments/few-shot.py \
    -api_base "$API_BASE" \
    -model "$MODEL" \
    -train_path "$TRAIN_PATH" \
    -test_path "$TEST_PATH" \
    -output_dir "$OUTPUT_DIR" \
    -mode "$MODE" \
    -few_shot_examples_per_class "$FEW_SHOT_EXAMPLES_PER_CLASS" \
    -max_test_samples "$MAX_TEST_SAMPLES" \
    -request_timeout "$REQUEST_TIMEOUT" \
    -sleep "$SLEEP" \
    -max_tokens "$MAX_TOKENS"