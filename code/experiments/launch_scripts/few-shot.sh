#!/usr/bin/env bash
set -euxo pipefail

API_BASE="${API_BASE:-https://openai.api.proxyapi.ru/v1}"
MODEL="${MODEL:-claude-haiku-4-5}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"

TRAIN_PATH="${TRAIN_PATH:-./data/multiclass_TACEI_data/train_ru.tsv}"
TEST_PATH="${TEST_PATH:-./data/multiclass_TACEI_data/test_ru.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments/few_shot_results/$MODEL_THING/}"

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

uv run python ./code/few-shot.py \
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