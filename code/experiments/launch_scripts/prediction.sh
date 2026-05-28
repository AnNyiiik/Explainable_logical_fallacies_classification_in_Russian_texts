#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-roberta-large}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"

SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-./data/saved_models}"
MODEL_PATH="${MODEL_PATH:-$SAVED_MODELS_DIR/$MODEL_THING/}"

DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
INPUT_DATA="${INPUT_DATA:-$DATA_DIR/test_en.txt}"
OUTPUT_DATA="${OUTPUT_DATA:-$DATA_DIR/output_en_$MODEL_THING.csv}"

mkdir -p "$MODEL_PATH"

uv run python ./code/main.py -mode prediction \
   -input_new_data_path "$INPUT_DATA" \
   -output_new_data_path "$OUTPUT_DATA" \
   -model_config "$MODEL" \
   -saved_model_path "$MODEL_PATH"