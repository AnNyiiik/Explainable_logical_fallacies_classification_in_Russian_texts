#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-roberta-large}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"
MODEL_PATH="${MODEL_PATH:-./data/saved_models/$MODEL_THING/}"
INPUT_DATA="${INPUT_DATA:-./data/test_en.txt}"
OUTPUT_DATA="${OUTPUT_DATA:-./data/output_en_$MODEL_THING.csv}"

mkdir -p "$MODEL_PATH"

uv run python ./code/main.py -mode prediction \
   -input_new_data_path "$INPUT_DATA" \
   -output_new_data_path "$OUTPUT_DATA" \
   -model_config "$MODEL" \
   -saved_model_path "$MODEL_PATH"