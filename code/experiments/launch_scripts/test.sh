#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-roberta-large}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"
MODEL_PATH="${MODEL_PATH:-./data/saved_models/$MODEL_THING/}"

TEST_TSV="${TEST_TSV:-./data/test_en.tsv}"
INPUT_TXT="${INPUT_TXT:-./data/test_en.txt}"
OUTPUT_CSV="${OUTPUT_CSV:-./data/test_en.csv}"
TEXT_COL="${TEXT_COL:-2}"

echo "Extracting texts from $TEST_TSV..."
cut -f"$TEXT_COL" "$TEST_TSV" | tail -n +2 > "$INPUT_TXT"

uv run python ./code/main.py -mode prediction \
   -input_new_data_path "$INPUT_TXT" \
   -output_new_data_path "$OUTPUT_CSV" \
   -model_config "$MODEL" \
   -saved_model_path "$MODEL_PATH"

echo "Predictions saved to $OUTPUT_CSV"