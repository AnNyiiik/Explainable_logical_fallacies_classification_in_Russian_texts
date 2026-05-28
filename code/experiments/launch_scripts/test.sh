#!/usr/bin/env bash
set -euxo pipefail

MODEL="${MODEL:-roberta-large}"
MODEL_THING="$(echo "$MODEL" | tr '/' '_')"

SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-./data/saved_models}"
MODEL_PATH="${MODEL_PATH:-$SAVED_MODELS_DIR/$MODEL_THING/}"

DATA_DIR="${DATA_DIR:-./data/multiclass_TACEI_data}"
TEST_TSV="${TEST_TSV:-$DATA_DIR/test_en.tsv}"
INPUT_TXT="${INPUT_TXT:-$DATA_DIR/test_en.txt}"
OUTPUT_CSV="${OUTPUT_CSV:-$DATA_DIR/test_en_pred.csv}"
TEXT_COL="${TEXT_COL:-2}"

echo "Extracting texts from $TEST_TSV..."
cut -f"$TEXT_COL" "$TEST_TSV" | tail -n +2 > "$INPUT_TXT"

uv run python ./code/main.py -mode prediction \
   -input_new_data_path "$INPUT_TXT" \
   -output_new_data_path "$OUTPUT_CSV" \
   -model_config "$MODEL" \
   -saved_model_path "$MODEL_PATH"

echo "Predictions saved to $OUTPUT_CSV"