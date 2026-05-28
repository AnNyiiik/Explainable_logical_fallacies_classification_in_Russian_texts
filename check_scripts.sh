#!/usr/bin/env bash
# Dry-check all launch scripts WITHOUT running training.
# Run from the repository root:  bash check_scripts.sh
set -uo pipefail

SCRIPTS_DIR="${SCRIPTS_DIR:-./code/experiments/launch_scripts}"
CODE_DIR="${CODE_DIR:-./code}"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n'  "$*"; }

fail=0

bold "== 1. Bash syntax check (bash -n) =="
for f in "$SCRIPTS_DIR"/*.sh; do
  if bash -n "$f" 2>/dev/null; then
    green "OK    $f"
  else
    red   "FAIL  $f"; bash -n "$f"; fail=1
  fi
done
echo

bold "== 2. Referenced Python entry points exist =="
for py in main.py optuna_tuning.py train_contrastive.py train_binary_classifier.py experiments/few-shot.py; do
  if [ -f "$CODE_DIR/$py" ]; then green "OK    $CODE_DIR/$py"; else red "MISSING  $CODE_DIR/$py"; fail=1; fi
done
echo

bold "== 3. Dry run: print the python command each script would execute =="
# Stub out uv so nothing actually runs; just echo the assembled command.
uv() { echo "    WOULD RUN: $*"; }
export -f uv
for f in "$SCRIPTS_DIR"/*.sh; do
  echo "--- $f ---"
  # run in a subshell; +x to avoid noise, errors are caught
  ( set +x; bash "$f" ) 2>&1 | grep "WOULD RUN" || red "    (no command produced — check the script)"
done
unset -f uv
echo

bold "== 4. argparse smoke test (-h) — requires deps installed =="
echo "Skipped by default (needs the environment). To run manually:"
echo "    uv run python $CODE_DIR/main.py -h"
echo

if [ "$fail" -eq 0 ]; then
  green "All static checks passed."
else
  red "Some checks failed — see above."
  exit 1
fi
