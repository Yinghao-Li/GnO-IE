#!/bin/bash

# Quit if there are any errors
set -e

PROMPT_FORMAT="ino"

for DATASET in "conll04" "NYT" "PolyIE"
do
  PYTHONPATH="." \
  python ./tasks/relation_extr.py \
    --dataset "$DATASET" \
    --data_dir "./data/RE/" \
    --result_dir "./output/RE/gpt-3.5" \
    --report_dir "./report/RE/gpt-3.5" \
    --prompt_format $PROMPT_FORMAT \
    --tasks generation evaluation \
    --save_report
done
