#!/bin/bash

# Quit if there are any errors
set -e

PROMPT_FORMAT="ino"

for DATASET in "ncbi" "bc5cdr" "CoNLL 2003" "PolyIE"
do

  if [ "$DATASET" == "bc5cdr" ]
  then
    N_SAMPLES=1000
  else
    N_SAMPLES=0
  fi

  PYTHONPATH="." \
  python ./tasks/ner_gpt.py \
    --dataset "$DATASET" \
    --data_dir "./data/NER/" \
    --result_dir "./output/NER/gpt-3.5" \
    --report_dir "./report/NER/gpt-3.5" \
    --prompt_format $PROMPT_FORMAT \
    --tasks generation evaluation \
    --entities_to_exclude "else" "Condition" \
    --n_test_samples $N_SAMPLES

done
