#!/bin/bash

# Quit if there are any errors
set -e
cuda_device="2,3"

MODEL="../models/Llama-2-70b-chat-hf"
PROMPT_FORMAT="ino"

for DATASET in "ncbi" "bc5cdr" "CoNLL 2003"  "PolyIE-Sent"
do

  if [ "$DATASET" == "bc5cdr" ]
  then
    N_SAMPLES=1000
  else
    N_SAMPLES=0
  fi

  CUDA_VISIBLE_DEVICES=$cuda_device \
  PYTHONPATH="." \
  python ./tasks/ner_llama.py \
    --dataset "$DATASET" \
    --model_dir "$MODEL" \
    --data_dir "./data/NER/" \
    --report_dir "./report/NER/llama2-70b" \
    --result_dir "./output/NER/llama2-70b" \
    --prompt_format $PROMPT_FORMAT \
    --tasks generation evaluation \
    --entities_to_exclude "else" "Condition" \
    --max_tokens 4096 \
    --n_test_samples $N_SAMPLES

done
