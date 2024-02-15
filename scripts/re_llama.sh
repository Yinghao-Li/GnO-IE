#!/bin/bash

# Quit if there are any errors
set -e
cuda_device="2,3"

MODEL="../models/Llama-2-70b-chat-hf"
PROMPT_FORMAT="ino"

for DATASET in "conll04" "NYT" "PolyIE"
do
  CUDA_VISIBLE_DEVICES=$cuda_device \
  PYTHONPATH="." \
  python re_llama.py \
    --dataset "$DATASET" \
    --model_dir "$MODEL" \
    --data_dir "./data/IE-Datasets/RE/" \
    --result_dir "./output/RE/llama2-70b" \
    --report_dir "./report/RE/llama2-70b" \
    --tasks generation evaluation \
    --max_tokens 4096 \
    --save_report
done
