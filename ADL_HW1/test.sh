#!/bin/bash
python interface_mc.py \
  --model_path model/chinese-lert-base_mc/ \
  --max_seq_length 512 \
  --test_file data/test.json \
  --context_file data/context.json \
  --output_file test_mc_out.json \

python interface_qa.py \
  --model_path model/chinese-lert-base_qa \
  --max_seq_length 512 \
  --test_file test_mc_out.json \
  --context_file data/context.json \
  --output_file predicition.csv \