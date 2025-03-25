#!/bin/bash
python interface_mc.py \
    --model_path model/chinese-lert-base_mc/ \
    --test_file $2 \
    --context_file $1 \
    --output_file test_mc_out.json \
    --max_seq_length 512 \

python interface_qa.py \
    --model_path model/chinese-lert-base_qa \
    --test_file test_mc_out.json \
    --context_file $1 \
    --output_file $3 \
    --max_seq_length 512 
