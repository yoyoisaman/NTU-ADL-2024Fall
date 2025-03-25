python infer.py \
    --test_file data/filtered_public.jsonl \
    --model_name_or_path ./model \
    --source_prefix "summarize: " \
    --text_column maintext \
    --per_device_eval_batch_size 8 \
    --max_target_length 128 \
    --preprocessing_num_workers 4 \
    --output_file test_res.jsonl \
    --num_beams 5