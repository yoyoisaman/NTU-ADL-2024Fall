
python run_swag_no_trainer.py \
  --model_type bert \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --output_dir model/chinese-lert-base_mc \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file data/context.json \
  --gradient_accumulation_steps 2 \
  --tokenizer_name hfl/chinese-lert-base \
  --model_name_or_path hfl/chinese-lert-base \
  --with_tracking \

python run_qa_no_trainer.py \
  --model_type bert \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --output_dir model/chinese-lert-base_qa  \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file data/context.json \
  --gradient_accumulation_steps 2 \
  --with_tracking \
  --per_device_eval_batch_size 8 \
  --tokenizer_name hfl/chinese-lert-base  \
  --model_name_or_path hfl/chinese-lert-base

