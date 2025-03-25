# coding=UTF-8


import torch
import json
import transformers
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    PreTrainedTokenizerBase,
    Adafactor,
    get_scheduler,
    HfArgumentParser,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)
from typing import Dict, List, Union, Optional, Sequence
import copy

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pandas as pd
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from datasets import Dataset
from utils import get_bnb_config, get_prompt
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class TrainingArguments:
    output_dir: str = "./ckp"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    learning_rate: float = 7e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.15 
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine_with_restarts"
    max_length : int = 1024
    source_max_len : int = 1024
    target_max_len : int = 256
    gradient_accumulation_steps: int = 8 

def parse_args():
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args

# https://github.com/Bavest/fin-llama/blob/main/qlora.py
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def load_datasets(save_path, train_name, val_name):
    import random
    random.seed(42)
    with open(f"{save_path}/{train_name}.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f) 
        # # train_data = random.sample(train_data, sample_size)
        # # dataset_train = train_data
        # train_data = json.load(f) 
        # sample_size = len(train_data)
        # train_data = random.sample(train_data, sample_size)
        dataset_train = train_data
        # print(f"隨機選取了 {len(dataset_train)} 筆訓練資料")  # 加入print以確認資料筆數
        # dataset_train = process_data(train_data)
    
    with open(f"{save_path}/{val_name}.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
        dataset_val = val_data
        # dataset_val = process_data(val_data)
    
    return dataset_train, dataset_val

def initialize_model(bnb_config, tokenizer_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, padding_side="right", add_eos_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        use_cache=False,
        low_cpu_mem_usage=True
    )   
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def configure_lora():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    args = parse_args()
    
    save_path = "data"
    dataset_train_name = 'train'
    dataset_val_name = 'public_test'
    
    dataset_train, dataset_val = load_datasets(save_path, dataset_train_name, dataset_val_name)
    
    bnb_config = get_bnb_config()
    model, tokenizer = initialize_model(bnb_config, "zake7749/gemma-2-2b-it-chinese-kyara-dpo", "zake7749/gemma-2-2b-it-chinese-kyara-dpo")
    
    config = configure_lora()
    model = get_peft_model(model, config)

    def prepare_data_for_model(data_list, tokenizer, max_length: int = 512):
        
        # 將原始資料轉換為 Dataset 格式
        dataset = Dataset.from_dict({
            "input": [get_prompt(item['instruction']) for item in data_list],
            "output": [item['output'] for item in data_list]
        })


        return dataset

    dataset_train = prepare_data_for_model(dataset_train, tokenizer, args.max_length)
    
    dataset_val = prepare_data_for_model(dataset_val, tokenizer, args.max_length)
    # response_template = "<start_of_turn>model"
    # data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=False,
        predict_with_generate=False,
    )
    train_loader = DataLoader(dataset_train, batch_size=args.per_device_train_batch_size, shuffle=True, num_workers=8, collate_fn=data_collator)
    valid_loader = DataLoader(dataset_val, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=8, collate_fn=data_collator)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay = args.weight_decay,
        scale_parameter=False,
        warmup_init=False,
        relative_step=False
    )
    # num_training_steps = (args.num_train_epochs * len(train_loader)) // args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    def get_scheduler_with_warmup(optimizer, num_training_steps: int, args):
        # 根據 warmup_ratio 計算 warmup_steps
        num_warmup_steps = int(num_training_steps * args.warmup_ratio)
        
        return get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    lr_scheduler = get_scheduler_with_warmup(optimizer, num_training_steps, args)

    max_grad_norm = args.max_grad_norm

    def train(epoch):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, position=0, leave=True)  # 宣告進度條
        global_step = (epoch - 1) * len(train_loader)
        for step, input_datas in enumerate(train_pbar):
            for key in input_datas.keys():
                input_datas[key] = input_datas[key].to(device)
            optimizer.zero_grad()
            
            outputs = model(**input_datas)
            
            loss = outputs.loss
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            train_pbar.set_description(f'Train Epoch {epoch}')
            train_pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            
            train_loss += loss.item()
            global_step += 1

            if global_step > 0 and global_step % 1000 == 0:
                save_checkpoint(epoch, global_step)
                save_scores(epoch, train_loss / (step + 1), None)
        return train_loss / len(train_loader)
    
    def validate(epoch):
        model.eval()
        val_loss = 0
        val_pbar = tqdm(valid_loader, position=0, leave=True)  # 宣告進度條
        
        with torch.no_grad():
            for input_datas in val_pbar:
                for key in input_datas.keys():
                    input_datas[key] = input_datas[key].to(device)
                
                outputs = model(**input_datas)
                
                loss = outputs.loss
                
                val_pbar.set_description(f'Validate Epoch {epoch}')
                val_pbar.set_postfix({'loss': f'{loss.item():.3f}'})
                
                val_loss += loss.item()
        
        return val_loss / len(valid_loader)
    def save_checkpoint(epoch, step):
        # output_dir = os.path.join(args.output_dir, f"train5_checkpoint-epoch-{epoch}-step-{step}")
        output_dir = os.path.join(args.output_dir, f"train5_checkpoint-step-{step}")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, save_embedding_layers=True, safe_serialization=False)
        print(f"Checkpoint saved at {output_dir}")
    def save_scores(epoch, train_loss, val_loss):
        score_file = os.path.join(args.output_dir, "score_train5.json")
        
        if os.path.exists(score_file):
            with open(score_file, 'r') as f:
                scores = json.load(f)
        else:
            scores = {}

        # 修改保存格式以包含step信息
        if val_loss is not None:  # epoch結束時的完整記錄
            scores[f"epoch_{epoch}"] = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
        else:  # 訓練過程中的checkpoint記錄
            scores[f"epoch_{epoch}_step_{global_step}"] = {
                'epoch': epoch,
                'train_loss': train_loss,
            }

        with open(score_file, 'w') as f:
            json.dump(scores, f, indent=4)
    num_epochs = args.num_train_epochs
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train(epoch)
        val_loss = validate(epoch)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}')
        # save_checkpoint(epoch, global_step)
        save_scores(epoch, train_loss, val_loss)

if __name__ == "__main__":
    main()