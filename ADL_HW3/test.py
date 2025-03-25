# coding=UTF-8

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


def generate_answers(
    model, tokenizer, data, max_length=2048, num_beams=5
):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    results = []

    # Generate responses
    for i in tqdm(range(data_size)):
        input_text = instructions[i]
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]]), 
            inputs['input_ids']
        ], dim=1).cuda()
        
        attention_mask = torch.cat([
            torch.ones((1, 1), dtype=torch.long), 
            inputs['attention_mask']
        ], dim=1).cuda()

        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,  # 可以根據需要調整
                num_beams=num_beams,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "model:" in generated_text:
            generated_text = generated_text.split("model:", 1)[1].strip()
        
        # 儲存結果
        result = {
            "output": generated_text,
            "id": data[i].get("id")  # 如果資料中有id就使用，否則用索引
        }
        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="zake7749/gemma-2-2b-it-chinese-kyara-dpo",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./res.json",
        help="Path to save generation results"
    )

    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()
    
    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        print("找不到模型找不到模型找不到模型找不到模型找不到模型找不到模型找不到模型找不到模型找不到模型找不到模型找不到模型")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    print(f"Loading LoRA from {args.peft_path}")
    model = PeftModel.from_pretrained(model, args.peft_path)
    model = model.cuda()  # 移到GPU

    print(f"Loading test data from {args.test_data_path}")
    with open(args.test_data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    model.eval()
    results = generate_answers(
        model, 
        tokenizer, 
        data
    )

    print(f"Saving results to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Done!")