# coding=UTF-8

from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    prefix = "你是繁體中文與文言文互相翻譯的人工智能助理，以下是你的翻譯任務"
    text = f"<start_of_turn>{prefix} user:{instruction} model:<end_of_turn>"
    return text
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )


def get_fewshot_prompt(instruction, few_shot_examples=None):
    if few_shot_examples:
        examples_text = ""
        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"範例{i}:\n"
            examples_text += f"user: {example['instruction']}\n"
            examples_text += f"model: {example['output']}\n\n"
        
        return f"<start_of_turn>你是繁體中文與文言文互相翻譯的人工智能助理，首先我會給予你幾個正確翻譯的例子:\n{examples_text}\n 以下是你的翻譯任務 user:{instruction}\n model:<end_of_turn>"
    else:
        return f"<start_of_turn>你是繁體中文與文言文互相翻譯的人工智能助理 {instruction}<end_of_turn>"
