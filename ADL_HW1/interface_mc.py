# -*- coding: utf-8 -*-
import json
import argparse
from itertools import chain
from dataclasses import dataclass
from typing import Optional, Union
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase
)
from transformers.utils import PaddingStrategy

from datasets import load_dataset
from accelerate import Accelerator
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with a trained model on the SWAG dataset.")
    parser.add_argument(
        "--context_file",
        type=str,
        default="data/context.json",
        help="Path to the context file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="path/to/your/model",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test.json",
        help="Path to the input file for inference.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test/test_mc.json",
        help="Path to the output file for saving predictions.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()
    return args

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    model = AutoModelForMultipleChoice.from_pretrained(args.model_path)
    accelerator = Accelerator()
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    padding = False

    
    def preprocess_function(examples):
    
        # 抓context
        with open(args.context_file, 'r',  encoding='utf-8') as f:
            context = json.load(f)
        
        # 上文x4 -> 問題x4 question
        first_sentences = [[question] * 4 for question in examples['question']]
        
        # question_headers = examples[question_header_name]
        
        # 選項 自己有4個
        second_sentences = []
        # paragraphs -> idx  2018,6952,8264,836
        # idx-> context_file 
        for idx in examples['paragraphs']:
            cand = []
            for i in idx:
                cand.append(context[i])
            second_sentences.append(cand)
    

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        labels = []
        for i in range(len(examples['id'])):
            labels.append(0)
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
        )
    test_dataset = processed_datasets["test"]
    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None
        data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)
    
    model.eval()
    # raw_datasets = load_dataset('json', data_files={'test': args.input_file})
    
    # processed_datasets = raw_datasets.map(
    #     lambda examples: preprocess_function(examples, context, tokenizer, args.max_seq_length, padding='max_length'),
    #     batched=True,
    #     remove_columns=raw_datasets["test"].column_names
    # )
    device = accelerator.device
    # print(device)
    model = model.to(device)
    print("==================================")
    print("Start    TEST")
    print("==================================")
    predictions = []
    result = torch.zeros((0),dtype=int)
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    for batch in tqdm(test_dataloader):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        # 選項
        result = torch.cat((result, predictions.cpu()))

    result = result.numpy()
    with open(args.test_file, 'r', encoding='utf-8') as f:
        res_json = json.load(f)
    # print("=========")
    # print(res_json)
    # print("=========")
    # print(result)
    # print("=========")
    for idx in range(len(result)):
        res_json[idx]['relevant'] = int(res_json[idx]['paragraphs'][int(result[idx])])
    # print(res_json)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(res_json, f, ensure_ascii=False)
    

    
if __name__ == "__main__":
    main()