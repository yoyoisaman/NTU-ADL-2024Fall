r13922022 資工所一 王凱右
# ADL Homework 3 - Instruction Tuning (Classical Chinese)

## Preparation
1. Download and unzip the pretrained Gemma2 LoRA weights:
```bash
bash ./download.sh
```

(Optional)If you want to test Llama model
2. Download and unzip the pretrained Llama LoRA weights:
```bash
bash ./download_llama.sh
```

## Setup the environments

Python 3.10.14

```bash
pip install -r requirements.txt
```

## Testing the Best Model

1. Modify hyperparameters and file path in the `.sh` file as needed.


2. Execute the testing script:
```bash
bash run.sh /path/to/model-folder /path/to/adapter_checkpoint /path/to/input /path/to/output
```
Gemma2 Example:
```bash
bash run.sh zake7749/gemma-2-2b-it-chinese-kyara-dpo ./adapter_checkpoint ./data/private_test.json ./prediction.json
```
Llama3 Example:
```bash
bash run.sh yentinglin/Llama-3-Taiwan-8B-Instruct ./llama_adapter_checkpoint ./data/private_test.json ./llama_prediction.json 
```

## Testing the Perplexity of Model

```bash
bash eval.sh /path/to/model-folder /path/to/adapter_checkpoint /path/to/input
```
Gemma2 Example:
```bash
bash eval.sh zake7749/gemma-2-2b-it-chinese-kyara-dpo ./adapter_checkpoint ./data/public_test.json 
```
Llama3 Example:
```bash
bash eval.sh yentinglin/Llama-3-Taiwan-8B-Instruct ./llama_adapter_checkpoint ./data/public_test.json 
```
## Training the Model
1. Modify hyperparameters, model type and file path in the `train.py` file (for Gemma2) and `train_llama.py` file (for Llama3) as needed.

2. Execute the training script:
   ```bash
   python train.py
   ```
   ```bash
   python train_llama.py
   ```
## Visualization

Refer to the `plot` folder for visualization resources:

- `plot_ppl.py`: Python script that plots the relationship between model training steps and perplexity (PPL) values.

To generate plots:
```bash
python plot_ppl.py
```

## Additional Notes
- Ensure all paths in the scripts are correctly set before running.
