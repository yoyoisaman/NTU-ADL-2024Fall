

r13922022 資工所一 王凱右
# ADL Homework 1 - Chinese Question Answering

## Environment
- GPU: NVIDIA GeForce RTX 2060 (Laptop)

## Preparation
Download the pretrained weights:
```bash
bash ./download.sh
```

## Testing the Best Model
Run the following command:
```bash
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

## Training the Model
1. Execute the training script:
   ```bash
   bash ./train.sh
   ```
2. Modify hyperparameters and file path in the `.sh` file as needed.

## Visualization

Refer to the `plot` folder for visualization resources:

- `pretrained.json`: Shows training results of the hfl/chinese-roberta-wwm-ext QA Model over 5 epochs.
- `scratch.json`: Shows training results of the bert-base-chinese QA Model (trained from scratch) over 5 epochs.
- `plot.py`: Python script to generate plots from JSON files. Modify the code to change input files or plot parameters.

To generate plots:
```bash
python plot.py
```

## Additional Notes
- Ensure all paths in the scripts are correctly set before running.
- For any modifications to the plotting process, edit the `plot.py` file directly.
