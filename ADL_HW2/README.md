r13922022 資工所一 王凱右
# ADL Homework 2 - Natural Language Generation

## Preparation
Download and unzip the pretrained weights:
```bash
bash ./download.sh
```

## Testing the Best Model
1. Execute the testing script:
```bash
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

2. Modify hyperparameters and file path in the `.sh` file as needed.

## Training the Model
1. Execute the training script:
   ```bash
   bash ./train.sh
   ```
2. Modify hyperparameters and file path in the `.sh` file as needed.

## Visualization

Refer to the `plot` folder for visualization resources:

- `score.json`: Shows training results of the google/mt5-small Model over 16 epochs.
- `plot.py`: Python script to generate plots from JSON files.

To generate plots:
```bash
python plot.py
```

## Additional Notes
- Ensure all paths in the scripts are correctly set before running.

