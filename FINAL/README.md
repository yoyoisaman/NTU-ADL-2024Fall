# 2024 ADL final project 
## Can LLM learn with incoming streams of questions?

資工所碩一 R13922022 王凱右 
資工系大五 B09501048 方群鈞
資工系大四 B10902011 賴詠晴


This guide will help you set up your environment, prepare the datasets, and implement your LLM agent. Please read through all instructions carefully.



## Step 1: Setup Environment

Ensure you have Python 3.10 or above installed on your system. You can check your Python version by running:

```
python --version
$ Python 3.10.12
```

If you need to update Python, visit the [official Python website](https://www.python.org/downloads/).

Next, it's recommended to create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Then, install the required packages:

```
pip install -r requirements.txt
```

## Step 2: Setup Dataset

Run the following command to set up the necessary datasets:

```
python setup_data.py
```

This script will download and prepare all required data for the project. Ensure you have a stable internet connection and sufficient disk space.

## Step 3: Check Your Environment

After setting up the datasets and installing the required packages, it's crucial to verify your environment:

```
python test_env.py
```

You should see each task marked as PASSED when evaluated on ground truth data. For example:

```
Task: sql_generation_public
Metric: EX
Score: 0.9993 (score > 0.99)
Result: PASSED
------------------------------
```

If any task fails, double-check your setup and ensure all datasets are correctly placed.


### Run Our main.py:

Medical Diagnosis: 0.32985
Text-to-SQL Generation: 0.79081
```For SQL
python -m main --use_wandb  --bench_name sql_generation_public --model_name "google/gemma-2-9b-it" --output_path ./output_sql_generation.csv --use_8bit --device cuda:0
```

```For MED
python -m main --use_wandb  --bench_name classification_public --model_name "google/gemma-2-9b-it" --output_path ./output_classification.csv --use_8bit --device cuda:0
```


