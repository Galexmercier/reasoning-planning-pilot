#!/bin/bash

echo "=== Checking GPU ==="
nvidia-smi

echo "=== Downloading Datasets ==="
wget -qnc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train.jsonl
wget -qnc https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train_self-instruct.jsonl
wget -qnc https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
wget -qnc https://raw.githubusercontent.com/mlcommons/ailuminate/main/airr_official_1.0_demo_en_us_prompt_set_release.csv

echo "=== Installing Required Packages ==="
pip install -U datasets trl bitsandbytes transformers accelerate peft optuna

python3 eval_script.py
