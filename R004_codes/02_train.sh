#!/bin/bash

date

source ~/.bashrc
conda activate nanonmd

# ============================================================
# 0. Basic settings
# ============================================================
base_root_dir=/your_work_path

root_data_dir=${base_root_dir}/Nm_deep_learning/code/github_NanoNmD/NanoNmD

cd ${root_data_dir}
# ============================================================
# Test
# ============================================================

python $(pwd)/R004_codes/examples/examples_codes/train_r004.py \
    --pos $(pwd)/R004_codes/examples/example_data/06_train_scale/positive.feature.R004.txt \
    --neg $(pwd)/R004_codes/examples/example_data/06_train_scale/negative.feature.R004.txt \
    --output $(pwd)/R004_codes/examples/example_data/08_train_demo \
    --epochs 10

# ============================================================
date



