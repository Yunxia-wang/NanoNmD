#!/bin/bash

date

source ~/.bashrc
conda activate nanonmd

root_data_dir=/your_work_path/NanoNmD

# ============================================================
# 0. Basic settings
# ============================================================
cd ${root_data_dir}

ROOT_DIR="$(pwd)/examples/example_rRNA"
SITE_FEATURE_MERGE_DIR="${ROOT_DIR}/04_site_feature"
outDIR=$(pwd)/R004_codes/examples/example_data/07_test_demo

mkdir "${outDIR}"
# ============================================================
# Test
# ============================================================

python ${root_data_dir}/R004_codes/examples/examples_codes/test_r004.py \
    --input $(pwd)/R004_codes/examples/example_data/05_tsv_out/feature.R004.txt \
    --model $(pwd)/R004_codes/checkpoints_r004/Para_0.0001_64_epoch_200model.pth \
    --scaler $(pwd)/R004_codes/checkpoints_r004/scaler.pkl \
    --out ${outDIR}



# ============================================================
date



