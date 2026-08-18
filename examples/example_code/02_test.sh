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
SITE_FEATURE_MERGE_DIR="${ROOT_DIR}/04_site_feature_merge"

# ============================================================
# Test
# ============================================================

python ${root_data_dir}/test.py \
    --input ${SITE_FEATURE_MERGE_DIR}/example_rRNA.all_kmer.rRNA.extract.sites.csv.gz \
    --model ${root_data_dir}/checkpoints/Para_0.0001_64_epoch_140model.pth \
    --scaler ${root_data_dir}/checkpoints/scaler.pkl \
    --out   ${root_data_dir}/examples/example_rRNA/results/predictions

# ============================================================
date