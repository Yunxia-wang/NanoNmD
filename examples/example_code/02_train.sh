#!/bin/bash

date

source ~/.bashrc
conda activate nanonmd

base_root_dir=/your_work_path
# ============================================================
# 0. Basic settings
# ============================================================
root_data_dir=${base_root_dir}/Nm_deep_learning/code/github_NanoNmD/NanoNmD
cd ${root_data_dir}

ROOT_DIR="$(pwd)/examples/example_rRNA"
SITE_FEATURE_MERGE_DIR="${ROOT_DIR}/04_site_feature"

# ============================================================
# Test
# ============================================================

python ${root_data_dir}/train.py \
    --pos  ${SITE_FEATURE_MERGE_DIR}/FAQ00082_91d45c54_1.all_kmer.rRNA.extract.sites.csv.gz \
    --neg  ${SITE_FEATURE_MERGE_DIR}/FAQ00082_91d45c54_0.all_kmer.rRNA.extract.sites.csv.gz \
    --out  ${ROOT_DIR}/results/retrain_demo \
    --epochs 50



# ============================================================
date