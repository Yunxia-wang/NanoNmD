#!/bin/bash

date

source ~/.bashrc
conda activate nanonmd


# ============================================================
# 0. Basic settings
# ============================================================
root_data_dir=/your_work_path/NanoNmD
cd ${root_data_dir}

ROOT_DIR="$(pwd)/examples/example_rRNA"
SITE_FEATURE_MERGE_DIR="${ROOT_DIR}/04_site_feature"

# ============================================================
# Tranining
# ============================================================

python ${root_data_dir}/train.py \
    --pos  ${SITE_FEATURE_MERGE_DIR}/FAQ00082_91d45c54_1.all_kmer.rRNA.extract.sites.csv.gz \
    --neg  ${SITE_FEATURE_MERGE_DIR}/FAQ00082_91d45c54_0.all_kmer.rRNA.extract.sites.csv.gz \
    --out  ${ROOT_DIR}/results/retrain_demo \
    --epochs 50



# ============================================================
date