#!/bin/bash
#SBATCH --job-name=ctrl1rRNA
#SBATCH --partition=bch-compute			# bch-compute,bch-gpu-pe,cbp-compute (-A cbp)
#SBATCH -A cbp 
#SBATCH --output=output_ctrl1rRNA_%A_%a.txt
#SBATCH --time=150:0:0
#SBATCH --array=1  # 1 is the best option
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

date

source ~/.bashrc
conda activate nanonmd

base_root_dir=/your_work_path

root_data_dir=${base_root_dir}/Nm_deep_learning/code/github_NanoNmD/NanoNmD

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