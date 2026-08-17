#!/bin/bash
#SBATCH --job-name=ctrl1rRNA
#SBATCH --partition=bch-compute			# bch-compute,bch-gpu-pe,cbp-compute (-A cbp)
#SBATCH -A cbp 
#SBATCH --output=output_R004_train_%A_%a.txt
#SBATCH --time=150:0:0
#SBATCH --array=1  # 1 is the best option
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

date

source ~/.bashrc
conda activate nanonmd

base_root_dir=/lab-share/Cardio-Chen-e2/Public/Yunxia

root_data_dir=${base_root_dir}/Nm_deep_learning/code/github_NanoNmD/NanoNmD

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



