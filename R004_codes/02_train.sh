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

# ============================================================
# 0. Basic settings
# ============================================================
base_root_dir=/lab-share/Cardio-Chen-e2/Public/Yunxia

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



