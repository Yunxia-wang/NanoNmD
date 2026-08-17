#!/bin/bash

date
# ------------------------------------------------

git clone https://github.com/Yunxia-wang/NanoNmD.git
cd NanoNmD
conda env create -f nanonmd.yml
conda activate nanonmd



# The following is some key packages in this env
# ------------------------------------------------
# source ~/.bashrc
# conda activate NanoNmD
# ------------------------------------------------
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# ------------------------------------------------
# pip install numpy pandas scikit-learn matplotlib tqdm einops
# ------------------------------------------------
# conda install -c bioconda ont-fast5-api minimap2 samtools
# conda install hcc::jvarkit-sam2tsv
# ------------------------------------------------

# conda install -c bioconda ont-tombo #install tombo 
# ------------------------------------------------
# source ~/.bashrc
# conda activate NanoNmD
# pip install statsmodels

#  ------------------------------------------------
# source ~/.bashrc
# conda activate nanonmd
# conda install -c bioconda samtools
# conda install -c bioconda ont-fast5-api
# pip install pod5

# ------------------------------------------------
date