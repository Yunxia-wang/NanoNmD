#!/bin/bash

date
# -----------------------------
# Paths
# -----------------------------
base_root_dir=/your_work_path

PROJECTDIR="${base_root_dir}/Nm_deep_learning/code/github_NanoNmD/NanoNmD"

OUTDIR="${PROJECTDIR}/R004_codes/examples/example_data/01_basecall_out"
REF="${PROJECTDIR}/data/reference/human_uniq.rRNA.fa"
GENERATE_FEATURES="${PROJECTDIR}/R004_codes/examples/examples_codes/generate_features.py"
MOTIF="${PROJECTDIR}/R004_codes/examples/examples_codes/kmer_1024.txt"

# ========================== CONFIGURATION ==========================
# dorado_basecaller could be downloaded in https://doi.org/10.5281/zenodo.21896153

# Tool paths
declare -A programs_array01=(
    [dorado_basecaller]="${base_root_dir}/software/dorado-1.0.0-linux-x64/bin/dorado"
)

# Threads and model
THREADS=20
cd ${PROJECTDIR}/R004_codes/software
unzip rna004_130bps_hac@v5.2.0.zip -d ${PROJECTDIR}/R004_codes/software
modelfile="${PROJECTDIR}/R004_codes/software/rna004_130bps_hac@v5.2.0"
ref_trans="${PROJECTDIR}/data/reference/human_uniq.rRNA.fa"

# Input/output directories
input_dir="${PROJECTDIR}/R004_codes/examples/example_data/00_raw_data"
output_dir="${PROJECTDIR}/R004_codes/examples/example_data/01_basecall_out"
feature_out="${PROJECTDIR}/R004_codes/examples/example_data/04_feature_out"
tsv_out="${PROJECTDIR}/R004_codes/examples/example_data/05_tsv_out"

# Create output directories if not exist
mkdir -p "$output_dir" "$feature_out" "$tsv_out"

# -----------------------------
# Conda environment
# -----------------------------
# source /programs/biogrids.shrc
source ~/.bashrc
conda activate nanonmd

# ============================= PIPELINE =============================
for pod5file in "$input_dir"/*.rRNA.pod5; do
    filename=$(basename "$pod5file")
    sample="${filename%.rRNA.pod5}"

    echo "=== Processing sample: $sample ==="

    # Define file paths
    bam="${output_dir}/${sample}.bam"
    sorted_bam="${output_dir}/${sample}.sorted.bam"
    sam="${output_dir}/${sample}.sorted.sam"

    # ---------- Step 1: Basecalling ----------
    echo "[1] Basecalling with Dorado..."
    "${programs_array01[dorado_basecaller]}" basecaller \
     "$modelfile" \
     "$pod5file" \
     --reference "$REF" \
     --emit-moves \
     --min-qscore 7 \
     --mm2-opts "-w 10 --secondary no -k 14" \
     > "$bam"

    # # ---------- Step 2: Process BAM ----------
    echo "[2] Sorting and indexing BAM..."
    samtools sort -o "$sorted_bam" "$bam"
    samtools index "$sorted_bam"
    samtools view -h -o "$sam" "$sorted_bam"

    # # ---------- Step 6: get the nnaopore feature from bam and pod5 files ----------
    python "$GENERATE_FEATURES" \
        --bam "$sorted_bam" \
        --input "$pod5file" \
        --output "$feature_out" \
        --ref "$REF" \
        --file_type pod5 \
        --seq_type rna \
        --window 2 \
        --motif "$MOTIF" \
        --motif_label 1 \
        --threads 16 \
        --output_tsv "$tsv_out/feature.R004.txt"
done

echo ""
echo "=========================================="
echo "R004 feature generation finished: $(date)"
echo "=========================================="
