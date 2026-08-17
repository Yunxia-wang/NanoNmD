#!/bin/bash

date

# ============================================================
# NanoNmD Nanopore RNA preprocessing pipeline
#
# Step 1: Split multi-read FAST5 into single-read FAST5
# Step 2: Basecalling with Guppy
# Step 3: Resquiggle with Tombo
# Step 4: Extract signal features
# Step 5: Mapping with minimap2
# Step 6: BAM processing and sam2tsv
# Step 7: Extract site-level features
# ============================================================

# ============================================================
# 0. USER CONFIGURATION
#
# Only modify this section.
# ============================================================

base_root_dir=/your_work_dir


root_data_dir=${base_root_dir}/Nm_deep_learning/code/github_NanoNmD/NanoNmD

# ============================================================
# 0. Basic settings
# ============================================================
cd ${root_data_dir}

SAMPLE="example_rRNA"
DATA_TYPE="rRNA"

ROOT_DIR="$(pwd)/examples/example_rRNA"

RAW_FAST5_DIR="${ROOT_DIR}/00_raw_fast5"

SPLIT_DIR="${ROOT_DIR}/01_splited_reads_fast5/${SAMPLE}_fast5_splited"

GUPPY_OUT_DIR="${ROOT_DIR}/02_guppy_tombo_resquiggle_extract_data"

BAM_DIR="${ROOT_DIR}/03_bam_signal_data"

SITE_FEATURE_DIR="${ROOT_DIR}/04_site_feature"

SITE_FEATURE_MERGE_DIR="${ROOT_DIR}/04_site_feature_merge"

# ============================================================
# Software
# ont-guppy could be downloaded in https://doi.org/10.5281/zenodo.21896153
# ============================================================

GUPPY_BIN="${base_root_dir}/software/ont-guppy-cpu/bin/guppy_basecaller"

GUPPY_CFG="rna_r9.4.1_70bps_hac.cfg"

EXTRACT_PY="$(pwd)/extract_raw_and_feature_fast_AUCG.py"

POSTPROCESS_PY="$(pwd)/postprocess_sites.py"

# ============================================================
# Reference
# ============================================================

REFERENCE="$(pwd)/data/reference/human_uniq.rRNA.fa"

GENOME_REF="${REFERENCE}"

# ============================================================
# Parameters
# ============================================================

THREADS=40

PARALLEL_JOBS=1


# ============================================================
# FAST5 list
#
# One FAST5 basename per line
# No .fast5 extension
# raw data could be downloaded in https://doi.org/10.5281/zenodo.21896153
# ============================================================
# ----------------------------------------
cd ${root_data_dir}/examples/example_rRNA/00_raw_fast5

unzip FAQ00082_91d45c54_0.fast5.zip -d ${root_data_dir}/examples/example_rRNA/00_raw_fast5
unzip FAQ00082_91d45c54_1.fast5.zip -d ${root_data_dir}/examples/example_rRNA/00_raw_fast5

# ----------------------------------------
cat > ${root_data_dir}/examples/example_rRNA/fast5_filenames.txt << 'EOF'
FAQ00082_91d45c54_0
FAQ00082_91d45c54_1
EOF
# ----------------------------------------
FAST5_LIST="${ROOT_DIR}/fast5_filenames.txt"
# ============================================================
# Create output directories
# ============================================================

mkdir -p "$SPLIT_DIR"
mkdir -p "$GUPPY_OUT_DIR"
mkdir -p "$BAM_DIR"
mkdir -p "$SITE_FEATURE_DIR"
mkdir -p "$SITE_FEATURE_MERGE_DIR"


# ============================================================
# Step 1
# Split multi-read FAST5 into single-read FAST5
# ============================================================

echo "=========================================="
echo "Step 1: Split FAST5"
echo "=========================================="

source ~/.bashrc
conda activate nanonmd

while IFS= read -r SAMPLE_ID
do

    mkdir -p "${SPLIT_DIR}/${SAMPLE_ID}"

    multi_to_single_fast5 \
        -i "${RAW_FAST5_DIR}/${SAMPLE_ID}.fast5" \
        -s "${SPLIT_DIR}/${SAMPLE_ID}" \
        --recursive \
        -t "$THREADS"

done < "$FAST5_LIST"


# ============================================================
# Step 2
# Basecalling with Guppy
# ============================================================

echo "=========================================="
echo "Step 2: Guppy basecalling"
echo "=========================================="

source ~/.bashrc
conda activate nanonmd

while IFS= read -r SAMPLE_ID
do

    GUPPY_DIR="${GUPPY_OUT_DIR}/${SAMPLE_ID}_guppy"

    mkdir -p "$GUPPY_DIR"

    "$GUPPY_BIN" \
        -i "${SPLIT_DIR}/${SAMPLE_ID}/0" \
        -s "$GUPPY_DIR" \
        --num_callers "$PARALLEL_JOBS" \
        --recursive \
        --fast5_out \
        --config "$GUPPY_CFG" \
        --cpu_threads_per_caller "$THREADS"

done < "$FAST5_LIST"


# ============================================================
# Step 3
# Tombo resquiggle
# ============================================================

source ~/.bashrc
conda activate nanonmd

echo "=========================================="
echo "Step 3: Tombo resquiggle"
echo "=========================================="
echo "$REFERENCE"

while IFS= read -r SAMPLE_ID
do

    WORKSPACE="${GUPPY_OUT_DIR}/${SAMPLE_ID}_guppy/workspace/"

    tombo resquiggle \
        --rna \
        --overwrite \
        "$WORKSPACE" \
        "$REFERENCE" \
        --processes "$THREADS" \
        --fit-global-scale \
        --include-event-stdev \
        --failed-reads-filename \
        "${GUPPY_OUT_DIR}/${SAMPLE_ID}.failed.name.txt" \
        2>"${WORKSPACE}/${SAMPLE_ID}.err"

done < "$FAST5_LIST"


# ============================================================
# Step 4
# Extract signal features
# ============================================================

echo "=========================================="
echo "Step 4: Extract signal features"
echo "=========================================="

source ~/.bashrc
conda activate nanonmd

while IFS= read -r SAMPLE_ID
do

    GUPPY_DIR="${GUPPY_OUT_DIR}/${SAMPLE_ID}_guppy"

    LIST_FILE="${GUPPY_DIR}/${SAMPLE_ID}_guppy.list"

    find "${GUPPY_DIR}/workspace/" \
        -name "*.fast5" > "$LIST_FILE"

    python "$EXTRACT_PY" \
        --cpu="$THREADS" \
        --fl="$LIST_FILE" \
        -o "${GUPPY_DIR}/${SAMPLE_ID}_guppy.feature" \
        --clip=10

    if [[ -f "${GUPPY_DIR}/${SAMPLE_ID}_guppy.feature.feature.tsv" ]]
    then
        gzip "${GUPPY_DIR}/${SAMPLE_ID}_guppy.feature.feature.tsv"
    fi

done < "$FAST5_LIST"


# ============================================================
# Step 5
# Mapping with minimap2
# ============================================================

# source ~/.bashrc
# conda activate nanonmd

echo "=========================================="
echo "Step 5: Mapping"
echo "=========================================="

while IFS= read -r SAMPLE_ID
do

    FA_FILE="${GUPPY_OUT_DIR}/${SAMPLE_ID}_guppy/${SAMPLE_ID}_guppy.feature.feature.fa"

    SAMPLE_BAM_DIR="${BAM_DIR}/${SAMPLE_ID}"

    BAM_FILE="${SAMPLE_BAM_DIR}/${SAMPLE_ID}.extract.sort.bam"

    mkdir -p "$SAMPLE_BAM_DIR"

    minimap2 \
        --secondary=no \
        -ax splice \
        -uf \
        -k 14 \
        -t "$THREADS" \
        "$GENOME_REF" \
        "$FA_FILE" \
    | samtools view -@ "$THREADS" -bS - \
    | samtools sort -@ "$THREADS" - \
        > "$BAM_FILE"

done < "$FAST5_LIST"


# # ============================================================
# # Step 6
# # Index BAM + SAM + depth + sam2tsv
# # ============================================================

echo "=========================================="
echo "Step 6: BAM processing"
echo "=========================================="

# # source ~/.bashrc
# # conda activate nanonmd

while IFS= read -r SAMPLE_ID
do

    SAMPLE_BAM_DIR="${BAM_DIR}/${SAMPLE_ID}"

    BAM="${SAMPLE_BAM_DIR}/${SAMPLE_ID}.extract.sort.bam"

    # ----------------------------
    # 6.1 Index BAM
    # ----------------------------

    samtools index "$BAM"


    # ----------------------------
    # 6.2 BAM -> SAM
    # ----------------------------

    samtools view "$BAM" \
        > "${SAMPLE_BAM_DIR}/${SAMPLE_ID}.extract.sam"


    # ----------------------------
    # 6.3 Calculate depth
    # ----------------------------

    samtools depth \
        -d 100000000 \
        "$BAM" \
        > "${SAMPLE_BAM_DIR}/${SAMPLE_ID}.extract.depth"


    # ----------------------------
    # 6.4 BAM -> TSV
    # ----------------------------

    sam2tsv \
        -r "$REFERENCE" \
        "$BAM" \
    | gzip -c \
        > "${SAMPLE_BAM_DIR}/${SAMPLE_ID}.extract.sort.bam.tsv.gz"



done < "$FAST5_LIST"


# # ============================================================
# # Step 7
# # Extract site-level features
# # ============================================================

# # source ~/.bashrc
# # conda activate nanonmd

echo "=========================================="
echo "Step 7: Site feature extraction"
echo "=========================================="

python "$POSTPROCESS_PY" \
    --bam_dir "$BAM_DIR" \
    --split_batch_dir "$GUPPY_OUT_DIR" \
    --site_feature_dir "$SITE_FEATURE_DIR"


# # ============================================================
# # Finished
# # ============================================================

echo "=========================================="
echo "Pipeline finished!"
echo "=========================================="

echo "Split FAST5:"
echo "$SPLIT_DIR"

echo ""

echo "Guppy/Tombo:"
echo "$GUPPY_OUT_DIR"

echo ""

echo "BAM:"
echo "$BAM_DIR"

echo ""

echo "Site features:"
echo "$SITE_FEATURE_DIR"

# ============================================================
# Test
# ============================================================

# # source ~/.bashrc
# # conda activate nanonmd

cd ${SITE_FEATURE_DIR}

{
    zcat "$(ls *all_kmer.rRNA.extract.sites.csv.gz | head -1)"
    for f in *all_kmer.rRNA.extract.sites.csv.gz; do
        zcat "$f" | tail -n +2
    done
} | gzip > ${SITE_FEATURE_MERGE_DIR}/${SAMPLE}.all_kmer.rRNA.extract.sites.csv.gz

# ============================================================
date