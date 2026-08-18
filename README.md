# NanoNmD

**NanoNmD** is a deep learning framework for detecting **2′-O-methylation (Nm)** sites in RNA from Oxford Nanopore Technology (ONT) direct RNA sequencing data.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [End-to-End Workflow](#end-to-end-workflow)
  - [Step 0 — Prepare your file list](#step-0--prepare-your-file-list)
  - [Step 1–7 — Preprocessing: raw fast5 → model input](#step-17--preprocessing-raw-fast5--model-input)
  - [Step 8 — Train the model (optional)](#step-8--train-the-model-optional)
  - [Step 9 — Predict Nm sites and calculate Nm ratio](#step-9--predict-nm-sites-and-calculate-nm-ratio)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Pre-trained Model](#pre-trained-model)
- [Citation](#citation)
- [License](#license)

---

## Overview

The full NanoNmD workflow takes raw Nanopore `.fast5` files as input and produces per-read Nm predictions and per-site Nm modification ratios as output.

```
Raw fast5 files
      │
      ▼
[pipeline.sh]  ←─ Steps 1–7: split, basecall, resquiggle,
      │             extract signal features, map, index, merge
      ▼
Signal & sequence features per site  (.csv.gz)
      │
      ├──▶  [train.py]   Train a new model on labelled data
      │
      └──▶  [test.py]    Predict Nm sites with a trained model
                │
                ▼
          predictions.csv   (per-read: probability, binary prediction)
          nm_ratio.csv      (per-site: Nm ratio = modified reads / total reads)
          metrics.csv       (optional: Sn, Sp, AUC, MCC, … if labels provided)
```

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/Yunxia-wang/NanoNmD.git
cd NanoNmD

# 2. Create and activate conda environment
# Create NanoNmD environment
conda env create -f environment/nanonmd.yml

```

---

## End-to-End Workflow

### Step 0 — Prepare your file list

Create a plain-text file listing every fast5 batch name (one per line, **no `.fast5` extension**, **no directory path**):

```
batch_001
batch_002
batch_003
```

Save it as `fast5_filenames.txt` in your `ROOT_DIR`. This file drives every step of the pipeline.

---

### Step 1–7 — Preprocessing: raw fast5 → model input

All preprocessing is handled by a single script: **`pipeline.sh`**.

#### Directory layout created by the pipeline

```
ROOT_DIR/
├── 00_raw_fast5/
├── 01_splited_reads_fast5/
│   └── <sample>_fast5_splited/
├── 02_guppy_tombo_resquiggle_extract_data/
│   └── <batch>_guppy/
│       ├── workspace/                      ← resquiggled fast5 files
│       ├── <batch>_guppy.list
│       └── <batch>_guppy.feature.feature.tsv.gz   ← signal features (Step 4)
├── 03_bam_signal_data/
│   └── <batch>/
│       ├── <batch>.extract.sort.bam
│       ├── <batch>.extract.sort.bam.bai
│       ├── <batch>.extract.sort.bam.tsv.gz
│       ├── <batch>.extract.sam
│       └── <batch>.extract.depth
└── 04_site_feature/
    └── <sample>.all_kmer.rRNA.extract.sites.csv.gz   ← MODEL INPUT
```

The file in `04_site_feature/` is the direct input to `train.py` and `test.py`.

---

### Step 8 — Train the model (optional)

Skip this step if you are using the provided pre-trained weights.

To train on your own labelled data, prepare two separate CSV files (see [Input Data Format](#input-data-format)) — one for Nm-positive sites and one for Nm-negative (IVT / unmodified) control sites.

```bash
python train.py \
    --pos  data/positive_sites.csv \
    --neg  data/negative_sites.csv \
    --out  results/my_run \
    --epochs 200
```
Select the best checkpoint by comparing the `AUC` or `MCC` column across results CSVs.

---

### Step 9 — Predict Nm sites and calculate Nm ratio

`test.py` loads the feature file from Step 7, runs inference with a trained model, and outputs per-read predictions together with per-site Nm ratios.

#### Prediction only (no ground-truth labels)

```bash
python test.py \
    --input  04_site_feature/c42_rRNA_ctrl_1.all_kmer.rRNA.extract.sites.csv.gz \
    --model  checkpoints/Para_0.0001_128_epoch_20model.pth \
    --scaler checkpoints/scaler.pkl \
    --out    results/c42_rRNA_ctrl_1_predictions
```

#### Prediction + performance metrics (with ground-truth labels)

Provide a single-column CSV of true labels (1 = Nm, 0 = unmodified) in the same row order as the input feature file:

```bash
python test.py \
    --input   04_site_feature/c42_rRNA_ctrl_1.all_kmer.rRNA.extract.sites.csv.gz \
    --model   checkpoints/Para_0.0001_128_epoch_20model.pth \
    --out     results/c42_rRNA_ctrl_1_predictions \
    --labels  data/true_labels.csv
```

---

## Input Data Format

The feature CSV produced by `pipeline.sh` Step 7 (direct input to `train.py` and `test.py`):

| Columns | Content |
|---|---|
| Column 0 | Row index |
| Columns 1–20 | 20 signal features: 4 statistics × 5 nucleotide positions |
| Column 21 (last) | 5-mer nucleotide motif string, e.g. `ACGTA` |

Only A, G, C, T are supported in the motif column.

The labels file (for training or evaluation) is a single-column CSV with **no header**, one integer per row (1 = Nm, 0 = unmodified), in the same row order as the feature file.

---

## Output Files

### Preprocessing (`pipeline.sh`)

| File | Location | Description |
|---|---|---|
| `<sample>.all_kmer.rRNA.extract.sites.csv.gz` | `04_site_feature/` | Signal & sequence features per read per site |

### Training (`train.py`)

| File | Description |
|---|---|
| `Para_<lr>_<bs>_epoch_<N>model.pth` | Model checkpoint at epoch N |
| `results_lr<lr>_bs<bs>.csv` | Per-epoch metrics: loss, Sn, Sp, Pre, F1, AUC, PRAUC, MCC |
| `acc_*.png` / `loss_*.png` | Training and validation curves |

### Inference (`test.py`)

| File | Description |
|---|---|
| `predictions.csv` | Per-read: `true_label` (if provided), `probability_Nm`, `prediction` |
| `nm_ratio.csv` | Per-site: `site_strand`, `total_reads`, `modified_reads`, `nm_ratio`, `mean_probability` |
| `metrics.csv` | Sn, Sp, Pre, F1, Acc, MCC, AUC-ROC, AUC-PR (only when `--labels` is provided) |

---

## Pre-trained Model

A model pre-trained on human rRNA Nanopore reads is available in `checkpoints/`:

```bash
python test.py \
    --input  04_site_feature/your_sample.all_kmer.rRNA.extract.sites.csv.gz \
    --model  checkpoints/Para_0.0001_128_epoch_20model.pth \
    --scaler checkpoints/scaler.pkl \
    --out    results/your_sample_predictions
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
