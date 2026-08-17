# NanoNmD - RNA004 Pipeline

**NanoNmD** is a deep learning framework for detecting **2′-O-methylation (Nm)** sites in RNA from Oxford Nanopore Technologies (ONT) direct RNA sequencing data.

This repository provides a preprocessing and deep learning workflow specifically configured for the **ONT RNA004 chemistry**, using `.pod5` raw signal files and the **Dorado** basecaller.

---

## Table of Contents

- [Overview](#overview)
- [Requirements & Installation](#requirements--installation)
- [End-to-End Workflow](#end-to-end-workflow)
  - [Step 1 — Preprocessing: raw pod5 → features](#step-1--preprocessing-raw-pod5--features)
  - [Step 2 — Predict Nm sites (Inference)](#step-2--predict-nm-sites-inference)
  - [Step 3 — Train the model (Optional)](#step-3--train-the-model-optional)
- [Directory Layout](#directory-layout)
- [Pre-trained Model & Scaler](#pre-trained-model--scaler)

---

## Overview

The RNA004 NanoNmD workflow takes raw Nanopore `.pod5` files as input, performs basecalling and read mapping, extracts signal features, and runs deep learning inference to predict Nm modifications.

```text
Raw .pod5 files
      │
      ▼
[00_preprocess_raw_data.sh]  ←─ Basecall with Dorado, sort BAM, 
      │                         extract pod5/bam features
      ▼
Feature file (feature.R004.txt)
      │
      ├──▶  [02_train.sh]   (Optional) Train a new model on labelled positive/negative data
      │
      └──▶  [01_test.sh]    Predict Nm sites using a pre-trained model and scaler
                │
                ▼
           Output predictions in target directory
```
---

## Requirements & Installation

Conda Environment

NanoNmD requires the `nanonmd` Conda environment.

If the repository provides the environment file `nanonmd.yml`, create the environment with:

```bash
conda env create -f nanonmd.yml
```

Then activate it:

```bash
conda activate nanonmd
```
---

## End-to-End Workflow

All preprocessing, training, and inference steps are configured to run as **.sh file**.

The workflow consists of three major steps.

---

### Step 1 — Preprocessing: raw pod5 → features

**Script:**

```text
00_preprocess_raw_data.sh
```

This step converts raw RNA004 `.pod5` files into the feature file required by NanoNmD.

The script processes all `.pod5` files located in:

```text
00_raw_data/
```

#### 1. Dorado Basecalling
#### 2. BAM Sorting and Indexing
#### 3. Signal Feature Generation

The extracted features are combined into the final RNA004 feature file:

```text
feature.R004.txt
```

This file is the direct input for NanoNmD model training and inference.

---

### Step 2 — Predict Nm Sites (Inference)

**Script:**

```text
01_test.sh
```

This step performs Nm site prediction using a pre-trained NanoNmD model.

The inference workflow takes:

```text
feature.R004.txt
```

as input. The prediction results will be written to:

```text
07_test_demo/
```

---

### Step 3 — Train the Model (Optional)

**Script:**

```text
02_train.sh
```

This step is **optional**.

You do not need to run this step if your goal is simply to predict Nm sites using the provided pre-trained NanoNmD model.

Run this step if you have ground-truth labelled data and want to:

* train a new NanoNmD model;
* fine-tune the model;
* evaluate model performance on your own labelled dataset.

The training workflow requires separated positive and negative feature files.

For example:

```text
positive.feature.R004.txt
negative.feature.R004.txt
```

The training script uses these labelled datasets to train the NanoNmD model.

Training checkpoints and associated results will be saved to:

```text
08_train_demo/
```

The output includes PyTorch model checkpoint files (`.pth`) and training metrics.

---

# Directory Layout

With the default configuration, the working directory is organized as follows:

```text
examples_data/
│
├── 00_raw_data/
│   └── *.pod5
│
├── 01_basecall_out/
│   ├── *.bam
│   ├── *.sorted.bam
│   ├── *.sorted.bam.bai
│   └── *.sam
│
├── 04_feature_out/
│   └── Intermediate signal-feature files
│
├── 05_tsv_out/
│   └── feature.R004.txt
│
├── 06_train_scale/
│   ├── positive.feature.R004.txt
│   └── negative.feature.R004.txt
│
├── 07_test_demo/
│   └── Inference results
│
└── 08_train_demo/
    ├── *.pth
    └── Training metrics
```

### Input

The primary raw input is:

```text
00_raw_data/
└── *.pod5
```

### Intermediate files

Basecalling and signal-processing intermediate files are stored in:

```text
01_basecall_out/
04_feature_out/
```

### Model input

The final preprocessing output is:

```text
05_tsv_out/
└── feature.R004.txt
```

This is the main feature file used by the NanoNmD RNA004 model.

### Inference output

Prediction results are stored in:

```text
07_test_demo/
```

### Training output

Training checkpoints and metrics are stored in:

```text
08_train_demo/
```

---

# Pre-trained Model & Scaler

The NanoNmD RNA004 inference workflow requires two files:

1. A pre-trained PyTorch model
2. The feature scaler used during model training

These files should be placed in the checkpoint directory specified by `01_test.sh`.

## Model Weights

The pre-trained model is:

```text
Para_0.0001_64_epoch_200model.pth
```

This file contains the trained NanoNmD model parameters.

---

## Feature Scaler

The corresponding scaler is:

```text
scaler.pkl
```

The scaler is used to normalize the signal features using the same transformation applied during model training.

**The scaler should not be re-fitted on the test dataset.**

During inference, `scaler.pkl` should be loaded directly and applied to the input features.

This ensures that the preprocessing of the test data is consistent with the preprocessing used during model training.

---

## Checkpoint Directory

A typical checkpoint directory may therefore contain:

```text
checkpoints/
├── Para_0.0001_64_epoch_200model.pth
└── scaler.pkl
```

Make sure that both files are available before running:

```bash
sbatch 01_test.sh
```

---
