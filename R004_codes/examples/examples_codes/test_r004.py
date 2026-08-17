#!/usr/bin/env python3

"""
NanoNmD R004 - Inference / Test Script

Input:
    feature.R004.txt or feature.R004.txt.gz

Output:
    predictions.csv
    nm_ratio.csv
    metrics.csv (only when --labels is provided)

The scaler used for normalization MUST be the scaler.pkl
saved during training.
"""

import os
import pickle
import argparse
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn import metrics
from tqdm import tqdm


# ============================================================
# 1. Dataset
# ============================================================

class TestDataset(Dataset):

    KMER_TO_INT = {"A": 0, "G": 1, "C": 2, "T": 3}

    def __init__(self, signal_data, sequence_data, labels=None):
        self.X1 = torch.tensor(signal_data, dtype=torch.float32)
        self.X2 = torch.tensor(sequence_data, dtype=torch.long)
        self.has_labels = labels is not None
        self.y = torch.tensor(np.asarray(labels).flatten(), dtype=torch.long) if labels is not None else torch.zeros(len(self.X1), dtype=torch.long)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.y[index]


# ============================================================
# 2. R004 CNN model
# ============================================================

class CNN_model(nn.Module):

    def __init__(self, kernel_sizes, num_filters, num_classes, d_prob, mode, vacab_size, embedding_dim):
        super(CNN_model, self).__init__()
        self.vocab_size = vacab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding_table = nn.Embedding(vacab_size, int(embedding_dim / 2))
        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters[i], kernel_size=kernel_sizes[i]) for i in range(len(kernel_sizes))])
        self.fc = nn.Sequential(nn.Linear(int(np.sum(num_filters)), 128), nn.ReLU(), nn.Linear(128, num_classes))
        self.dropout = nn.Dropout(p=d_prob)

    def forward(self, x):
        x_sig, x_seq = x
        x_com = None

        for index, start in enumerate(range(0, 20, 4)):
            x_seq0 = self.embedding_table(x_seq[:, index])
            x0 = torch.cat([x_sig[:, start:start + 4], x_seq0], dim=1).unsqueeze(1)
            x_com = x0 if x_com is None else torch.cat([x_com, x0], dim=1)

        x_reshaped = x_com.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]).squeeze(2) for x_conv in x_conv_list]
        x_fc = torch.cat(x_pool_list, dim=1)
        return self.fc(self.dropout(x_fc))


# ============================================================
# 3. Parse R004 feature
# ============================================================

def parse_feature(feature):
    if pd.isna(feature):
        raise ValueError("Feature contains NaN.")

    feature = str(feature).strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")

    if "|" in feature:
        values = feature.split("|")
    elif "," in feature:
        values = feature.split(",")
    else:
        values = feature.split()

    values = [x.strip() for x in values if x.strip() != ""]

    return [float(x) for x in values]


# ============================================================
# 4. Load R004 feature.R004.txt
# ============================================================

def load_r004_feature_file(filepath):

    print(f"\nLoading feature file: {filepath}")

    df = pd.read_csv(filepath, sep="\t", dtype=str)

    print(f"Input samples: {len(df):,}")

    signal_data = []
    sequence_data = []
    metadata = []

    for index, row in df.iterrows():

        try:

            sequence = str(row["Base_Sequence"]).upper().strip()

            if len(sequence) != 5:
                continue

            if any(base not in "ACGT" for base in sequence):
                continue

            features = []

            for column in ["Features1", "Features2", "Features3", "Features4"]:
                values = parse_feature(row[column])

                if len(values) != 5:
                    raise ValueError(f"{column} contains {len(values)} values instead of 5.")

                features.append(values)

            signal = np.asarray(features, dtype=np.float32).T.reshape(-1)

            if signal.shape[0] != 20:
                raise ValueError(f"Signal contains {signal.shape[0]} features instead of 20.")

            signal_data.append(signal)
            sequence_data.append(sequence)

            metadata.append({
                "Read_Name": row["Read_Name"],
                "Ref_Name": row["Ref_Name"],
                "Ref_Coordinate": row["Ref_Coordinate"],
                "Base_Sequence": sequence,
                "Reference_Sequence": row.get("Reference_Sequence", ""),
                "real_ref_seq": row.get("real_ref_seq", "")
            })

        except Exception as e:
            print(f"Skipping row {index}: {e}")

    X_signal = np.asarray(signal_data, dtype=np.float32)
    X_sequence = np.asarray(sequence_data)

    metadata = pd.DataFrame(metadata)

    print(f"Valid samples: {len(X_signal):,}")
    print(f"Signal shape: {X_signal.shape}")

    return X_signal, X_sequence, metadata


# ============================================================
# 5. Encode 5-mer
# ============================================================

def encode_sequences(sequences):

    KMER_TO_INT = {"A": 0, "G": 1, "C": 2, "T": 3}

    encoded = []

    for sequence in sequences:
        sequence = str(sequence).upper()

        if len(sequence) != 5:
            raise ValueError(f"Invalid 5-mer: {sequence}")

        encoded.append([KMER_TO_INT[base] for base in sequence])

    return np.asarray(encoded, dtype=np.int64)


# ============================================================
# 6. Load scaler
# ============================================================

def load_scaler(scaler_path):

    print(f"\nLoading scaler: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("Scaler loaded successfully.")

    if not hasattr(scaler, "mean_"):
        raise ValueError("Loaded scaler does not look like a fitted StandardScaler.")

    print(f"Scaler features: {len(scaler.mean_)}")

    return scaler


# ============================================================
# 7. Load model
# ============================================================

def load_model(model_path, device):

    print(f"\nLoading model: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = CNN_model(kernel_sizes=[2, 3, 4], num_filters=[100, 100, 100], num_classes=2, d_prob=0.2, mode="static", vacab_size=4, embedding_dim=8).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Detected checkpoint dictionary.")
        state_dict = checkpoint["model_state_dict"]
    else:
        print("Detected direct model state_dict.")
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# 8. Prediction
# ============================================================

@torch.no_grad()
def run_inference(model, loader, device, has_labels=False):

    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []

    for data1, data2, target in tqdm(loader, desc="Predicting"):

        data1 = data1.to(device)
        data2 = data2.to(device)

        outputs = model((data1, data2))

        probs = torch.softmax(outputs, dim=1)[:, 1]

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        if has_labels:
            all_targets.extend(target.tolist())

    return all_preds, all_probs, all_targets if has_labels else None


# ============================================================
# 9. Metrics
# ============================================================

def calc_metrics(y_label, y_proba, y_predict):

    cm = metrics.confusion_matrix(y_label, y_predict, labels=[0, 1])

    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    P = TP + FN
    N = TN + FP

    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = TP / (TP + FP) if (TP + FP) > 0 else 0

    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    MCC = (TP * TN - FP * FN) / denominator if denominator != 0 else 0

    try:
        fpr, tpr, _ = metrics.roc_curve(y_label, y_proba)
        AUC = metrics.auc(fpr, tpr)
    except Exception:
        AUC = np.nan

    try:
        f1 = f1_score(y_label, y_predict, average="binary")
    except Exception:
        f1 = np.nan

    try:
        lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_proba)
        PRAUC = metrics.auc(lr_recall, lr_precision)
    except Exception:
        PRAUC = np.nan

    return Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC


# ============================================================
# 10. Calculate per-site Nm ratio
# ============================================================

def calc_nm_ratio(metadata, predictions, probabilities):

    result_df = metadata.copy().reset_index(drop=True)

    result_df["prediction"] = predictions
    result_df["probability_Nm"] = probabilities

    result_df["Ref_Coordinate"] = pd.to_numeric(result_df["Ref_Coordinate"], errors="coerce")

    nm_ratio_df = result_df.groupby(["Ref_Name", "Ref_Coordinate"], sort=False).agg(
        Base_Sequence=("Base_Sequence", "first"),
        total_reads=("prediction", "count"),
        modified_reads=("prediction", "sum"),
        mean_probability=("probability_Nm", "mean")
    ).reset_index()

    nm_ratio_df["nm_ratio"] = nm_ratio_df["modified_reads"] / nm_ratio_df["total_reads"]

    nm_ratio_df = nm_ratio_df[["Ref_Name", "Ref_Coordinate", "Base_Sequence", "total_reads", "modified_reads", "nm_ratio", "mean_probability"]]

    return nm_ratio_df


# ============================================================
# 11. Main test function
# ============================================================

def test_model(input_data_path, model_path, scaler_path, output_path, device, batch_size=2048, labels_path=None):

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------------------------------
    # Load feature
    # --------------------------------------------------------

    print(f"\nLoading R004 feature data: {input_data_path}")

    X_signal, X_sequence_string, metadata = load_r004_feature_file(input_data_path)

    # --------------------------------------------------------
    # Load saved scaler
    # --------------------------------------------------------

    scaler = load_scaler(scaler_path)

    if X_signal.shape[1] != len(scaler.mean_):
        raise ValueError(f"Feature number mismatch: test data has {X_signal.shape[1]} features, but scaler expects {len(scaler.mean_)} features.")

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT fit scaler here.
    # --------------------------------------------------------

    print("\nApplying saved StandardScaler...")

    X_signal = scaler.transform(X_signal).astype(np.float32)

    print(f"Normalized signal shape: {X_signal.shape}")

    # --------------------------------------------------------
    # Encode sequence
    # --------------------------------------------------------

    X_sequence = encode_sequences(X_sequence_string)

    print(f"Sequence shape: {X_sequence.shape}")

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    test_labels = None
    has_labels = False

    if labels_path is not None:

        print(f"\nLoading labels: {labels_path}")

        test_labels = pd.read_csv(labels_path, header=None).iloc[:, 0].values

        if len(test_labels) != len(X_signal):
            raise ValueError(f"Number of labels ({len(test_labels)}) does not match number of samples ({len(X_signal)}).")

        has_labels = True

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = TestDataset(X_signal, X_sequence, test_labels)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=torch.cuda.is_available())

    print(f"\nTest samples: {len(test_dataset):,}")
    print(f"Batch size: {batch_size}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(model_path, device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions, probabilities, targets = run_inference(model, test_loader, device, has_labels)

    # --------------------------------------------------------
    # predictions.csv
    # --------------------------------------------------------

    predictions_df = metadata.copy()

    predictions_df["probability_Nm"] = probabilities
    predictions_df["prediction"] = predictions
    predictions_df["prediction_label"] = predictions_df["prediction"].map({0: "negative", 1: "positive"})

    predictions_csv = os.path.join(output_path, "predictions.csv")

    predictions_df.to_csv(predictions_csv, index=False)

    print(f"\nPredictions saved: {predictions_csv}")

    # --------------------------------------------------------
    # nm_ratio.csv
    # --------------------------------------------------------

    print("\nCalculating per-site Nm ratios...")

    nm_ratio_df = calc_nm_ratio(metadata, predictions, probabilities)

    nm_ratio_csv = os.path.join(output_path, "nm_ratio.csv")

    nm_ratio_df.to_csv(nm_ratio_csv, index=False)

    print(f"Nm ratio saved: {nm_ratio_csv}")
    print(f"Total sites: {len(nm_ratio_df):,}")
    print(f"Sites with Nm ratio > 0.5: {(nm_ratio_df['nm_ratio'] > 0.5).sum():,}")

    # --------------------------------------------------------
    # metrics.csv
    # --------------------------------------------------------

    if has_labels and targets is not None:

        Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC = calc_metrics(targets, probabilities, predictions)

        metrics_dict = {
            "Sensitivity (Sn)": Sn,
            "Specificity (Sp)": Sp,
            "Precision": Pre,
            "F1-score": f1,
            "Accuracy": Acc,
            "MCC": MCC,
            "AUC-ROC": AUC,
            "AUC-PR": PRAUC
        }

        print("\n========================================")
        print("Test Results")
        print("========================================")

        for name, value in metrics_dict.items():
            print(f"{name}: {value:.4f}")

        metrics_csv = os.path.join(output_path, "metrics.csv")

        pd.DataFrame(metrics_dict, index=[0]).to_csv(metrics_csv, index=False)

        print(f"\nMetrics saved: {metrics_csv}")

    else:

        print("\nNo labels provided; metrics.csv will not be generated.")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========================================")
    print("R004 TEST COMPLETE")
    print("========================================")
    print(f"Total reads: {len(predictions):,}")
    print(f"Predicted Nm reads: {np.sum(np.asarray(predictions) == 1):,}")
    print(f"Predicted unmodified reads: {np.sum(np.asarray(predictions) == 0):,}")
    print(f"Nm read ratio: {np.mean(np.asarray(predictions) == 1):.4f}")
    print("========================================")

    return predictions_df, nm_ratio_df


# ============================================================
# 12. Command line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(description="NanoNmD R004 inference and Nm stoichiometry prediction.")

    parser.add_argument("--input", required=True, help="Path to feature.R004.txt or feature.R004.txt.gz.")

    parser.add_argument("--model", required=True, help="Path to trained R004 model .pth.")

    parser.add_argument("--scaler", required=True, help="Path to training scaler.pkl.")

    parser.add_argument("--out", required=True, help="Output directory.")

    parser.add_argument("--batch_size", type=int, default=2048, help="Inference batch size. Default: 2048.")

    parser.add_argument("--labels", default=None, help="Optional single-column labels file for calculating metrics.")

    return parser.parse_args()


# ============================================================
# 13. Main
# ============================================================

def main():

    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"PyTorch {torch.__version__} | Device: {device}")

    test_model(input_data_path=args.input, model_path=args.model, scaler_path=args.scaler, output_path=args.out, device=device, batch_size=args.batch_size, labels_path=args.labels)


if __name__ == "__main__":
    main()