"""
NanoNmD - Inference / Test Script
Run a trained NanoNmD model on a single dataset and output:
  - predictions.csv  : per-read Nm probability and binary prediction
  - nm_ratio.csv     : per-site Nm stoichiometry ratio
  - metrics.csv      : performance metrics (only when --labels is provided)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from torch.nn import functional as F
import argparse
import math
import pickle
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ─────────────────────────────────────────────
# Dataset  (labels are optional)
# ─────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Dataset for inference. Labels are optional.

    The last column of the DataFrame is expected to be the 5-mer motif string
    (e.g. 'ACGTA'); all preceding columns are signal features.

    Args:
        data   (pd.DataFrame): feature DataFrame (signal cols + kmer col).
        labels (pd.Series | pd.DataFrame | None): true labels, or None for
                                                   unlabelled inference.
    """

    NM_KMERS    = ['A', 'G', 'C', 'T']
    KMER_TO_INT = {k: i for i, k in enumerate(NM_KMERS)}

    def __init__(self, data: pd.DataFrame, labels=None):
        self.has_labels = labels is not None
        (self.X1, self.X2), self.y = self._build_tensors(data, labels)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.y[index]

    def _build_tensors(self, data, labels):
        signal_data = data.iloc[:, :-1].values
        kmer_motifs = data.iloc[:, -1].tolist()

        kmer_ints = [
            [self.KMER_TO_INT[nt] for nt in str(motif)]
            for motif in kmer_motifs
        ]

        n  = len(signal_data)
        X1 = np.zeros((n, 20), dtype=np.float32)
        X2 = np.zeros((n, 5),  dtype=np.int32)

        for i, row in enumerate(signal_data):
            X1[i] = row[:20]
            X2[i] = kmer_ints[i]

        X1 = torch.from_numpy(X1).float()
        X2 = torch.from_numpy(X2).long()

        if labels is not None:
            y = torch.from_numpy(
                np.array(labels).flatten().astype(np.int32)
            ).long()
        else:
            y = torch.zeros(n, dtype=torch.long)   # dummy

        return (X1, X2), y


# ─────────────────────────────────────────────
# Model  (must match the architecture used during training)
# ─────────────────────────────────────────────

class CNN_model(nn.Module):
    """
    NanoNmD CNN architecture.

    Args:
        kernel_sizes  (list[int]): kernel sizes for parallel Conv1d layers.
        num_filters   (list[int]): number of filters per Conv1d layer.
        num_classes   (int):       number of output classes (2 for binary).
        d_prob        (float):     dropout probability.
        vacab_size    (int):       nucleotide vocabulary size (4 for A/G/C/T).
        embedding_dim (int):       total embedding dim; half used for lookup.
    """

    def __init__(self, kernel_sizes, num_filters, num_classes, d_prob,
                 vacab_size, embedding_dim):
        super().__init__()

        self.embedding_dim   = embedding_dim
        self.embedding_table = nn.Embedding(vacab_size, embedding_dim // 2)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(
                in_channels  = embedding_dim,
                out_channels = num_filters[i],
                kernel_size  = kernel_sizes[i],
            )
            for i in range(len(kernel_sizes))
        ])

        self.fc = nn.Sequential(
            nn.Linear(int(np.sum(num_filters)), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.dropout = nn.Dropout(p=d_prob)

    def forward(self, x):
        x_sig, x_seq = x                          # (B,20), (B,5)

        x_com = None
        for idx, start in enumerate(range(0, 20, 4)):
            emb  = self.embedding_table(x_seq[:, idx])
            feat = x_sig[:, start: start + 4]
            x0   = torch.cat([feat, emb], dim=1).unsqueeze(1)
            x_com = x0 if x_com is None else torch.cat([x_com, x0], dim=1)

        x_reshaped  = x_com.permute(0, 2, 1)
        x_conv_list = [F.relu(conv(x_reshaped)) for conv in self.conv1d_list]
        x_pool_list = [
            F.max_pool1d(xc, kernel_size=xc.shape[2]).squeeze(2)
            for xc in x_conv_list
        ]

        x_fc   = torch.cat(x_pool_list, dim=1)
        logits = self.fc(self.dropout(x_fc))
        return logits


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def calc_metrics(y_label, y_proba, y_predict):
    cm          = metrics.confusion_matrix(y_label, y_predict)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    P, N        = TP + FN, TN + FP

    Sn  = TP / P  if P  > 0 else 0
    Sp  = TN / N  if N  > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = TP / (TP + FP)      if (TP + FP) > 0 else 0

    tmp = (math.sqrt((TP + FP) * (TP + FN))
           * math.sqrt((TN + FP) * (TN + FN)))
    MCC = (TP * TN - FP * FN) / tmp if tmp != 0 else 0

    fpr, tpr, _            = metrics.roc_curve(y_label, y_proba)
    AUC                    = metrics.auc(fpr, tpr)
    f1                     = f1_score(y_label, y_predict, average='binary')
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_proba)
    PRAUC                  = metrics.auc(lr_recall, lr_precision)

    return Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC


# ─────────────────────────────────────────────
# Data helper
# ─────────────────────────────────────────────

def generate_gap(datapath):
    """
    Load the Step-7 feature CSV and split it into three pieces:
      - metadata   : read_pos, CHROM, REF_POS, BASE, seq_mer, FLAG, site_strand
                     (kept only for output columns, never fed to the scaler/model)
      - features   : 20 numeric signal features (4 pipe-separated columns x 5 kmer
                     positions), ready for StandardScaler
      - kmer       : new_seq_mer, the 5-nt motif string used by the model's
                     nucleotide embedding

    NOTE: the raw CSV also carries columns literally named "0", "1", "2", "7" —
    these are non-numeric duplicates of read_pos / BASE / seq_mer, not signal
    data. Only "3", "4", "5", "6" hold the real pipe-separated numeric values
    (this was previously "2"-"6", which included the non-numeric duplicate and
    would fail on .astype(float)).
    """
    df = pd.read_csv(datapath, index_col=0)

    metadata_cols = [
        "read_pos",
        "CHROM",
        "REF_POS",
        "BASE",
        "seq_mer",
        "FLAG",
        "site_strand"
    ]

    feature_cols = ["3", "4", "5", "6"]   # the 4 real pipe-separated numeric columns

    # 展开每个 feature 的 5 个 position
    feature_dict = {}
    for col in feature_cols:
        values = df[col].str.split("|", expand=True).astype(float)

        for pos in range(values.shape[1]):
            feature_dict[f"{col}_pos{pos}"] = values[pos]

    features = pd.DataFrame(feature_dict, index=df.index)

    # position-major order -> 4 features x 5 positions = 20 columns
    ordered_features = [
        f"{feature}_pos{pos}"
        for pos in range(5)
        for feature in feature_cols
    ]

    metadata = df[metadata_cols].reset_index(drop=True)
    features = features[ordered_features].reset_index(drop=True)
    kmer     = df["new_seq_mer"].reset_index(drop=True)

    return metadata, features, kmer

# ─────────────────────────────────────────────
# Nm ratio calculation
# ─────────────────────────────────────────────

def calc_nm_ratio(metadata: pd.DataFrame,
                  predictions: list,
                  probabilities: list) -> pd.DataFrame:
    """
    Aggregate per-read predictions to per-site Nm stoichiometry.

    Groups by (CHROM, REF_POS) — one row per genomic site — and carries
    along the reference BASE and seq_mer context for that site.

    Parameters
    ----------
    metadata      : DataFrame with (at least) CHROM, REF_POS, BASE, seq_mer
                    columns, row-aligned with predictions/probabilities.
    predictions   : List of per-read binary predictions (0 or 1).
    probabilities : List of per-read Nm probabilities (float 0–1).

    Returns
    -------
    pd.DataFrame with columns:
        CHROM, REF_POS, BASE, seq_mer, total_reads, modified_reads,
        nm_ratio, mean_probability
    """
    result_df = metadata[['CHROM', 'REF_POS', 'BASE', 'seq_mer']].copy().reset_index(drop=True)
    result_df['prediction']  = predictions
    result_df['probability'] = probabilities

    nm_ratio_df = (
        result_df
        .groupby(['CHROM', 'REF_POS'], sort=False)
        .agg(
            BASE             = ('BASE', 'first'),
            seq_mer          = ('seq_mer', 'first'),
            total_reads      = ('prediction', 'count'),
            modified_reads   = ('prediction', 'sum'),
            mean_probability = ('probability', 'mean'),
        )
        .reset_index()
    )
    nm_ratio_df['nm_ratio'] = (
        nm_ratio_df['modified_reads'] / nm_ratio_df['total_reads']
    )

    # Reorder columns for clarity
    nm_ratio_df = nm_ratio_df[[
        'CHROM', 'REF_POS', 'BASE', 'seq_mer',
        'total_reads', 'modified_reads', 'nm_ratio', 'mean_probability',
    ]]

    return nm_ratio_df


# ─────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────

def run_inference(model, loader, device, has_labels=False):
    """
    Run the model over all batches in loader.

    Returns
    -------
    preds   : list[int]   – binary predictions
    probs   : list[float] – Nm probabilities
    targets : list[int] | None – true labels if has_labels, else None
    """
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for data1, data2, target in tqdm(loader, desc='Predicting'):
            data1, data2 = data1.to(device), data2.to(device)
            outputs      = model((data1, data2))
            probs        = torch.softmax(outputs, dim=1)[:, 1]
            _, preds     = torch.max(outputs.data, 1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            if has_labels:
                all_targets.extend(target.tolist())

    return all_preds, all_probs, (all_targets if has_labels else None)


# ─────────────────────────────────────────────
# Main test routine
# ─────────────────────────────────────────────

def test_model(input_data_path, model_path, scaler_path, output_path, device,
               batch_size=128, labels_path=None):
    """
    Full inference pipeline:
      1. Load & preprocess the feature CSV.
      2. Load the trained model checkpoint.
      3. Run per-read inference.
      4. Save predictions.csv and nm_ratio.csv.
      5. If labels are provided, save metrics.csv.
    """
    os.makedirs(output_path, exist_ok=True)

    # ── Load & prepare data ──────────────────────────────────────────────
    print(f'Loading data from: {input_data_path}')
    metadata, feature_df, kmer = generate_gap(input_data_path)
    print(f'  Total samples: {feature_df.shape[0]:,}')

    has_labels  = False
    test_labels = None
    if labels_path:
        print(f'Loading true labels from: {labels_path}')
        test_labels = pd.read_csv(labels_path, header=None).iloc[:, 0]
        has_labels  = True

    # ── Load the scaler fitted during training ─────────────────────────────
    print(f'Loading scaler from: {scaler_path}')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print('Scaler loaded successfully.')

    # Apply the training scaler to test data.
    # IMPORTANT: DO NOT call fit() or fit_transform() here.
    X_norm = scaler.transform(feature_df)

    X_norm_df = pd.concat(
        [
            pd.DataFrame(
                X_norm,
                columns=feature_df.columns
            ),
            kmer.rename('kmer')
        ],
        axis=1
    )

    test_dataset = TestDataset(X_norm_df, test_labels)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # ── Load model ───────────────────────────────────────────────────────
    print(f'Loading model from: {model_path}')
    model = CNN_model(
        kernel_sizes  = [2, 3, 4],
        num_filters   = [100, 100, 100],
        num_classes   = 2,
        d_prob        = 0.2,
        vacab_size    = 4,
        embedding_dim = 8,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print('Model loaded successfully.')

    # ── Run inference ────────────────────────────────────────────────────
    preds, probs, targets = run_inference(model, test_loader, device, has_labels)

    # ── Save per-read predictions ────────────────────────────────────────
    out_df = metadata[['read_pos', 'CHROM', 'REF_POS', 'seq_mer']].copy()
    out_df['probability_Nm'] = probs

    pred_csv = os.path.join(output_path, 'predictions.csv')
    out_df.to_csv(pred_csv, index=False)
    print(f'\nPredictions saved  → {pred_csv}')

    # ── Compute & save per-site Nm ratio ─────────────────────────────────
    print('\nCalculating per-site Nm ratios...')
    nm_ratio_df  = calc_nm_ratio(metadata, preds, probs)
    nm_ratio_csv = os.path.join(output_path, 'nm_ratio.csv')
    nm_ratio_df.to_csv(nm_ratio_csv, index=False)
    print(f'Nm ratio saved     → {nm_ratio_csv}')
    print(f'  Total sites      : {len(nm_ratio_df):,}')
    print(f'  Sites Nm ratio > 0.5 : {(nm_ratio_df["nm_ratio"] > 0.5).sum():,}')

    # ── Metrics (labelled data only) ─────────────────────────────────────
    if has_labels and targets is not None:
        Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC = calc_metrics(targets, probs, preds)

        metrics_dict = {
            'Sensitivity (Sn)': Sn,
            'Specificity (Sp)': Sp,
            'Precision':        Pre,
            'F1-score':         f1,
            'Accuracy':         Acc,
            'MCC':              MCC,
            'AUC-ROC':          AUC,
            'AUC-PR':           PRAUC,
        }

        print('\n── Test Results ──────────────────────────────')
        for name, val in metrics_dict.items():
            print(f'  {name:<22} {val:.4f}')
        print('──────────────────────────────────────────────')

        metrics_csv = os.path.join(output_path, 'metrics.csv')
        pd.DataFrame(metrics_dict, index=[0]).to_csv(metrics_csv, index=False)
        print(f'Metrics saved      → {metrics_csv}')


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'NanoNmD Inference: predict Nm modifications and compute '
            'per-site Nm stoichiometry ratios.'
        )
    )
    parser.add_argument('--input',      required=True,
                        help='Path to the preprocessed feature CSV (output of pipeline.sh Step 7)')
    parser.add_argument('--model',      required=True,
                        help='Path to trained model .pth checkpoint')
    parser.add_argument('--scaler',     required=True,
                        help='Path to scaler.pkl saved during training')
    parser.add_argument('--out',        required=True,
                        help='Output directory for predictions, nm_ratio, and metrics')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference (default: 128)')
    parser.add_argument('--labels',     type=str, default=None,
                        help='(Optional) Path to a single-column CSV of true labels '
                             '(1=Nm, 0=unmodified) for computing performance metrics')
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'PyTorch {torch.__version__} | Device: {device}')

    test_model(
        input_data_path = args.input,
        model_path      = args.model,
        scaler_path     = args.scaler,
        output_path     = args.out,
        device          = device,
        batch_size      = args.batch_size,
        labels_path     = args.labels,
    )


if __name__ == '__main__':
    main()