"""
NanoNmD - Training Script
CNN-based model for Nm (2'-O-methylation) detection from Nanopore signal data.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math
import argparse
import random
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pickle

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def seed_torch(seed=1029):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class CustomDataset(Dataset):
    """
    Dataset for NanoNmD.

    Each sample contains:
        X1 : float tensor of shape (20,)  – normalised signal features
        X2 : long  tensor of shape (5,)   – k-mer nucleotide indices
        y  : int label (1 = Nm, 0 = unmodified)

    The last column of the input DataFrame is expected to be the 5-mer motif
    string (e.g. 'ACGTA'); all preceding columns are signal features.
    """

    NM_KMERS = ['A', 'G', 'C', 'T']
    KMER_TO_INT = {k: i for i, k in enumerate(NM_KMERS)}

    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
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

        n = len(signal_data)
        X1 = np.zeros((n, 20), dtype=np.float32)
        X2 = np.zeros((n, 5),  dtype=np.int32)
        y  = labels.values.flatten().astype(np.int32)

        for i, row in enumerate(signal_data):
            X1[i] = row[:20]
            X2[i] = kmer_ints[i]

        X1 = torch.from_numpy(X1).float()
        X2 = torch.from_numpy(X2).long()
        y  = torch.from_numpy(y).long()
        return (X1, X2), y


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class CNN_model(nn.Module):
    """
    NanoNmD CNN architecture.

    For each of the 5 nucleotide positions:
      - Look up an embedding for the nucleotide (dim = embedding_dim/2 = 4)
      - Concatenate with the 4 signal features at that position
    This forms a (5 × embedding_dim) sequence fed to parallel 1-D convolutions
    with different kernel sizes, followed by max-pooling and a fully-connected
    classifier head.

    Args:
        kernel_sizes  (list[int]): kernel sizes for parallel Conv1d layers.
        num_filters   (list[int]): number of filters per Conv1d layer.
        num_classes   (int):       number of output classes (2 for binary).
        d_prob        (float):     dropout probability.
        vacab_size    (int):       nucleotide vocabulary size (4 for A/G/C/T).
        embedding_dim (int):       total embedding dim; half used for lookup.
    """

    def __init__(
        self,
        kernel_sizes,
        num_filters,
        num_classes,
        d_prob,
        vacab_size,
        embedding_dim,
    ):
        super().__init__()

        self.kernel_sizes  = kernel_sizes
        self.num_filters   = num_filters
        self.num_classes   = num_classes
        self.d_prob        = d_prob
        self.vocab_size    = vacab_size
        self.embedding_dim = embedding_dim

        self.embedding_table = nn.Embedding(vacab_size, embedding_dim // 2)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters[i],
                kernel_size=kernel_sizes[i],
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
        x_sig, x_seq = x                      # (B,20), (B,5)

        x_com = None
        for idx, start in enumerate(range(0, 20, 4)):
            emb  = self.embedding_table(x_seq[:, idx])           # (B, emb/2)
            feat = x_sig[:, start: start + 4]                    # (B, 4)
            x0   = torch.cat([feat, emb], dim=1)                 # (B, emb)
            x0   = x0.unsqueeze(1)                               # (B, 1, emb)
            x_com = x0 if x_com is None else torch.cat([x_com, x0], dim=1)
        # x_com: (B, 5, emb)

        x_reshaped = x_com.permute(0, 2, 1)                      # (B, emb, 5)

        x_conv_list = [F.relu(conv(x_reshaped)) for conv in self.conv1d_list]
        x_pool_list = [
            F.max_pool1d(xc, kernel_size=xc.shape[2]).squeeze(2)
            for xc in x_conv_list
        ]

        x_fc   = torch.cat(x_pool_list, dim=1)                   # (B, sum_filters)
        logits = self.fc(self.dropout(x_fc))
        return logits


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def calc_metrics(y_label, y_proba, y_predict):
    cm  = metrics.confusion_matrix(y_label, y_predict)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    P, N = TP + FN, TN + FP

    Sn  = TP / P if P > 0 else 0
    Sp  = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = TP / (TP + FP) if (TP + FP) > 0 else 0

    tmp = (
        math.sqrt((TP + FP) * (TP + FN))
        * math.sqrt((TN + FP) * (TN + FN))
    )
    MCC = (TP * TN - FP * FN) / tmp if tmp != 0 else 0

    fpr, tpr, _ = metrics.roc_curve(y_label, y_proba)
    AUC          = metrics.auc(fpr, tpr)
    f1           = f1_score(y_label, y_predict, average='binary')

    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_proba)
    PRAUC = metrics.auc(lr_recall, lr_precision)

    return Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC


# ─────────────────────────────────────────────
# Schedulers / Early Stopping
# ─────────────────────────────────────────────

class LRScheduler:
    """Reduce LR on validation loss plateau."""

    def __init__(self, optimizer, patience=10, min_lr=1e-6, factor=0.5):
        self.optimizer    = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience,
            factor=factor, min_lr=min_lr,
        )

    def __call__(self, val_loss):
        prev_lr = self.optimizer.param_groups[0]['lr']
        self.lr_scheduler.step(val_loss)
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f'  INFO: reducing learning rate to {new_lr:.6g}')


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=20, min_delta=0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping triggered.')
                self.early_stop = True


# ─────────────────────────────────────────────
# Training / Validation loops
# ─────────────────────────────────────────────

def fit(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss, running_correct, total, counter = 0.0, 0, 0, 0

    for data1, data2, target in tqdm(train_loader, desc='  Train'):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model((data1, data2))
        loss    = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item()
        _, preds         = torch.max(outputs.data, 1)
        running_correct += (preds == target).sum().item()
        total           += target.size(0)
        counter         += 1

    return running_loss / counter, running_correct / total


def validate(model, valid_loader, loss_fn, device):
    model.eval()
    running_loss, running_correct, total, counter = 0.0, 0, 0, 0
    predicts, targets, pre_probs = [], [], []

    with torch.no_grad():
        for data1, data2, target in tqdm(valid_loader, desc='  Valid'):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            outputs  = model((data1, data2))
            loss     = loss_fn(outputs, target)

            running_loss    += loss.item()
            probs            = torch.softmax(outputs, dim=1)[:, 1]
            _, preds         = torch.max(outputs.data, 1)
            running_correct += (preds == target).sum().item()
            total           += target.size(0)
            counter         += 1

            pre_probs.extend(probs.cpu().tolist())
            targets.extend(target.cpu().tolist())
            predicts.extend(preds.cpu().tolist())

    val_loss = running_loss / counter
    Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC = calc_metrics(targets, pre_probs, predicts)
    return val_loss, Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def generate_gap(datapath):
    """
    Load a Step-7 feature CSV (positive or negative) and build the 20 signal
    features + kmer motif column expected by CustomDataset.

    The raw CSV columns are: read_pos, CHROM, REF_POS, BASE, seq_mer, FLAG,
    site_strand, "0", "1", "2", "3", "4", "5", "6", "7", new_seq_mer.
    Only "3", "4", "5", "6" are real pipe-separated numeric feature columns
    (4 features x 5 kmer positions = 20 columns); "0", "1", "2", "7" are
    non-numeric duplicates of read_pos/BASE/seq_mer and are ignored — same
    schema, and same fix, as test.py's generate_gap().

    The previous version of this function (`data[:, i:20:5]` column-stride
    slicing) assumed the CSV had no metadata columns at all — 20 raw numeric
    columns immediately followed by the kmer column. That doesn't match this
    pipeline's actual output and silently picked up metadata/text columns
    instead of signal features.
    """
    df = pd.read_csv(datapath, index_col=0)

    feature_cols = ["3", "4", "5", "6"]   # the 4 real pipe-separated numeric columns

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

    features = features[ordered_features].reset_index(drop=True)
    kmer     = df["new_seq_mer"].reset_index(drop=True)

    result = pd.concat([features, kmer.rename('kmer')], axis=1)
    return result


def split_data(df, validation_split=0.1, random_seed=666):
    n       = df.shape[0]
    indices = list(range(n))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split   = int(np.floor(validation_split * n))
    return df.iloc[indices[split:]], df.iloc[indices[:split]]


def generate_label(pos_data, neg_data):
    data   = np.concatenate([pos_data, neg_data], axis=0)
    labels = [1] * len(pos_data) + [0] * len(neg_data)
    return pd.DataFrame(data), pd.DataFrame(labels)


def check_dataset(dataset, num_class=2):
    loader     = DataLoader(dataset, batch_size=512, shuffle=False)
    class_nums = [0] * num_class
    for _, _, y in loader:
        for label in y:
            class_nums[label.item()] += 1
    for i, n in enumerate(class_nums):
        print(f"  Class {i}: {n} samples")


def make_plot(train_acc, val_acc, train_loss, val_loss, output_path, tag):
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc,  color='green', label='Train accuracy')
    plt.plot(val_acc,    color='blue',  label='Validation accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(os.path.join(output_path, f'acc_{tag}.png'))
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='Train loss')
    plt.plot(val_loss,   color='red',    label='Validation loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(output_path, f'loss_{tag}.png'))
    plt.close()


# ─────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────

def train_model(pos_data_path, neg_data_path, output_path, device, epochs,
                oversample_factor=3):
    os.makedirs(output_path, exist_ok=True)

    # ── Load & reorder features ──────────────────────────────────────────
    print('Loading data...')
    human_data_pos = generate_gap(pos_data_path)
    human_data_neg = generate_gap(neg_data_path)
    print(f'  Positive samples: {human_data_pos.shape[0]}')
    print(f'  Negative samples: {human_data_neg.shape[0]}')

    # ── Train / validation split ─────────────────────────────────────────
    pos_train, pos_val = split_data(human_data_pos)
    neg_train, neg_val = split_data(human_data_neg)

    # Oversample positives in training set to reduce class imbalance
    pos_train_arr = np.repeat(pos_train.values, oversample_factor, axis=0)

    train_data, label_train = generate_label(pos_train_arr, neg_train.values)
    valid_data, label_valid = generate_label(pos_val.values, neg_val.values)
    print(f'  Train size: {len(train_data)}, Validation size: {len(valid_data)}')

    # ── Normalise signal features (exclude last kmer column) ─────────────
    scaler = StandardScaler()

    X_tr_norm = scaler.fit_transform(train_data.iloc[:, :-1])
    X_va_norm = scaler.transform(valid_data.iloc[:, :-1])

    # Save scaler fitted on training data
    scaler_path = os.path.join(output_path, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'  Scaler saved → {scaler_path}')

    X_tr_norm = pd.concat([pd.DataFrame(X_tr_norm), train_data.iloc[:, -1].reset_index(drop=True)], axis=1)
    X_va_norm = pd.concat([pd.DataFrame(X_va_norm), valid_data.iloc[:, -1].reset_index(drop=True)], axis=1)

    # ── Build datasets ───────────────────────────────────────────────────
    train_dataset = CustomDataset(X_tr_norm, label_train)
    valid_dataset = CustomDataset(X_va_norm, label_valid)
    print('Class distribution — Train:')
    check_dataset(train_dataset)
    print('Class distribution — Validation:')
    check_dataset(valid_dataset)

    # ── Hyperparameter grid search ───────────────────────────────────────
    learning_rates = [0.0001, 0.001]
    batch_sizes    = [64, 32, 128]
    param_grid     = [[lr, bs] for lr in learning_rates for bs in batch_sizes]

    for param in param_grid:
        lr, bs     = param
        tag        = f'lr{lr}_bs{bs}'
        print(f'\n{"="*60}')
        print(f'Hyperparameters: lr={lr}, batch_size={bs}')
        print('='*60)

        model = CNN_model(
            kernel_sizes  = [2, 3, 4],
            num_filters   = [100, 100, 100],
            num_classes   = 2,
            d_prob        = 0.2,
            vacab_size    = 4,
            embedding_dim = 8,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Total parameters: {n_params:,}')

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                  drop_last=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)

        optimizer     = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn       = nn.CrossEntropyLoss()
        lr_scheduler  = LRScheduler(optimizer)
        early_stop    = EarlyStopping()

        results = {}
        start   = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            print(f'\nEpoch {epoch}/{epochs}')

            tr_loss, tr_acc = fit(model, train_loader, optimizer, loss_fn, device)
            va_loss, Sn, Sp, Pre, f1, Acc, MCC, AUC, PRAUC = validate(
                model, valid_loader, loss_fn, device
            )

            lr_scheduler(va_loss)
            early_stop(va_loss)

            elapsed = (time.time() - t0) / 60
            print(f'  Train  — loss: {tr_loss:.4f}  acc: {tr_acc:.4f}')
            print(f'  Valid  — loss: {va_loss:.4f}  Acc: {Acc:.4f}  '
                  f'AUC: {AUC:.4f}  F1: {f1:.4f}  MCC: {MCC:.4f}')

            results[epoch] = [tr_loss, tr_acc, va_loss, Sn, Sp, Pre, f1,
                               Acc, MCC, AUC, PRAUC, elapsed, param]

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                ckpt_path = os.path.join(output_path,
                                         f'Para_{lr}_{bs}_epoch_{epoch}model.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f'  Checkpoint saved → {ckpt_path}')

                cols  = ['train_loss', 'train_acc', 'val_loss', 'Sn', 'Sp',
                         'Pre', 'F1', 'Acc', 'MCC', 'AUC', 'PRAUC',
                         'time_min', 'params']
                df    = pd.DataFrame(results, index=cols).T
                df.to_csv(os.path.join(output_path, f'results_{tag}.csv'))

            if early_stop.early_stop:
                break

        total_time = (time.time() - start) / 60
        print(f'\nTraining complete in {total_time:.2f} minutes.')

        # Final CSV & plots
        cols = ['train_loss', 'train_acc', 'val_loss', 'Sn', 'Sp', 'Pre',
                'F1', 'Acc', 'MCC', 'AUC', 'PRAUC', 'time_min', 'params']
        df   = pd.DataFrame(results, index=cols).T
        df.to_csv(os.path.join(output_path, f'results_{tag}.csv'))
        make_plot(df['train_acc'].tolist(), df['Acc'].tolist(),
                  df['train_loss'].tolist(), df['val_loss'].tolist(),
                  output_path, tag)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train NanoNmD: CNN for Nm site detection from Nanopore signals'
    )
    parser.add_argument('--pos',     required=True, help='Path to positive (Nm) CSV file')
    parser.add_argument('--neg',     required=True, help='Path to negative (unmodified) CSV file')
    parser.add_argument('--out',     required=True, help='Output directory for checkpoints and results')
    parser.add_argument('--epochs',  type=int, default=200, help='Maximum training epochs (default: 200)')
    parser.add_argument('--seed',    type=int, default=1029, help='Random seed (default: 1029)')
    parser.add_argument('--oversample', type=int, default=3,
                        help='Repeat positive samples N times to balance classes (default: 3)')
    return parser.parse_args()


def main():
    args = parse_args()
    seed_torch(args.seed)

    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'PyTorch {torch.__version__} | Device: {device}')

    train_model(
        pos_data_path    = args.pos,
        neg_data_path    = args.neg,
        output_path      = args.out,
        device           = device,
        epochs           = args.epochs,
        oversample_factor= args.oversample,
    )


if __name__ == '__main__':
    main()
