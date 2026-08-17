#!/usr/bin/env python3

import os
import json
import pickle
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_curve
from tqdm import tqdm


# ============================================================
# 1. Random seed
# ============================================================

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ============================================================
# 2. Parse one feature
# ============================================================

def parse_feature(feature):
    """Parse one Features1/2/3/4 field into a list of floats."""

    if pd.isna(feature):
        raise ValueError("Feature contains NaN.")

    feature = str(feature).strip()
    feature = feature.replace("[", "").replace("]", "").replace("(", "").replace(")", "")

    if "|" in feature:
        values = feature.split("|")
    elif "," in feature:
        values = feature.split(",")
    else:
        values = feature.split()

    values = [x.strip() for x in values if x.strip()]

    try:
        return [float(x) for x in values]
    except ValueError:
        raise ValueError(f"Cannot parse feature: {feature}")


# ============================================================
# 3. Load feature.R004.txt
# ============================================================

def load_r004_feature_file(filepath, sequence_column="Base_Sequence"):
    """
    Read feature.R004.txt.

    Returns:
        X_signal:   (N, 20)
        X_sequence: (N, 5) string sequences
        metadata:   DataFrame
    """

    print("\n========================================")
    print("Loading feature file")
    print("========================================")
    print(filepath)

    df = pd.read_csv(filepath, sep="\t", dtype=str)
    print("Number of samples:", len(df))

    required_columns = [
        "Read_Name", "Ref_Name", "Ref_Coordinate",
        sequence_column, "Features1", "Features2",
        "Features3", "Features4"
    ]

    missing = [x for x in required_columns if x not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    signal_data, sequence_data, metadata, bad_rows = [], [], [], []

    for index, row in df.iterrows():
        try:
            # Sequence
            sequence = str(row[sequence_column]).upper().strip()

            if len(sequence) != 5:
                raise ValueError(f"Sequence length = {len(sequence)}, expected 5.")

            if any(base not in "ACGT" for base in sequence):
                raise ValueError(f"Invalid sequence: {sequence}")

            # Four signal tracks
            features = []
            for feature_column in ["Features1", "Features2", "Features3", "Features4"]:
                values = parse_feature(row[feature_column])

                if len(values) != 5:
                    raise ValueError(
                        f"{feature_column} has {len(values)} values, expected 5."
                    )

                features.append(values)

            # Convert:
            # F1p1 F1p2 F1p3 F1p4 F1p5
            # F2p1 F2p2 F2p3 F2p4 F2p5
            # F3p1 F3p2 F3p3 F3p4 F3p5
            # F4p1 F4p2 F4p3 F4p4 F4p5
            #
            # ->
            #
            # F1p1 F2p1 F3p1 F4p1
            # F1p2 F2p2 F3p2 F4p2
            # ...

            signal = np.array(features, dtype=np.float32).T.reshape(-1)

            signal_data.append(signal)
            sequence_data.append(sequence)

            metadata.append({
                "Read_Name": row["Read_Name"],
                "Ref_Name": row["Ref_Name"],
                "Ref_Coordinate": row["Ref_Coordinate"],
                "Base_Sequence": sequence
            })

        except Exception as e:
            bad_rows.append((index, str(e)))

    if bad_rows:
        print(f"\nWARNING: {len(bad_rows)} rows were removed.")
        for index, reason in bad_rows[:10]:
            print(f"Row: {index} | Reason: {reason}")

    X_signal = np.asarray(signal_data, dtype=np.float32)
    X_sequence = np.asarray(sequence_data)
    metadata = pd.DataFrame(metadata)

    print("\nSignal shape:", X_signal.shape)
    print("Sequence shape:", X_sequence.shape)
    print("\nExample signal:")
    print(X_signal[:2])
    print("\nExample sequence:")
    print(X_sequence[:2])

    return X_signal, X_sequence, metadata


# ============================================================
# 4. Sequence encoding
# ============================================================

NM_KMERS = ["A", "G", "C", "T"]
KMER_TO_INT = {base: i for i, base in enumerate(NM_KMERS)}


def encode_sequences(sequences):
    return np.asarray(
        [[KMER_TO_INT[base] for base in sequence] for sequence in sequences],
        dtype=np.int64
    )


# ============================================================
# 5. Dataset
# ============================================================

class CustomDataset(Dataset):

    def __init__(self, X_signal, X_sequence, y):
        self.X1 = torch.tensor(X_signal, dtype=torch.float32)
        self.X2 = torch.tensor(X_sequence, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.y[index]


# ============================================================
# 6. CNN model
# ============================================================

class CNN_model(nn.Module):

    def __init__(
        self,
        kernel_sizes,
        num_filters,
        num_classes,
        d_prob,
        mode,
        vacab_size,
        embedding_dim
    ):
        super().__init__()

        self.vocab_size = vacab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode

        self.embedding_table = nn.Embedding(vacab_size, embedding_dim // 2)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters[i],
                kernel_size=kernel_sizes[i]
            )
            for i in range(len(kernel_sizes))
        ])

        self.fc = nn.Sequential(
            nn.Linear(sum(num_filters), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.dropout = nn.Dropout(p=d_prob)

    def forward(self, x):
        x_sig, x_seq = x
        x_com = None

        for index, num in enumerate(range(0, 20, 4)):
            x_seq0 = self.embedding_table(x_seq[:, index])

            x0 = torch.cat([
                x_sig[:, num:num + 4],
                x_seq0
            ], dim=1)

            x_all_0 = x0.unsqueeze(1)

            if index == 0:
                x_com = x_all_0
            else:
                x_com = torch.cat([x_com, x_all_0], dim=1)

        x_reshaped = x_com.permute(0, 2, 1)

        x_conv_list = [
            F.relu(conv1d(x_reshaped))
            for conv1d in self.conv1d_list
        ]

        x_pool_list = [
            F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list
        ]

        x_fc = torch.cat(
            [x_pool.squeeze(dim=2) for x_pool in x_pool_list],
            dim=1
        )

        return self.fc(self.dropout(x_fc))


# ============================================================
# 7. Metrics
# ============================================================

def calc_metrics(y_label, y_proba, y_predict):

    con_matrix = confusion_matrix(y_label, y_predict, labels=[0, 1])
    TN, FP = con_matrix[0]
    FN, TP = con_matrix[1]

    P = TP + FN
    N = TN + FP

    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = TP / (TP + FP) if (TP + FP) > 0 else 0

    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = ((TP * TN - FP * FN) / np.sqrt(denominator)) if denominator > 0 else 0

    if len(np.unique(y_label)) == 2:
        fpr, tpr, _ = roc_curve(y_label, y_proba)
        AUC = auc(fpr, tpr)

        lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_proba)
        PRAUC = auc(lr_recall, lr_precision)
    else:
        AUC = np.nan
        PRAUC = np.nan

    f1score = f1_score(y_label, y_predict, zero_division=0)

    return Sn, Sp, Pre, f1score, Acc, MCC, AUC, PRAUC


# ============================================================
# 8. Training
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):

    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for data in tqdm(loader, desc="Training", leave=False):
        data1, data2, target = [x.to(device) for x in data]

        optimizer.zero_grad()

        outputs = model((data1, data2))
        loss = loss_fn(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * target.size(0)
        running_correct += (outputs.argmax(dim=1) == target).sum().item()
        total += target.size(0)

    return running_loss / total, running_correct / total


# ============================================================
# 9. Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, loss_fn, device):

    model.eval()

    running_loss = 0.0
    total = 0

    targets = []
    predicts = []
    probabilities = []

    for data in tqdm(loader, desc="Validation", leave=False):
        data1, data2, target = [x.to(device) for x in data]

        outputs = model((data1, data2))
        loss = loss_fn(outputs, target)

        running_loss += loss.item() * target.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        targets.extend(target.cpu().numpy())
        predicts.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

        total += target.size(0)

    val_loss = running_loss / total
    metrics = calc_metrics(targets, probabilities, predicts)

    return (val_loss,) + tuple(metrics)


# ============================================================
# 10. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Train NanoNmD on R004 features.")

    parser.add_argument("--pos", required=True, help="Positive feature.R004.txt")
    parser.add_argument("--neg", required=True, help="Negative feature.R004.txt")
    parser.add_argument("--output", required=True, help="Output model directory")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1029)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Seed and device
    # --------------------------------------------------------

    seed_torch(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\nUsing device:", device)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    os.makedirs(args.output, exist_ok=True)

    # --------------------------------------------------------
    # Load positive and negative data
    # --------------------------------------------------------

    pos_signal, pos_sequence, pos_metadata = load_r004_feature_file(args.pos)
    neg_signal, neg_sequence, neg_metadata = load_r004_feature_file(args.neg)

    print("\nPositive samples:", len(pos_signal))
    print("Negative samples:", len(neg_signal))

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    X_signal = np.concatenate([pos_signal, neg_signal], axis=0)
    X_sequence_string = np.concatenate([pos_sequence, neg_sequence], axis=0)

    y = np.concatenate([
        np.ones(len(pos_signal), dtype=np.int64),
        np.zeros(len(neg_signal), dtype=np.int64)
    ])

    X_sequence = encode_sequences(X_sequence_string)

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    indices = np.arange(len(y))

    train_idx, valid_idx = train_test_split(
        indices,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=y
    )

    print("\nTraining samples:", len(train_idx))
    print("Validation samples:", len(valid_idx))

    # --------------------------------------------------------
    # StandardScaler
    # IMPORTANT: fit ONLY on training data
    # --------------------------------------------------------

    scaler = StandardScaler()
    scaler.fit(X_signal[train_idx])

    X_train_signal = scaler.transform(X_signal[train_idx]).astype(np.float32)
    X_valid_signal = scaler.transform(X_signal[valid_idx]).astype(np.float32)

    X_train_sequence = X_sequence[train_idx]
    X_valid_sequence = X_sequence[valid_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    scaler_path = os.path.join(args.output, "scaler.pkl")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\nSaved scaler:", scaler_path)

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_dataset = CustomDataset(X_train_signal, X_train_sequence, y_train)
    valid_dataset = CustomDataset(X_valid_signal, X_valid_sequence, y_valid)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=use_cuda
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=use_cuda
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNN_model(
        kernel_sizes=[2, 3, 4],
        num_filters=[100, 100, 100],
        num_classes=2,
        d_prob=0.2,
        mode="static",
        vacab_size=4,
        embedding_dim=8
    ).to(device)

    print("\nModel parameters:", sum(p.numel() for p in model.parameters()))

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)

    print("\nTraining POS:", n_positive)
    print("Training NEG:", n_negative)

    class_weights = torch.tensor(
        [1.0, n_negative / n_positive],
        dtype=torch.float32,
        device=device
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # --------------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=10,
        factor=0.5,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_auc = -np.inf
    best_val_loss = np.inf
    early_counter = 0
    history = []

    best_model_path = os.path.join(args.output, "best_model.pth")
    final_model_path = os.path.join(args.output, "final_model.pth")

    print("\n========================================")
    print("START TRAINING")
    print("========================================")

    for epoch in range(1, args.epochs + 1):

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        (
            val_loss,
            Sn,
            Sp,
            Pre,
            f1score,
            Acc,
            MCC,
            AUC,
            PRAUC
        ) = validate(model, valid_loader, loss_fn, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train loss:       {train_loss:.6f}")
        print(f"Train accuracy:   {train_accuracy:.6f}")
        print(f"Validation loss:  {val_loss:.6f}")
        print(f"Sn:               {Sn:.6f}")
        print(f"Sp:               {Sp:.6f}")
        print(f"Precision:        {Pre:.6f}")
        print(f"F1:               {f1score:.6f}")
        print(f"Accuracy:         {Acc:.6f}")
        print(f"MCC:              {MCC:.6f}")
        print(f"AUC:              {AUC:.6f}")
        print(f"PRAUC:            {PRAUC:.6f}")
        print(f"Learning rate:    {current_lr:.2e}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "Sn": Sn,
            "Sp": Sp,
            "Precision": Pre,
            "F1": f1score,
            "Accuracy": Acc,
            "MCC": MCC,
            "AUC": AUC,
            "PRAUC": PRAUC,
            "learning_rate": current_lr
        })

        # ----------------------------------------------------
        # Save best model according to validation AUC
        # ----------------------------------------------------

        if not np.isnan(AUC) and AUC > best_auc:

            best_auc = AUC

            torch.save({
                "model_state_dict": model.state_dict(),
                "kernel_sizes": [2, 3, 4],
                "num_filters": [100, 100, 100],
                "num_classes": 2,
                "d_prob": 0.2,
                "mode": "static",
                "vocab_size": 4,
                "embedding_dim": 8,
                "best_val_auc": best_auc,
                "sequence_length": 5
            }, best_model_path)

            print(f"\n*** Best model saved | AUC = {best_auc:.6f} ***")

        # ----------------------------------------------------
        # Early stopping according to validation loss
        # ----------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
        else:
            early_counter += 1
            print(f"Early stopping counter: {early_counter}/{args.patience}")

        if early_counter >= args.patience:
            print("\nEarly stopping.")
            break

    # ========================================================
    # Save final model
    # ========================================================

    torch.save({
        "model_state_dict": model.state_dict(),
        "kernel_sizes": [2, 3, 4],
        "num_filters": [100, 100, 100],
        "num_classes": 2,
        "d_prob": 0.2,
        "mode": "static",
        "vocab_size": 4,
        "embedding_dim": 8,
        "sequence_length": 5
    }, final_model_path)

    # ========================================================
    # Save history
    # ========================================================

    history_path = os.path.join(args.output, "training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    # ========================================================
    # Save config
    # ========================================================

    config = {
        "positive_file": os.path.abspath(args.pos),
        "negative_file": os.path.abspath(args.neg),
        "best_model": os.path.abspath(best_model_path),
        "scaler": os.path.abspath(scaler_path),
        "sequence_length": 5,
        "signal_features": 20,
        "kernel_sizes": [2, 3, 4],
        "num_filters": [100, 100, 100],
        "embedding_dim": 8,
        "dropout": 0.2,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "validation_fraction": args.val_fraction,
        "seed": args.seed,
        "best_val_auc": float(best_auc)
    }

    config_path = os.path.join(args.output, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # ========================================================
    # Done
    # ========================================================

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")
    print("\nBest model:", best_model_path)
    print("Scaler:", scaler_path)
    print("Training history:", history_path)
    print("Config:", config_path)


if __name__ == "__main__":
    main()