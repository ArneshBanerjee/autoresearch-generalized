"""
One-time data preparation for ETTh1 time series forecasting.
Downloads the dataset and creates windowed train/val/test splits.

Usage: uv run prepare.py
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import torch

DATA_DIR = os.path.join("data", "etth1")
RAW_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

# Task config
INPUT_LEN = 96     # Look-back window (hours)
PRED_LEN = 24      # Prediction horizon (hours)
TARGET_COL = "OT"  # Target column


def download_etth1():
    """Download ETTh1.csv if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "ETTh1.csv")
    if os.path.exists(csv_path):
        print(f"  ETTh1.csv already exists at {csv_path}")
        return csv_path

    print("  Downloading ETTh1.csv...")
    urllib.request.urlretrieve(RAW_URL, csv_path)
    print(f"  Saved to {csv_path}")
    return csv_path


def create_splits(csv_path):
    """Create windowed train/val/test splits as tensors."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["date"])

    # Normalize using training statistics
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end]
    mean = train_df.mean()
    std = train_df.std()
    std[std < 1e-8] = 1.0  # avoid division by zero
    df_norm = (df - mean) / std

    feature_cols = [c for c in df_norm.columns if c != TARGET_COL]
    all_cols = feature_cols + [TARGET_COL]
    data = df_norm[all_cols].values.astype(np.float32)
    target_idx = len(feature_cols)  # last column

    splits = {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:],
    }

    for name, split_data in splits.items():
        windows_x, windows_y = [], []
        for i in range(len(split_data) - INPUT_LEN - PRED_LEN + 1):
            windows_x.append(split_data[i : i + INPUT_LEN])                     # all features
            windows_y.append(split_data[i + INPUT_LEN : i + INPUT_LEN + PRED_LEN, target_idx])  # target only
        x = torch.tensor(np.array(windows_x))
        y = torch.tensor(np.array(windows_y))
        torch.save({"x": x, "y": y}, os.path.join(DATA_DIR, f"{name}.pt"))
        print(f"  {name}: {len(x)} windows, x={tuple(x.shape)}, y={tuple(y.shape)}")

    # Save normalization stats for later use
    torch.save({"mean": mean.values, "std": std.values, "columns": all_cols},
               os.path.join(DATA_DIR, "stats.pt"))
    print(f"  Saved normalization stats")


def prepare():
    print("Preparing ETTh1 dataset...")
    csv_path = download_etth1()
    print(f"\nCreating windowed splits (input={INPUT_LEN}h, pred={PRED_LEN}h, target={TARGET_COL})...")
    create_splits(csv_path)
    print(f"\nData ready at {DATA_DIR}")


if __name__ == "__main__":
    prepare()
