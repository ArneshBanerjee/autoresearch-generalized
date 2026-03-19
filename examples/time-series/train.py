"""
ETTh1 time series forecasting training script for autoresearch.
Agent edits this file (and model.py).

Prints key: value pairs for metric extraction.
Usage: python train.py
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model import make_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", 120))
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
DATA_DIR = os.path.join("data", "etth1")

# Task config (must match prepare.py)
INPUT_LEN = 96
PRED_LEN = 24
NUM_FEATURES = 7

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_data = torch.load(os.path.join(DATA_DIR, "train.pt"), weights_only=True)
val_data = torch.load(os.path.join(DATA_DIR, "val.pt"), weights_only=True)

train_dataset = TensorDataset(train_data["x"], train_data["y"])
val_dataset = TensorDataset(val_data["x"], val_data["y"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")

# ---------------------------------------------------------------------------
# Model + Optimizer
# ---------------------------------------------------------------------------

model = make_model(input_len=INPUT_LEN, num_features=NUM_FEATURES, pred_len=PRED_LEN).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params / 1e6:.4f}M")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

t_start = time.time()
epoch = 0
best_val_mse = float("inf")

while True:
    elapsed = time.time() - t_start
    if elapsed >= TIME_BUDGET:
        break

    epoch += 1
    model.train()
    train_loss = 0.0
    train_count = 0

    for x_batch, y_batch in train_loader:
        if time.time() - t_start >= TIME_BUDGET:
            break

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)
        train_count += x_batch.size(0)

    scheduler.step()

    # Evaluate
    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)
            val_count += x_batch.size(0)

    val_mse = val_loss / val_count if val_count > 0 else float("inf")
    train_mse = train_loss / train_count if train_count > 0 else float("inf")
    best_val_mse = min(best_val_mse, val_mse)

    print(f"Epoch {epoch}: train_mse={train_mse:.6f} val_mse={val_mse:.6f} lr={scheduler.get_last_lr()[0]:.6f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total_time = time.time() - t_start

if device.type == "cuda":
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
elif device.type == "mps":
    peak_memory_mb = torch.mps.driver_allocated_size() / 1024 / 1024
else:
    peak_memory_mb = 0.0

print("---")
print(f"val_mse:          {best_val_mse:.6f}")
print(f"training_seconds: {total_time:.1f}")
print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
print(f"num_epochs:       {epoch}")
print(f"num_params_M:     {num_params / 1e6:.4f}")
