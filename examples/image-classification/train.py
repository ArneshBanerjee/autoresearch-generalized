"""
CIFAR-10 training script for autoresearch.
Agent edits this file (and model.py).

Prints key: value pairs for metric extraction.
Usage: python train.py
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import make_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", 120))
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 2
DATA_DIR = os.path.join("data", "cifar-10")

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

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# ---------------------------------------------------------------------------
# Model + Optimizer
# ---------------------------------------------------------------------------

model = make_model(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params / 1e6:.2f}M")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

t_start = time.time()
epoch = 0
best_val_acc = 0.0

while True:
    elapsed = time.time() - t_start
    if elapsed >= TIME_BUDGET:
        break

    epoch += 1
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        if time.time() - t_start >= TIME_BUDGET:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    train_acc = correct / total if total > 0 else 0.0

    # Evaluate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = val_correct / val_total
    best_val_acc = max(best_val_acc, val_acc)
    print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

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
print(f"val_accuracy:     {best_val_acc:.6f}")
print(f"training_seconds: {total_time:.1f}")
print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
print(f"num_epochs:       {epoch}")
print(f"num_params_M:     {num_params / 1e6:.2f}")
