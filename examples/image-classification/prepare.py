"""
One-time data preparation for CIFAR-10 image classification.
Downloads the dataset and creates train/val splits.

Usage: uv run prepare.py
"""

import os
import torchvision
import torchvision.transforms as transforms

DATA_DIR = os.path.join("data", "cifar-10")

def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading CIFAR-10 training set...")
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)

    print("Downloading CIFAR-10 test set...")
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    print(f"\nData ready at {DATA_DIR}")
    print("  Training: 50,000 images")
    print("  Test:     10,000 images")

if __name__ == "__main__":
    prepare()
