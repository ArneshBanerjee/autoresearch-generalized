"""
One-time data preparation for breast cancer classification.
Downloads the dataset and saves it as CSV.

Usage: uv run prepare.py
"""

import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = "data"


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    path = os.path.join(DATA_DIR, "breast_cancer.csv")
    df.to_csv(path, index=False)

    print(f"\nData ready at {path}")
    print(f"  Samples:  {len(df)}")
    print(f"  Features: {len(data.feature_names)}")
    print(f"  Classes:  {len(data.target_names)} ({', '.join(data.target_names)})")


if __name__ == "__main__":
    prepare()
