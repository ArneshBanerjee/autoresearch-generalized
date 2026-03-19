"""
Wrapper that runs the training notebook via papermill and prints metrics.
Agent edits this file.

Usage: uv run train.py
"""

import json
import os
import sys
import time

NOTEBOOK_INPUT = "train_notebook.ipynb"
NOTEBOOK_OUTPUT = "output.ipynb"
METRICS_FILE = "metrics.json"


def main():
    t_start = time.time()

    # Run the notebook via papermill
    import papermill as pm

    print(f"Running {NOTEBOOK_INPUT} via papermill...")
    try:
        pm.execute_notebook(
            NOTEBOOK_INPUT,
            NOTEBOOK_OUTPUT,
            kernel_name="python3",
            cwd=os.getcwd(),
        )
    except pm.PapermillExecutionError as e:
        print(f"Notebook execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Read metrics written by the notebook
    if not os.path.exists(METRICS_FILE):
        print(f"ERROR: {METRICS_FILE} not found — notebook did not write metrics", file=sys.stderr)
        sys.exit(1)

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    total_time = time.time() - t_start

    # Print metrics in autoresearch key: value format
    print("---")
    print(f"val_accuracy:     {metrics['val_accuracy']:.6f}")
    print(f"train_accuracy:   {metrics.get('train_accuracy', 0.0):.6f}")
    print(f"num_features:     {metrics.get('num_features', 0)}")
    print(f"training_seconds: {total_time:.1f}")


if __name__ == "__main__":
    main()
