# Notebook Classification Example

Breast cancer classification with sklearn pipelines. Demonstrates autoresearch with a Jupyter notebook as read-only context — the notebook orchestrates training by importing from editable `.py` modules.

## Pattern

Notebooks don't play well with autoresearch directly (no stdout metrics, messy diffs). The solution:

1. **`model.py`** (editable) — pipeline definition, hyperparameters, feature selection
2. **`train.py`** (editable) — runs the notebook via papermill, extracts and prints metrics
3. **`train_notebook.ipynb`** (read-only context) — orchestrator that imports `model.py`, trains, evaluates, writes `metrics.json`

The agent edits `model.py` and `train.py`. The notebook is read-only context that shows the agent how the pieces fit together.

## Setup

```bash
uv run prepare.py        # Download breast cancer dataset
uv run train.py          # Single training run (~10 sec)
```

## Files

- `prepare.py` — Downloads breast cancer dataset to `data/` (read-only)
- `model.py` — Pipeline definition + hyperparameters (agent edits this)
- `train_notebook.ipynb` — Training orchestrator notebook (read-only context)
- `train.py` — Wrapper: runs notebook via papermill, prints metrics (agent edits this)
- `autoresearch.yaml` — Autoresearch config

## Metric

- **val_accuracy** — higher is better (0.0 to 1.0)
