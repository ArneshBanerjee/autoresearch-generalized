# Time Series Forecasting Example

ETTh1 (Electricity Transformer Temperature) forecasting with an MLP baseline. Demonstrates autoresearch on a regression task with tabular-ish time series data.

## Setup

```bash
uv sync
uv run prepare.py    # Download ETTh1 + create windowed splits
uv run train.py      # Single training run (~2 min)
```

## Files

- `prepare.py` — Downloads ETTh1, creates windowed train/val/test splits (read-only)
- `model.py` — Simple MLP baseline (agent edits this)
- `train.py` — Training loop with MSE loss (agent edits this)
- `autoresearch.yaml` — Autoresearch config

## Metric

- **val_mse** — mean squared error on validation set, lower is better

## Task

Predict the next 24 hours of the "OT" (oil temperature) feature given the previous 96 hours of all 7 features.
