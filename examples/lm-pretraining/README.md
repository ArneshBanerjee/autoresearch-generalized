# LM Pretraining Example

This example reproduces the original autoresearch setup: GPT pretraining on ClimbMix data with val_bpb as the metric.

## Setup

```bash
uv sync
uv run prepare.py    # Download data + train tokenizer (~2 min)
uv run train.py      # Single training run (~5 min)
```

## Files

- `prepare.py` — Data download, tokenizer training, dataloader, evaluation (read-only)
- `train.py` — GPT model, optimizer, training loop (agent edits this)
- `autoresearch.yaml` — Autoresearch config

## Metric

- **val_bpb** (validation bits per byte) — lower is better
- Vocab-size-independent, so architectural changes are fairly compared
