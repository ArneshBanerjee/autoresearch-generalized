# Image Classification Example

CIFAR-10 classification with a ResNet baseline. Demonstrates autoresearch with multi-file editing and a maximize metric direction.

## Setup

```bash
pip install torch torchvision
python prepare.py    # Download CIFAR-10 (~170MB)
python train.py      # Single training run (~2 min)
```

## Files

- `prepare.py` — Downloads CIFAR-10 and creates train/val splits (read-only)
- `model.py` — ResNet-18 baseline (agent edits this)
- `train.py` — SGD training loop (agent edits this)
- `autoresearch.yaml` — Autoresearch config

## Metric

- **val_accuracy** — higher is better (0.0 to 1.0)
