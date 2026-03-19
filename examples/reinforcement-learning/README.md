# Reinforcement Learning Example

CartPole-v1 with a PPO agent. Demonstrates autoresearch in a completely different domain — the agent only edits `agent.py` while `train.py` (the eval harness) is read-only.

## Setup

```bash
pip install torch gymnasium numpy
python prepare.py    # Verify gym install
python train.py      # Single training run (~3 min)
```

## Files

- `prepare.py` — Verifies gymnasium is installed (read-only)
- `train.py` — Environment setup and evaluation harness (read-only)
- `agent.py` — PPO agent implementation (agent edits this)
- `autoresearch.yaml` — Autoresearch config

## Metric

- **mean_reward** — average reward over 100 evaluation episodes, higher is better
- CartPole-v1 max score is 500
