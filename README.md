# autoresearch

A framework for autonomous ML research. Give an AI agent a training setup and let it experiment overnight — it modifies code, trains, measures, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model.

Inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch). This repo extracts the domain-agnostic core into a configurable framework that works with **any** ML project — LLMs, image classification, RL, time series, or anything else.
It's also optimized to run on Nvidia GPU, or Apple M series chips or a normal CPU.

## How it works

The autoresearch loop is simple:

1. Agent reads `autoresearch.yaml` to understand your project
2. Agent establishes a baseline by running training as-is
3. Agent modifies the editable files with an experimental idea
4. Agent commits, runs training, extracts the metric
5. If improved → keep. If not → discard (git reset)
6. Repeat forever

The power comes from **constraints**: fixed time budget, single scalar metric, git tracking, limited editable files. These constraints are now **configurable** via `autoresearch.yaml` while preserving simplicity.

## Quick start

### 1. Initialize in your repo

First, clone autoresearch somewhere on your machine:

```bash
git clone https://github.com/ArneshBanerjee/autoresearch-generalized.git ~/autoresearch
```

Then, from your ML project's root directory (must be a git repo), run the init script:

```bash
~/autoresearch/bin/autoresearch-init
```

This copies two files into your project:
- `autoresearch.yaml` — config file (you fill this in)
- `program.md` — agent instructions (works out of the box)

### 2. Configure `autoresearch.yaml`

Three required fields:

```yaml
editable:
  - "train.py"          # Files the agent can modify

metric:
  name: "val_loss"      # Your metric name
  direction: "minimize" # "minimize" or "maximize"
  extract: "grep '^val_loss:' run.log | tail -1 | awk '{print $2}'"

run: "uv run train.py"  # Command to run one experiment
```

### 3. Make your training script compatible

Your script needs to:
1. **Run from repo root** — `python train.py`, `uv run train.py`, etc.
2. **Print a parseable metric** — `val_loss: 0.1234` (key-value format)
3. **Respect time budget** — read `AUTORESEARCH_TIME_BUDGET` env var (seconds)
4. **Be in a git repo**

No base classes, no imports from the framework, no decorators. Any language, any ML framework.

### 4. Start the agent

Point your AI coding agent at the config:

```
Read program.md and autoresearch.yaml, then let's start experimenting.
```

The agent runs autonomously — you can leave it overnight and come back to results.

## Configuration reference

See [`autoresearch.yaml.example`](autoresearch.yaml.example) for the fully annotated config. Key fields:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `editable` | Yes | — | Files the agent can modify |
| `metric.name` | Yes | — | Metric name as printed by training script |
| `metric.direction` | Yes | — | `"minimize"` or `"maximize"` |
| `metric.extract` | Yes | — | Shell command to extract metric from run.log |
| `run` | Yes | — | Command to run one experiment |
| `context` | No | `[]` | Read-only files for agent context |
| `time_budget` | No | `300` | Training wall-clock seconds |
| `timeout` | No | `2x time_budget` | Hard kill threshold |
| `prepare` | No | — | One-time data/setup command |
| `prepare_check` | No | — | Skip prepare if this exits 0 |
| `results_columns` | No | `[]` | Extra TSV columns with extract commands |
| `constraints` | No | `[]` | Secondary metrics with soft limits |
| `dependencies` | No | — | Dependency file (agent won't modify) |

## Examples

Five complete examples demonstrating different domains. Each example includes:

- **`.py` scripts** (`model.py`, `train.py`) — What autoresearch executes. Modular files that the agent edits independently.
- **`.ipynb` notebooks** (`train_notebook.ipynb`) — Self-contained, single-file versions for interactive exploration in Jupyter. All model definitions and training logic are inlined — no imports from local `.py` modules. Useful for prototyping, understanding the pipeline, or running experiments manually. Not used by autoresearch directly.

### [LM Pretraining](examples/lm-pretraining/)
The original autoresearch use case. GPT pretraining on ClimbMix data, optimizing val_bpb (minimize). Single editable file, 5-minute budget.

### [Image Classification](examples/image-classification/)
CIFAR-10 with ResNet-18. Optimizes val_accuracy (maximize). Multi-file editing (model.py + train.py), 2-minute budget.

### [Reinforcement Learning](examples/reinforcement-learning/)
CartPole-v1 with PPO. Optimizes mean_reward (maximize). Agent only edits the policy (agent.py), while the eval harness (train.py) is read-only. 3-minute budget.

### [Time Series Forecasting](examples/time-series/)
ETTh1 electricity transformer temperature forecasting with MLP. Optimizes val_mse (minimize). Multi-file editing, 2-minute budget.

### [Notebook Classification](examples/notebook-classification/)
Breast cancer classification with sklearn. Demonstrates the **notebook pattern** — a Jupyter notebook is read-only context that orchestrates training, while the agent edits the `.py` modules it imports. Uses papermill to execute the notebook and extract metrics.

## Working with Notebooks

There are two notebook patterns in this repo:

### 1. Standalone notebooks (for exploration)

Each example's `train_notebook.ipynb` is fully self-contained — all model definitions and training logic are inlined. Open one file and run everything, no `.py` imports needed. These are for interactive exploration, prototyping, and understanding the pipeline. autoresearch does not use them.

### 2. Papermill pattern (for autoresearch integration)

Notebooks don't work well as editable files in autoresearch (no stdout metrics, messy git diffs). The recommended pattern for autoresearch-driven notebook projects:

1. **Extract editable logic into `.py` modules** — pipeline definitions, hyperparameters, feature selection
2. **Keep the notebook as read-only context** — it imports from the `.py` modules, trains, and writes `metrics.json`
3. **Add a `train.py` wrapper** — runs the notebook via [papermill](https://papermill.readthedocs.io/), reads `metrics.json`, prints metrics to stdout

See [`examples/notebook-classification/`](examples/notebook-classification/) for a complete working example.

## Design philosophy

- **Zero coupling** — Your code doesn't import anything from autoresearch. The framework is just a config file and agent instructions.
- **Constraints are features** — Fixed time budget, single metric, limited editable files. These constraints make the agent effective by keeping the search space manageable.
- **Simplicity criterion** — All else being equal, simpler is better. An improvement that adds ugly complexity isn't worth it. Deleting code for equal results is a win.
- **Any domain** — If you can express your problem as "run command, extract metric," autoresearch can optimize it.

## License

MIT
