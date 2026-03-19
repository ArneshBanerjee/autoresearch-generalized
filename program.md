# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Read the config**: Read `autoresearch.yaml` in the repo root. This defines the project, editable files, metric, run command, and all other settings.
2. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master/main.
4. **Read the context files**: Read every file listed in `context` in the config. These give you the full picture of the project.
5. **Read the editable files**: Read every file listed in `editable`. These are the files you'll be modifying.
6. **Run data preparation** (if configured): If `prepare` is set in the config, check whether `prepare_check` passes first (if defined). If the check fails or isn't defined, run the prepare command.
7. **Initialize results.tsv**: Create the results file (default: `results.tsv`) with just the header row. The columns are: `commit`, the metric name from the config, then any additional columns from `results_columns`, then `status`, then `description`.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single machine. The training script runs for a **fixed time budget** as defined in `autoresearch.yaml` (default: 5 minutes wall clock). You launch it using the `run` command from the config.

The environment variable `AUTORESEARCH_TIME_BUDGET` is set to the configured `time_budget` value when running experiments.

**What you CAN do:**
- Modify files listed in `editable` in the config. Everything within those files is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify any file NOT listed in `editable`. Context files are read-only. They contain fixed evaluation, data loading, and constants. (Note: Jupyter notebooks listed in `context` are read-only — edit the `.py` modules they import instead.)
- Install new packages or add dependencies. You can only use what's already declared in the project's dependency file (if `dependencies` is set in the config).
- Modify the evaluation harness or metric computation.

**The goal is simple: optimize the metric.** Check `metric.direction` in the config — if `minimize`, get the lowest value; if `maximize`, get the highest. Since the time budget is fixed, you don't need to worry about training time — it's always the same. Everything within the editable files is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as-is.

## Output format

The training script prints key-value pairs in the format `key: value`. To extract the primary metric from the log:

Use the `metric.extract` command from the config, e.g.:
```
grep '^val_bpb:' run.log | tail -1 | awk '{print $2}'
```

Any additional columns defined in `results_columns` have their own `extract` commands.

## Logging results

When an experiment is done, log it to the results file (default: `results.tsv`). This is tab-separated (NOT comma-separated — commas break in descriptions).

The TSV has a header row. The columns are:

1. `commit` — git commit hash (short, 7 chars)
2. The primary metric name (from `metric.name`) — the value achieved (use 0 for crashes)
3. Any additional columns from `results_columns` (use 0 for crashes)
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short text description of what this experiment tried

Example (for a project with metric `val_bpb` and extra column `memory_gb`):

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Constraints

If `constraints` are defined in the config, these are secondary metrics to monitor. They don't affect keep/discard decisions, but you should be aware of soft limits. If a constraint exceeds its `soft_limit`, note it but don't necessarily discard — use judgment about whether the tradeoff is worthwhile.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify the editable file(s) with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `<run_command> > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context). Set `AUTORESEARCH_TIME_BUDGET=<time_budget>` in the environment.
5. Read out the results using the extraction commands from the config.
6. If the extraction output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the TSV
8. If the metric improved (in the configured direction), you "advance" the branch, keeping the git commit
9. If the metric is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take roughly the configured `time_budget` (+ a few seconds for startup and eval overhead). If a run exceeds the `timeout` (default: 2x time_budget), kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
