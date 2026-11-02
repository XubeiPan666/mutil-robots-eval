# NHR Alex: official evaluation through Slurm

This directory is a site-specific launcher added in this fork, not an upstream
evaluation protocol. `eval_official.sh` starts a JAX policy HTTP server and invokes
the repository's unchanged `scripts/evaluate.sh` inside a per-job source snapshot.
It uses the already configured training uv environment and separate simulator
Conda environment. No task code, prompt, seed budget, scoring predicate, or task
timeout is changed.

## Server prerequisites

The existing NHR workspace `mutil-robots` must contain:

```text
load_env.sh
load_eval_env.sh
code/mutil-robots/             # training checkout, uv environment, deploy_minimal.py
code/mutil-robots-eval/        # this checkout
```

The GPU node must provide NVIDIA/Vulkan support, `setsid`, `bc`, and Git.
The environments, assets and checkpoint must already exist. Environment setup
scripts are local infrastructure, not included in this repository.

## Preview without starting evaluation

On an Alex login node:

```bash
export PROJECT_WS="$(ws_find mutil-robots)"
source "$PROJECT_WS/load_eval_env.sh"
cd "$PROJECT_WS/code/mutil-robots-eval"
bash -n scripts/slurm/eval_official.sh
bash scripts/evaluate.sh --task stack_cubes \
  --server-url http://127.0.0.1:20011 --dry-run
```

## Submit

Set CHECKPOINT_DIR to a real saved step directory containing params/ and assets/:

```bash
export CHECKPOINT_DIR="/absolute/path/to/checkpoint/step"
ls "$CHECKPOINT_DIR"
mkdir -p "$PROJECT_WS/logs"
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/eval_official_%j.out" \
  --error="$PROJECT_WS/logs/eval_official_%j.err" \
  scripts/slurm/eval_official.sh --task stack_cubes
```

The default request is one A40, eight CPUs and 24 hours. The job continues after
the SSH client disconnects. It stops on completion, errors, cancellation, node
failure or the Slurm time limit. There is no automatic resume.

Task selection is passed directly to the official launcher:

```text
--task stack_cubes
--task push_cubes
--task stack_cubes --task push_cubes
--task all
```

No task arguments defaults to stack_cubes. That task runs four conditions with
100 seeds each (1000–1099), totaling 400 episodes. The current all suite runs 26
conditions / 2600 episodes and may exceed the 24-hour limit. These counts describe
the public convenience suite, not a frozen paper protocol. Start with one task.

## Outputs and provenance

Each job writes to `$PROJECT_WS/eval_runs/official_JOB_ID/`:

- `run_info.txt`: checkpoint, arguments, node/GPU, Git commits and working status.
- `training-local.patch`, `eval-local.patch`: tracked local source differences.
- `training-packages.txt`, `eval-packages.txt`: actual Python package versions.
- `official_commands.txt`: official dry-run output.
- `server.log`: model loading, warmup and HTTP server output.
- `evaluate.log`: official evaluation output; appears after server startup.
- `eval_repo/RoboFactory/`: original evaluation log/JSONL/summary/video layout.

The snapshot excludes generated artifacts and links to the existing shared
assets. The original checkout and its results are not overwritten. Verify that
each completed condition has 100 episode records before interpreting results.
Partial results are retained on failure and must not be reported as a full run.

## Warmup and reproducibility boundary

`serve_warmed.py` uses the repository's `deploy_minimal.main`, configuration,
transforms and HTTP handlers. Before opening the HTTP endpoint, it performs
non-scoring inference on synthetic 224x224 RGB / dual-arm inputs using the cube
phase prompts. It restores the original `Policy._rng` afterwards, so warmup does
not advance the official rollout sampling stream. It requires the current JAX
Policy internals and must be reviewed if those internals change.

The official client still uses its original request timeout. New input signatures
may trigger additional compilation; warmup does not guarantee every task avoids
timeouts. Model startup/warmup readiness has an external 1800-second limit.

This launcher preserves the checked-out evaluation protocol. It does not guarantee
the same hardware, driver or complete dependency set as the authors, bitwise
identical results, or paper reproduction. Record checkpoint, code and package
versions when comparing runs. Do not update dependencies merely to launch a job.

## Validation

Local Bash/Python syntax checks and the official task-selection dry-run passed.
The complete Slurm launcher has not yet been executed on the cluster. The model
HTTP service and one simulator episode were previously validated interactively.
