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

### Numbered objects required by the shipped scene YAMLs

The RoboFactory downloader alone does not provide the numbered objects referenced
under `release/perturbation_assets_v1/perturbation_assets_v1/distractors/` by clean
table scenes. The first all-task attempt (4903978) failed at stack_bowls because
`002_bowl/model_data3.json` was absent. This is distinct from the non-blocking
warning about a missing optional perturbation replay manifest.

The upstream RoboTwin dataset `TianxingChen/RoboTwin2.0` provides objects.zip.
The inspected revision is `ba1ebdd89aac671899a768e62dfe4dc6b29956c8`. This is a
documented asset source matching the required filenames, not proof of the exact
paper-run asset revision. Keep the original meshes, annotations and YAMLs intact.

After downloading that objects.zip to `$PROJECT_WS/downloads/robotwin2/` and saving
the revision in `$PROJECT_WS/robotwin_objects_revision.txt`, submit CPU preparation:

```bash
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/scene_assets_%j.out" \
  --error="$PROJECT_WS/logs/scene_assets_%j.err" \
  scripts/slurm/prepare_assets.sh
```

It requests four a100mig CPUs, no GPU, for up to one hour. It extracts the archive
to `$PROJECT_WS/assets/robotwin2/`, records archive SHA256/revision, maps the required
`distractors` path to the original objects tree with an absolute symlink, and
checks declared files for all nine default scene configs. No randomization or
replay is enabled by this mapping. Existing conflicting directories are refused.
Allow space for the expanded archive; failed staging directories are retained.

The check can also run independently from the simulator environment:

```bash
python scripts/slurm/check_scene_assets.py --repo "$PWD" --task all
```

It checks existence of declared file_path, filename and annotation_path values,
not physics correctness, rendering, hidden code-generated asset dependencies, or
equivalence to the authors' meshes. Each GPU job repeats the selected-task file
check before loading the policy. The per-job snapshot preserves the shared asset
symlink. Keep the shared asset files unchanged while jobs are running.

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

The model command explicitly selects the training uv project and its Python
interpreter. The external wrapper adds that checkout's root, src, and client src
directories to its own import path; it does not change the simulator's imports.
For an import-only check (no model or GPU allocation), after loading load_env.sh:

```bash
export TRAIN_ROOT="$PROJECT_WS/code/mutil-robots"
cd "$TRAIN_ROOT"
uv run --project "$TRAIN_ROOT" --frozen --with flask python \
  "$PROJECT_WS/code/mutil-robots-eval/scripts/slurm/serve_warmed.py" --check-imports
```

After warmup, Werkzeug binds a free loopback port using port 0 and writes its PID
and actual URL atomically to server_ready.json. The launcher passes this URL to
the unchanged official evaluator; it no longer uses a fixed 20011 port for actual
rollouts. This allows separate GPU jobs on the same node without port collisions.
wait_for_policy.py checks the bound endpoint and reports server.log on startup
failure instead of only raising ProcessLookupError.

This launcher preserves the checked-out evaluation protocol. It does not guarantee
the same hardware, driver or complete dependency set as the authors, bitwise
identical results, or paper reproduction. Record checkpoint, code and package
versions when comparing runs. Do not update dependencies merely to launch a job.

## Validation

The first cluster attempts (4902453 and 4903242) both failed before model loading:
the external wrapper could import deploy_minimal but not the src-layout openpi
package. The changes above address that observed import failure, add dynamic port
allocation, and improve startup diagnostics. They still need a new cluster run.
Regression tests cover external src-layout imports, restoring the warmup RNG,
two distinct loopback endpoints, and reporting the original startup error.
No scoring code or task timeout is changed. The model HTTP service and one
simulator episode were previously validated interactively.
