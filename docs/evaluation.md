# Evaluation

Complete [installation](installation.md), start your model server, and activate the simulator environment. Commands in the first sections run from the **repository root**. Replace example URLs with your own endpoints.

## Public entry point

```bash
bash scripts/evaluate.sh --help
bash scripts/evaluate.sh --list-tasks
```

Preview a task before running it:

```bash
bash scripts/evaluate.sh \
  --task stack_cubes \
  --server-url http://127.0.0.1:20011 \
  --dry-run
```

`--dry-run` prints the selected endpoint, working directory, and task commands without starting simulation, making network calls, or loading a model. It does not check that the server or scene assets are available.

### Single-server evaluation

A unified server receives both arms' observations and returns dual-arm actions:

```bash
bash scripts/evaluate.sh \
  --task stack_cubes \
  --server-url http://127.0.0.1:20011
```

Choose several tasks with repeated `--task`, or select the entire default suite:

```bash
export VLA_SERVER_URL=http://127.0.0.1:20011
bash scripts/evaluate.sh --task stack_cubes --task push_cubes
bash scripts/evaluate.sh --task all
```

If no task is specified, the launcher selects `all`. The endpoint must be supplied explicitly through `--server-url` or `VLA_SERVER_URL`.

### Per-arm evaluation

Two single-arm servers independently produce actions. The local proxy combines them:

```bash
bash scripts/evaluate.sh \
  --mode per-arm \
  --task stack_cubes \
  --arm0-url http://127.0.0.1:20011 \
  --arm1-url http://127.0.0.1:20012
```

You can also set `ARM0_VLA_SERVER_URL` and `ARM1_VLA_SERVER_URL` and omit the corresponding flags. Both servers must be healthy before evaluation begins. The proxy starts automatically and is stopped when the evaluation exits. See the [model-server contract](architecture.md#model-server-contract) for observation and action requirements.

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `VLA_SERVER_URL` | Unified server URL in single mode | Must be supplied |
| `ARM0_VLA_SERVER_URL` | Arm0 upstream URL in per-arm mode | Must be supplied |
| `ARM1_VLA_SERVER_URL` | Arm1 upstream URL in per-arm mode | Must be supplied |
| `PERARM_PROXY_HOST` | Local proxy bind address | `127.0.0.1` |
| `PERARM_PROXY_PORT` | Local proxy port | `31888` |
| `PERARM_INFER_TIMEOUT_S` | Upstream inference request timeout | `60` seconds |
| `PERARM_PROXY_LOG` | Proxy log path | `/tmp/robofactory_perarm_proxy_<PORT>_<PID>.log` |
| `PYTHON_BIN` | Python executable used to start the proxy | `python` |
| `REPLAY_MANIFEST_DIR` | Legacy task-wrapper manifest lookup directory | `release/manifests`, relative to `RoboFactory/` |

The evaluator subprocesses use the active environment's `python`. `PYTHON_BIN` selects the proxy interpreter only.

## Task coverage and evaluation conditions

The default suite runs nine dual-arm tasks in this order:

| Task argument | In-domain | Reorder | Sync | Sync + reorder |
| --- | :---: | :---: | :---: | :---: |
| `stack_bowls` | ✓ | ✓ | ✓ | ✓ |
| `stack_cubes` | ✓ | ✓ | ✓ | ✓ |
| `push_cubes` | ✓ | ✓ | ✓ | ✓ |
| `place_can_basket` | ✓ | — | ✓ | — |
| `place_burger_fries` | ✓ | ✓ | ✓ | ✓ |
| `place_object_cabinet` | ✓ | — | ✓ | — |
| `handover_bottles` | ✓ | ✓ | — | — |
| `place_cube_in_bowl` | ✓ | — | ✓ | — |
| `place_object_bowl_cabinet` | ✓ | — | ✓ | — |

The instruction-condition YAMLs define the actual phase instructions, transitions, and timing. In-domain uses the base task ordering; reorder changes the collaboration order; sync introduces coordinated phases; sync + reorder combines both. Each enabled condition runs seeds **1000–1099 inclusive**, one environment per process, for 100 episodes. The current default suite therefore schedules 26 conditions / 2,600 episodes when all task wrappers complete.

The top-level launcher preserves these task wrappers and their existing seeds, CPU physics, RGB observations, and condition choices. `--data-num` and `--checkpoint-num` in the lower-level Python files are legacy metadata fields; they do **not** select an HTTP server checkpoint or set the number of evaluated episodes. Configure the model checkpoint on the server itself.

Visual perturbation blocks in the default task wrappers are currently commented out. Finding a replay manifest does not enable those blocks. Use the explicit perturbation workflow below when evaluating visual changes.

The separate `three_arms_place_objects` research entry point is outside the default suite and requires a suitable three-arm server. It cannot use the two-arm proxy.

## One-episode smoke test

Task wrapper scripts run all 100 seeds per condition. To test **one** seed and one condition, call its Python evaluator directly from `RoboFactory/`:

```bash
cd RoboFactory
export VLA_SERVER_URL=http://127.0.0.1:20011
python script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.py \
  --config configs/table/two_arms_stack_cube.yaml \
  --status two_arms_stack_cube_12 \
  --phase-config script/MACG/two_arms_stack_cube/stack_cubes_indomain.yaml \
  --seed 1000 \
  --num-envs 1 \
  --obs-mode rgb \
  --sim-backend cpu \
  --render-mode rgb_array \
  --record-dir eval_video/smoke_stack_cubes \
  --quiet
```

Inspect the video, the final `success`/`failed` line, and the `METRICS_JSON=` line. This direct command emits an episode's metrics to stdout; it does not create the shell wrapper's aggregated log/JSONL/summary files. The evaluator's rollout limits and timeout remain at their task-specific defaults. Overriding these limits changes the evaluation protocol.

For the same single-seed test through the per-arm proxy, set both upstream URLs and prefix the Python command above with:

```bash
ARM0_VLA_SERVER_URL=http://127.0.0.1:20011 \
ARM1_VLA_SERVER_URL=http://127.0.0.1:20012 \
  bash script/MACG/eval_scripts_rule_base_peram/run_with_proxy.sh \
  python script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.py \
    --config configs/table/two_arms_stack_cube.yaml \
    --status two_arms_stack_cube_12 \
    --phase-config script/MACG/two_arms_stack_cube/stack_cubes_indomain.yaml \
    --seed 1000 --num-envs 1 --obs-mode rgb --sim-backend cpu \
    --render-mode rgb_array --record-dir eval_video/smoke_stack_cubes_perarm --quiet
```

## Results and metrics

Batch task wrappers write under `RoboFactory/`:

```text
eval_results_<scene>_<phase>_<clean-or-random>_<timestamp>.log
eval_results_<scene>_<phase>_<clean-or-random>_<timestamp>_episodes.jsonl
eval_results_<scene>_<phase>_<clean-or-random>_<timestamp>_summary.json
eval_video/...
```

The `.log` file records the legacy binary success count and running success rate. Each JSONL row captures one completed episode with its seed, status, phase config, task success, and progress metrics. The summary aggregates the captured rows. A crashed or disconnected rollout may fail before emitting a metrics row; verify the JSONL rollout count against the expected 100 episodes per condition before reporting results.

| Metric | Meaning |
| --- | --- |
| `task_success_rate` | Fraction of episodes with environment-level task success |
| `normalized_progress` | Mean fraction of milestones completed while satisfying the selected order/synchronization constraints |
| `subtask_completion` | Constraint-compliant completion rate for each milestone, with all episodes as denominator |
| `unconditional_completion` | Physical milestone completion without the constraint checks; diagnostic |
| `conditional_on_reached` | Diagnostic completion rate restricted to episodes eligible for that milestone |
| `macro_task_success`, `macro_normalized_progress` | Equal-weight averages across the task/condition groups in the supplied episode files |

Environment success and compliant progress answer different questions; report both. Tracker specs are in `script/MACG/eval_common/progress_specs/`. If tracker initialization is disabled or fails, the emitted fallback uses binary success as progress and has no milestone details. Look for `[progress] disabled` and empty milestone maps before interpreting progress results.

Aggregate selected files explicitly, from `RoboFactory/`:

```bash
python script/MACG/eval_common/aggregate_progress.py \
  eval_results_YOUR_RUN_episodes.jsonl \
  -o summary.json
```

The aggregator also accepts several JSONL files or a directory, recursively reading `*_episodes.jsonl`. Keep each model/run's files in a separate directory when using that option: repeated seeds from reruns are not deduplicated. Task wrappers refuse to reuse existing task video output directories. Archive those directories before rerunning a task or evaluating another checkpoint.

## Visual perturbations and reproducibility

Scene assets, visual perturbation assets, and recorded perturbation manifests are separate inputs. The default scene downloader supplies scene assets. Background textures, distractor objects, model checkpoints, and a verified paper-run manifest are not bundled in the Git source. Seeds alone do not establish identical visual inputs across machines or asset inventories.

The following creates a manifest for **your selected assets and environment**. It does not recreate an original paper run without the corresponding assets, revisions, and manifests. Run from `RoboFactory/`, using actual local paths for the two asset sources:

```bash
python scripts/perturbation/build_asset_bundle.py \
  --texture-root /path/to/background_texture \
  --distractor-root /path/to/objects \
  --out-dir release/perturbation_assets_v1

python scripts/perturbation/record_manifests.py \
  --task two_arms_stack_cube \
  --seeds 1000-1099 \
  --out-dir release/manifests \
  --asset-bundle-manifest release/perturbation_assets_v1/perturbation_assets_v1.manifest.json
```

`background_texture` should contain the `seen/` and/or `unseen/` folders selected in the perturbation YAML. The distractor source uses categories such as `001_...`, with `visual/`, optional `collision/`, and model metadata. Check the builder's warnings: missing source folders are skipped, so a successful command exit alone does not prove that a bundle is complete.

The current builder places the extracted asset tree at **`release/perturbation_assets_v1/perturbation_assets_v1/`**. Its manifest and tarball are in the outer directory. That nested layout matches the paths in the shipped perturbation YAMLs.

Verify selected seeds before evaluating:

```bash
export ROBOFACTORY_PERTURB_ASSETS="$PWD/release/perturbation_assets_v1/perturbation_assets_v1"
python scripts/perturbation/verify_replay.py \
  --task two_arms_stack_cube \
  --manifest release/manifests/two_arms_stack_cube_perturbed.json \
  --asset-bundle-root "$ROBOFACTORY_PERTURB_ASSETS" \
  --seeds 1000-1005
```

This checks the replayed recorded state within the tool's numerical tolerances; it does not establish pixel-identical rendering or identical model outputs on other hardware. Review the perturbation YAML when moving assets: `ROBOFACTORY_PERTURB_ASSETS` remaps recorded asset paths for replay, while fresh randomization still reads the configured source roots.

To run one perturbed episode, add these arguments to the direct Python smoke-test command:

```bash
--perturbation-config configs/perturbation/visual_perturbation_stackcube.yaml \
--replay-manifest release/manifests/two_arms_stack_cube_perturbed.json
```

For a 100-seed perturbed condition, pass the perturbation YAML as the seventh positional argument of the corresponding legacy task evaluator and supply its manifest:

```bash
VLA_SERVER_URL=http://127.0.0.1:20011 \
REPLAY_MANIFEST_FILE=release/manifests/two_arms_stack_cube_perturbed.json \
  bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh \
    configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_12 \
    script/MACG/two_arms_stack_cube/stack_cubes_indomain.yaml \
    configs/perturbation/visual_perturbation_stackcube.yaml
```

Keep the exact repository commit, simulator package export, scene and phase YAMLs, asset dataset revision, bundle hash, manifest, seed range, model/server revision, checkpoint, and result files together with any reported experiment.
