#!/bin/bash -l
#SBATCH --job-name=aevla-official-eval
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00

# Resource/process orchestration only. All scored rollouts use scripts/evaluate.sh.
set -eo pipefail
export PROJECT_WS="$(ws_find mutil-robots)"
test -n "$PROJECT_WS" && test -d "$PROJECT_WS"
export TRAIN_ROOT="$PROJECT_WS/code/mutil-robots"
export EVAL_SOURCE="$PROJECT_WS/code/mutil-robots-eval"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the actual checkpoint step directory}"
test -d "$CHECKPOINT_DIR/params" && test -d "$CHECKPOINT_DIR/assets"
if [[ $# -eq 0 ]]; then set -- --task stack_cubes; fi
command -v setsid >/dev/null

export RUN_DIR="$PROJECT_WS/eval_runs/official_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cp "$EVAL_SOURCE/scripts/slurm/eval_official.sh" "$EVAL_SOURCE/scripts/slurm/serve_warmed.py" \
    "$EVAL_SOURCE/scripts/slurm/wait_for_policy.py" "$RUN_DIR/"
export VLA_SERVER_URL=http://127.0.0.1:20011
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

SERVER_PID=""
cleanup() {
    local result=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Batch exit code: $result; results: $RUN_DIR"
    exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

source "$PROJECT_WS/load_eval_env.sh"
command -v bc >/dev/null || { echo 'Required official dependency is missing: bc'; exit 1; }

# Independent source snapshot: preserve the official scripts and output layout.
# Assets are shared, not copied. No files in the original checkout are changed.
python - <<'PY'
import os
import shutil
from pathlib import Path

source = Path(os.environ["EVAL_SOURCE"])
destination = Path(os.environ["RUN_DIR"]) / "eval_repo"
def ignore(directory, names):
    excluded = {".git", "__pycache__", ".venv", "eval_video", "tmp", "demos"}
    if Path(directory) == source / "RoboFactory":
        excluded.add("assets")
    return [name for name in names if name in excluded or name.startswith("eval_results_")]
shutil.copytree(source, destination, ignore=ignore, symlinks=True)
assets = source / "RoboFactory/assets"
if not assets.is_dir():
    raise FileNotFoundError(assets)
(destination / "RoboFactory/assets").symlink_to(assets, target_is_directory=True)
PY

EVAL_RUN="$RUN_DIR/eval_repo"
export WARMUP_PHASE_CONFIG="$EVAL_RUN/RoboFactory/script/MACG/two_arms_stack_cube/stack_cubes_indomain.yaml"
{
    echo "Started: $(date -Is)"
    echo "Node: $(hostname -f); Job: $SLURM_JOB_ID"
    echo "Checkpoint: $CHECKPOINT_DIR"
    echo "Evaluation Python: $(command -v python)"
    printf 'Official arguments:'; printf ' %q' "$@"; printf '\n'
    echo 'Training revision and status:'
    git -C "$TRAIN_ROOT" rev-parse HEAD
    git -C "$TRAIN_ROOT" status --short
    echo 'Evaluation revision and status:'
    git -C "$EVAL_SOURCE" rev-parse HEAD
    git -C "$EVAL_SOURCE" status --short
    nvidia-smi
} > "$RUN_DIR/run_info.txt"
python -m pip freeze > "$RUN_DIR/eval-packages.txt"
git -C "$TRAIN_ROOT" diff HEAD > "$RUN_DIR/training-local.patch"
git -C "$EVAL_SOURCE" diff HEAD > "$RUN_DIR/eval-local.patch"
for file in "$PROJECT_WS/macg_data_revision.txt" "$PROJECT_WS/eval_assets_revision.txt"; do
    if [[ -f "$file" ]]; then cp "$file" "$RUN_DIR/"; fi
done

# Validate the exact official task selection before loading the model.
bash "$EVAL_RUN/scripts/evaluate.sh" "$@" --server-url "$VLA_SERVER_URL" --dry-run \
    > "$RUN_DIR/official_commands.txt"
python "$EVAL_SOURCE/scripts/slurm/check_scene_assets.py" --repo "$EVAL_RUN" "$@" \
    > "$RUN_DIR/scene_assets_check.json"

(
    source "$PROJECT_WS/load_env.sh"
    export PYTHONNOUSERSITE=1
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
    cd "$TRAIN_ROOT"
    exec setsid uv run --project "$TRAIN_ROOT" --frozen --with flask python "$RUN_DIR/serve_warmed.py" \
        --config aevla_pi05_skill_lora_two_arms_multitask \
        --checkpoint-dir "$CHECKPOINT_DIR" --host 127.0.0.1 --port 0
) > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
export SERVER_PID

# The health endpoint is exposed only after model loading AND warmup finish.
python "$RUN_DIR/wait_for_policy.py" --pid "$SERVER_PID" --run-dir "$RUN_DIR" \
    > "$RUN_DIR/server_url.txt"
export VLA_SERVER_URL="$(cat "$RUN_DIR/server_url.txt")"
echo "Model ready after warmup: $VLA_SERVER_URL"
bash "$EVAL_RUN/scripts/evaluate.sh" "$@" --server-url "$VLA_SERVER_URL" --dry-run \
    > "$RUN_DIR/official_commands.txt"

echo "Starting official evaluator: $(date -Is)"
bash "$EVAL_RUN/scripts/evaluate.sh" "$@" --server-url "$VLA_SERVER_URL" \
    > "$RUN_DIR/evaluate.log" 2>&1
echo "Official evaluator completed: $(date -Is)"
echo "Official metrics and videos: $EVAL_RUN/RoboFactory"
