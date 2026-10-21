#!/usr/bin/env bash
# Public entry point for the released Future Robots evaluation protocol.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../RoboFactory" && pwd)"
TASK_DIR="$ROBOFACTORY_ROOT/script/MACG/eval_scripts_rule_base"
PROXY_SCRIPT="$ROBOFACTORY_ROOT/script/MACG/eval_scripts_rule_base_peram/run_with_proxy.sh"
ALL_TASKS=(
    stack_bowls stack_cubes push_cubes place_can_basket place_burger_fries
    place_object_cabinet handover_bottles place_cube_in_bowl place_object_bowl_cabinet
)

usage() {
    cat <<'EOF'
Usage: bash scripts/evaluate.sh [options]

Run the released clean, two-arm evaluation from any working directory.
With no --task, run all nine tasks in the original suite order. Each task runs
its shipped instruction modes with 100 episodes per mode (seeds 1000–1099).

Options:
  --task NAME          Select a task; repeat to select several. 'all' selects
                       the complete suite and cannot be combined with a task.
  --mode MODE          'single' (default) or 'per-arm'.
  --server-url URL     Unified policy endpoint (or set VLA_SERVER_URL).
  --arm0-url URL       Arm 0 endpoint (or set ARM0_VLA_SERVER_URL).
  --arm1-url URL       Arm 1 endpoint (or set ARM1_VLA_SERVER_URL).
  --dry-run            Print validated commands; start no evaluations or proxy.
  --list-tasks         List the supported tasks and exit.
  -h, --help           Show this help and exit.

Examples:
  bash scripts/evaluate.sh --task stack_cubes --server-url http://127.0.0.1:8000
  bash scripts/evaluate.sh --mode per-arm --task stack_bowls \
    --arm0-url http://127.0.0.1:8000 --arm1-url http://127.0.0.1:8001
  VLA_SERVER_URL=http://127.0.0.1:8000 bash scripts/evaluate.sh --dry-run

Endpoints must be compatible with the /health and /infer HTTP pickle protocol.
Per-arm mode starts and stops the existing local dual-arm proxy for each task;
PERARM_PROXY_HOST/PORT/LOG, PERARM_INFER_TIMEOUT_S and PYTHON_BIN are inherited.
The rollout scripts call 'python', so activate your simulation environment first.

Outputs retain their original layout: RoboFactory/eval_results_* (logs, episode
JSONL and summary JSON) and RoboFactory/eval_video/. Task wrappers refuse to
reuse existing video output directories; archive them before a rerun.
This entry point preserves task configs, seeds and metrics. Use the per-task
Python evaluators directly for custom episodes or visual perturbation runs.
EOF
}

fail() {
    printf 'Error: %s\nRun with --help for usage.\n' "$*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
}

validate_url() {
    local label="$1" value="$2"
    [[ "$value" =~ ^https?://[^[:space:]/]+(/[^[:space:]]*)?$ ]] \
        || fail "$label must be an http:// or https:// URL"
}

mode="single"
server_url="${VLA_SERVER_URL:-}"
arm0_url="${ARM0_VLA_SERVER_URL:-}"
arm1_url="${ARM1_VLA_SERVER_URL:-}"
explicit_server=0
explicit_arms=0
dry_run=0
tasks=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            require_value "$@"
            tasks+=("$2")
            shift 2
            ;;
        --mode)
            require_value "$@"
            mode="$2"
            shift 2
            ;;
        --server-url)
            require_value "$@"
            server_url="$2"
            explicit_server=1
            shift 2
            ;;
        --arm0-url|--arm1-url)
            require_value "$@"
            if [[ "$1" == --arm0-url ]]; then arm0_url="$2"; else arm1_url="$2"; fi
            explicit_arms=1
            shift 2
            ;;
        --dry-run) dry_run=1; shift ;;
        --list-tasks) printf '%s\n' "${ALL_TASKS[@]}"; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option: $1" ;;
    esac
done

[[ "$mode" == single || "$mode" == per-arm ]] || fail "--mode must be single or per-arm"
if [[ ${#tasks[@]} -eq 0 || (${#tasks[@]} -eq 1 && "${tasks[0]}" == all) ]]; then
    tasks=("${ALL_TASKS[@]}")
fi

selected=()
for task in "${tasks[@]}"; do
    found=0
    for candidate in "${ALL_TASKS[@]}"; do
        if [[ "$task" == "$candidate" ]]; then found=1; break; fi
    done
    [[ "$found" -eq 1 ]] || fail "unsupported task '$task'; use --list-tasks (use 'all' alone)"
    for candidate in "${selected[@]+"${selected[@]}"}"; do
        [[ "$task" != "$candidate" ]] || fail "task '$task' was selected more than once"
    done
    [[ -f "$TASK_DIR/$task.sh" ]] || fail "missing evaluation script: $TASK_DIR/$task.sh"
    selected+=("$task")
done

if [[ "$mode" == single ]]; then
    [[ "$explicit_arms" -eq 0 ]] || fail "--arm0-url and --arm1-url require --mode per-arm"
    [[ -n "$server_url" ]] || fail "set VLA_SERVER_URL or provide --server-url"
    validate_url "--server-url / VLA_SERVER_URL" "$server_url"
    export VLA_SERVER_URL="$server_url"
else
    [[ "$explicit_server" -eq 0 ]] || fail "--server-url requires --mode single"
    [[ -n "$arm0_url" && -n "$arm1_url" ]] \
        || fail "per-arm mode requires both arm endpoints (flags or ARM0/1_VLA_SERVER_URL)"
    validate_url "arm 0 endpoint" "$arm0_url"
    validate_url "arm 1 endpoint" "$arm1_url"
    [[ -f "$PROXY_SCRIPT" ]] || fail "missing per-arm proxy launcher: $PROXY_SCRIPT"
    proxy_port="${PERARM_PROXY_PORT:-31888}"
    [[ "$proxy_port" =~ ^[0-9]{1,5}$ ]] \
        && ((10#$proxy_port >= 1 && 10#$proxy_port <= 65535)) \
        || fail "PERARM_PROXY_PORT must be an integer from 1 to 65535"
    export ARM0_VLA_SERVER_URL="$arm0_url" ARM1_VLA_SERVER_URL="$arm1_url"
fi

if [[ "$dry_run" -eq 0 ]]; then
    for dependency in python bc; do
        command -v "$dependency" >/dev/null 2>&1 || fail "required command is missing: $dependency"
    done
    if [[ "$mode" == per-arm ]]; then
        command -v curl >/dev/null 2>&1 || fail "required command is missing: curl"
        command -v "${PYTHON_BIN:-python}" >/dev/null 2>&1 \
            || fail "proxy Python command is missing: ${PYTHON_BIN:-python}"
    fi
fi

printf '[evaluate] working directory: %s\n' "$ROBOFACTORY_ROOT"
printf '[evaluate] mode: %s; tasks:' "$mode"
printf ' %s' "${selected[@]}"
printf '\n'
if [[ "$mode" == single ]]; then
    printf '[evaluate] VLA_SERVER_URL=%s\n' "$VLA_SERVER_URL"
else
    printf '[evaluate] ARM0_VLA_SERVER_URL=%s\n' "$ARM0_VLA_SERVER_URL"
    printf '[evaluate] ARM1_VLA_SERVER_URL=%s\n' "$ARM1_VLA_SERVER_URL"
fi

cd -- "$ROBOFACTORY_ROOT"
for task in "${selected[@]}"; do
    # Legacy wrappers spawn Bash again. Inherit errexit so an inference-process
    # error cannot silently become a failed episode and continue for 100 seeds.
    command=(env SHELLOPTS=errexit bash "$TASK_DIR/$task.sh")
    if [[ "$mode" == per-arm ]]; then
        command=(bash "$PROXY_SCRIPT" "${command[@]}")
    fi
    printf '[evaluate] command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "$dry_run" -eq 0 ]]; then
        "${command[@]}"
    fi
done
