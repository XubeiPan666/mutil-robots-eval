#!/usr/bin/env bash
# Per-seed VLM evaluation runner for TwoArmsStackBowl.

set -u

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
if [ -f "$SCRIPT_DIR/.env.local" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.local"
    set +a
fi
export VLA_SERVER_URL="${VLA_SERVER_URL:-http://localhost:20020}"
VLA_AUTHORITY="${VLA_SERVER_URL#*://}"
VLA_AUTHORITY="${VLA_AUTHORITY%%/*}"
VLA_HOST="${VLA_AUTHORITY%%:*}"
export HTTP_PROXY="${OPENROUTER_PROXY:-${HTTP_PROXY:-}}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="$VLA_HOST,localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <config> <data_num> <checkpoint> [debug] [status] [phase_config] [perturbation_config] [prompt]"
    exit 2
fi

CONFIG_PATH="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
DEBUG_MODE="${4:-0}"
TASK_NAME="${5:-two_arms_stack_bowl_12}"
PHASE_CONFIG="${6:-script/MACG/eval_scripts_vlm/stack_bowls_phases.yaml}"
PERTURBATION_CONFIG="${7:-}"
PROMPT="${8:-stack the bowls}"

SEED_START="${SEED_START:-1000}"
SEED_END="${SEED_END:-1099}"
VLM_MODEL="${VLM_MODEL:-}"
VLM_MODE="${VLM_MODE:-phase}"
PLANNER_MODE="${PLANNER_MODE:-}"
VLM_EVERY_K="${VLM_EVERY_K:-}"
VLM_CAMERA="${VLM_CAMERA:-}"
VLM_LOG_DIR="${VLM_LOG_DIR:-tmp/vlm_logs}"
VLM_TIME_BUDGET_S="${VLM_TIME_BUDGET_S:-600}"
MAX_STEPS="${MAX_STEPS:-250}"
TIMEOUT_S="${TIMEOUT_S:-120}"
VLA_RETRIES="${VLA_RETRIES:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if (( SEED_START > SEED_END )); then
    echo "ERROR: SEED_START ($SEED_START) must be <= SEED_END ($SEED_END)."
    exit 2
fi

if [ -z "${OPENROUTER_API_KEY:-}" ] \
    && [[ "$PLANNER_MODE" != "rule_based" && "$PLANNER_MODE" != "timestep" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set."
    exit 2
fi

CONFIG_NAME="$(basename "$CONFIG_PATH" .yaml)"
PHASE_NAME="$(basename "$PHASE_CONFIG" .yaml)"
if [ -n "$PERTURBATION_CONFIG" ]; then
    EVAL_MODE="random"
else
    EVAL_MODE="clean"
fi
MODEL_LABEL="${VLM_MODEL:-yaml_default}"
PLANNER_LABEL="${PLANNER_MODE:-config_default}"
SAFE_MODEL="${MODEL_LABEL//\//_}"
LOG_FILE="eval_results_${CONFIG_NAME}_${PHASE_NAME}_${EVAL_MODE}_${PLANNER_LABEL}_${VLM_MODE}_${SAFE_MODEL}_$(date +%Y%m%d_%H%M%S).log"

{
    echo "Evaluating task: $TASK_NAME"
    echo "Config: $CONFIG_PATH"
    echo "Phase config: $PHASE_CONFIG"
    echo "Planner mode: $PLANNER_LABEL"
    echo "VLM model/mode: $MODEL_LABEL / $VLM_MODE"
    echo "Seeds: $SEED_START-$SEED_END"
    echo "Eval mode: $EVAL_MODE"
} | tee "$LOG_FILE"

TOTAL=0
SUCCESS=0
PROCESS_ERRORS=0

for ((SEED=SEED_START; SEED<=SEED_END; SEED++)); do
    CMD=(
        "$PYTHON_BIN" script/MACG/eval_scripts_vlm/eval_stackbowl_vlm.py
        "--config=$CONFIG_PATH"
        "--data-num=$DATA_NUM"
        "--checkpoint-num=$CHECKPOINT_NUM"
        "--render-mode=rgb_array"
        "-o=rgb"
        "-b=cpu"
        "-n" "1"
        "-s" "$SEED"
        "--status=$TASK_NAME"
        "--vlm-config=$PHASE_CONFIG"
        "--prompt=$PROMPT"
        "--max-steps=$MAX_STEPS"
        "--timeout-s=$TIMEOUT_S"
        "--vla-retries=$VLA_RETRIES"
        "--vlm-mode=$VLM_MODE"
        "--vlm-log-dir=$VLM_LOG_DIR"
        "--vlm-time-budget-s=$VLM_TIME_BUDGET_S"
    )
    if [ -n "$VLM_MODEL" ]; then
        CMD+=("--vlm-model=$VLM_MODEL")
    fi
    if [ -n "$PLANNER_MODE" ]; then
        CMD+=("--planner-mode=$PLANNER_MODE")
    fi
    if [ -n "$VLM_EVERY_K" ]; then
        CMD+=("--vlm-every-k=$VLM_EVERY_K")
    fi
    if [ -n "$VLM_CAMERA" ]; then
        CMD+=("--vlm-camera=$VLM_CAMERA")
    fi
    if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
        CMD+=("--quiet")
    fi
    if [ -n "$PERTURBATION_CONFIG" ]; then
        CMD+=("--perturbation-config=$PERTURBATION_CONFIG")
    fi
    if [ -n "${REPLAY_MANIFEST_FILE:-}" ]; then
        CMD+=("--replay-manifest=$REPLAY_MANIFEST_FILE")
    fi
    if [[ "${NO_GRIPPER_NUMBERS:-0}" == "1" || "${NO_GRIPPER_NUMBERS:-0}" == "true" ]]; then
        CMD+=("--no-gripper-numbers")
    fi

    echo "Running seed $SEED..."
    OUTPUT="$("${CMD[@]}" 2>&1)"
    EXIT_CODE=$?
    echo "$OUTPUT"
    if [ "$EXIT_CODE" -eq 0 ] && [ "$(printf '%s\n' "$OUTPUT" | tail -n 1)" = "success" ]; then
        RESULT=1
        SUCCESS=$((SUCCESS + 1))
    else
        RESULT=0
    fi
    if [ "$EXIT_CODE" -ne 0 ]; then
        PROCESS_ERRORS=$((PROCESS_ERRORS + 1))
    fi
    TOTAL=$((TOTAL + 1))
    SUCCESS_RATE="$(python -c "print(f'{100.0 * $SUCCESS / $TOTAL:.4f}')")"
    echo "$SEED, $RESULT, $SUCCESS_RATE%, exit=$EXIT_CODE" | tee -a "$LOG_FILE"
done

SUCCESS_RATE="$(python -c "print(f'{100.0 * $SUCCESS / $TOTAL:.4f}')")"
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" | tee -a "$LOG_FILE"
echo "Evaluation completed. Results saved in $LOG_FILE."
if [ "$PROCESS_ERRORS" -ne 0 ]; then
    echo "ERROR: $PROCESS_ERRORS evaluation process(es) exited nonzero."
    exit 1
fi
