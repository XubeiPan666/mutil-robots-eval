#!/usr/bin/env bash
# Top-level VLM evaluation for sequential and synchronous stack-bowl variants.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
ROBOFACTORY_ROOT="$(readlink -f "$SCRIPT_DIR/../../..")"
cd "$ROBOFACTORY_ROOT"

if [ -f "$SCRIPT_DIR/.env.local" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.local"
    set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ] \
    && [[ "${PLANNER_MODE:-}" != "rule_based" \
        && "${PLANNER_MODE:-}" != "timestep" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set."
    exit 1
fi

# Optionally route remote VLM requests through a user-configured proxy.
# Keep the local policy server off the proxy.
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

RUNNER="script/MACG/eval_scripts_vlm/stack_bowls_vlm.sh"
SCENE_CONFIG="configs/table/two_arms_stack_bowl_vlm.yaml"
PHASE_DIR="script/MACG/eval_scripts_vlm"
PERTURBATION_CONFIG="configs/perturbation/visual_perturbation.yaml"
PROMPT="${1:-stack the bowls}"

REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
MANIFEST="$REPLAY_MANIFEST_DIR/two_arms_stack_bowl_perturbed.json"
if [ -f "$MANIFEST" ]; then
    export REPLAY_MANIFEST_FILE="$MANIFEST"
    echo "[replay] Using manifest: $REPLAY_MANIFEST_FILE"
else
    unset REPLAY_MANIFEST_FILE || true
    echo "[replay] Manifest not found; perturbations will not be replay-exact."
fi

archive_video() {
    local label="$1"
    local source="eval_video/TwoArmsStackBowl-rf"
    local destination="eval_video/TwoArmsStackBowl-rf-${label}-vlm"
    if [ -d "$source" ]; then
        if [ -e "$destination" ]; then
            destination="${destination}-$(date +%Y%m%d_%H%M%S)"
        fi
        mv "$source" "$destination"
    fi
}

# Override with, for example: RUN_VARIANTS="indomain" SEED_END=1009 ...
RUN_VARIANTS="${RUN_VARIANTS:-indomain sync indomain_perturbed sync_perturbed}"
for variant in $RUN_VARIANTS; do
    case "$variant" in
        indomain)
            bash "$RUNNER" "$SCENE_CONFIG" 1 300 0 \
                two_arms_stack_bowl_12 "$PHASE_DIR/stack_bowls_phases.yaml" \
                "" "$PROMPT"
            archive_video "in-domain"
            ;;
        sync)
            bash "$RUNNER" "$SCENE_CONFIG" 1 300 0 \
                two_arms_stack_bowl_12 "$PHASE_DIR/stack_bowls_sync_phases.yaml" \
                "" "$PROMPT"
            archive_video "sync"
            ;;
        indomain_perturbed)
            bash "$RUNNER" "$SCENE_CONFIG" 1 300 0 \
                two_arms_stack_bowl_12 "$PHASE_DIR/stack_bowls_phases.yaml" \
                "$PERTURBATION_CONFIG" "$PROMPT"
            archive_video "in-domain-perturbed"
            ;;
        sync_perturbed)
            bash "$RUNNER" "$SCENE_CONFIG" 1 300 0 \
                two_arms_stack_bowl_12 "$PHASE_DIR/stack_bowls_sync_phases.yaml" \
                "$PERTURBATION_CONFIG" "$PROMPT"
            archive_video "sync-perturbed"
            ;;
        *)
            echo "ERROR: unknown RUN_VARIANTS entry: $variant"
            exit 2
            ;;
    esac
done
