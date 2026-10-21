#!/bin/bash
set -euo pipefail
# Rule-based evaluation for TwoArmsHandoverBottles with in-domain and reorder instruction modes.
# Each YAML uses mode: "rule_based" with gripper-state transitions.
#
# Usage: bash script/MACG/eval_scripts_rule_base/handover_bottles.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

EVAL_SCRIPT="$ROBOFACTORY_ROOT/script/MACG/two_arms_handover_bottle/eval_handover_bottles.sh"
SCENE_CFG="$ROBOFACTORY_ROOT/configs/table/two_arms_handover_bottles.yaml"
DATA_NUM=1
CKPT_NUM=300
DEBUG=0
YAML_DIR="$ROBOFACTORY_ROOT/script/MACG/two_arms_handover_bottle"
VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"
PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation_handover_bottles.yaml"

# --- Reproducibility: replay recorded perturbations -------------------------
# Default manifest dir is release/manifests (relative to RoboFactory root).
# Override by setting REPLAY_MANIFEST_DIR in the environment before running.
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/two_arms_handover_bottles_perturbed.json"
if [ -f "$_MANIFEST" ]; then
    export REPLAY_MANIFEST_FILE="$_MANIFEST"
    echo "[replay] Using manifest: $REPLAY_MANIFEST_FILE"
else
    echo "[replay] WARNING: manifest not found at $_MANIFEST"
    echo "  Perturbations will NOT be reproducible across machines."
    echo "  Run record_manifests.py on the original machine to generate manifests."
    unset REPLAY_MANIFEST_FILE
fi

# --- Clean evaluation (no perturbation) ---
unset REPLAY_MANIFEST_FILE

# In-domain: bottle1 first, then bottle2 (training distribution)
prepare_eval_video "$VIDEO_DIR/TwoArmsHandoverBottles-rf" "$VIDEO_DIR/TwoArmsHandoverBottles-rf-indomain-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_handover_bottles_12 "$YAML_DIR/handover_bottles_indomain.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsHandoverBottles-rf" "$VIDEO_DIR/TwoArmsHandoverBottles-rf-indomain-rule"

# Reorder: bottle2 first, then bottle1 (out-of-domain order)
prepare_eval_video "$VIDEO_DIR/TwoArmsHandoverBottles-rf" "$VIDEO_DIR/TwoArmsHandoverBottles-rf-reorder-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_handover_bottles_21 "$YAML_DIR/handover_bottles_reorder.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsHandoverBottles-rf" "$VIDEO_DIR/TwoArmsHandoverBottles-rf-reorder-rule"

# # --- Visual perturbation evaluation ---

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_handover_bottles_12 "$YAML_DIR/handover_bottles_indomain.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsHandoverBottles-rf" "$VIDEO_DIR/TwoArmsHandoverBottles-rf-indomain-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_handover_bottles_21 "$YAML_DIR/handover_bottles_reorder.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsHandoverBottles-rf" "$VIDEO_DIR/TwoArmsHandoverBottles-rf-reorder-perturbed-rule"
