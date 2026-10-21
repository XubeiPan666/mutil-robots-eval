#!/bin/bash
set -euo pipefail
# Rule-based evaluation for TwoArmsStackBowl with all 4 instruction modes.
# Each YAML now uses mode: "rule_based" with gripper-state transitions.
#
# Usage: bash script/MACG/eval_scripts_rule_base/stack_bowls.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

EVAL_SCRIPT="$ROBOFACTORY_ROOT/script/MACG/two_arms_stack_bowl/eval_stackbowl.sh"
SCENE_CFG="$ROBOFACTORY_ROOT/configs/table/two_arms_stack_bowl.yaml"
DATA_NUM=1
CKPT_NUM=300
DEBUG=0
YAML_DIR="$ROBOFACTORY_ROOT/script/MACG/two_arms_stack_bowl"
VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"

# --- Reproducibility: replay recorded perturbations -------------------------
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/two_arms_stack_bowl_perturbed.json"
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

# In-domain: agent0 first, then agent1
prepare_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-in-domain-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_12 "$YAML_DIR/stack_bowls_indomain.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-in-domain-rule"

# Reorder: agent1 first, then agent0
prepare_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-reorder-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_21 "$YAML_DIR/stack_bowls_reorder.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-reorder-rule"

# Sync: both pick simultaneously, agent0 places first
prepare_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-sync-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_12 "$YAML_DIR/stack_bowls_sync.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-sync-rule"

# Sync-reorder: both pick simultaneously, agent1 places first
prepare_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-syncreorder-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_21 "$YAML_DIR/stack_bowls_syncreorder.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-syncreorder-rule"

# --- Visual perturbation evaluation ---
# PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation.yaml"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_12 "$YAML_DIR/stack_bowls_indomain.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-in-domain-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_21 "$YAML_DIR/stack_bowls_reorder.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-reorder-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_12 "$YAML_DIR/stack_bowls_sync.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-sync-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_stack_bowl_21 "$YAML_DIR/stack_bowls_syncreorder.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackBowl-rf" "$VIDEO_DIR/TwoArmsStackBowl-rf-syncreorder-perturbed-rule"
