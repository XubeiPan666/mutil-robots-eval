#!/bin/bash
set -euo pipefail
# Rule-based evaluation for ThreeArmsPlaceObjects with in-domain, reorder, and sync instruction modes.
# Each YAML uses mode: "rule_based" with gripper-state transitions.
#
# Usage: bash script/MACG/eval_scripts_rule_base/three_arms_place_objects.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

EVAL_SCRIPT="$ROBOFACTORY_ROOT/script/MACG/three_arms_place_objects/eval_place_objects.sh"
SCENE_CFG="$ROBOFACTORY_ROOT/configs/table/three_arms_place_objects.yaml"
DATA_NUM=1
CKPT_NUM=300
DEBUG=0
YAML_DIR="$ROBOFACTORY_ROOT/script/MACG/three_arms_place_objects"
VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"
PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation_three_arms_place_objects.yaml"

# --- Reproducibility: replay recorded perturbations -------------------------
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/three_arms_place_objects_perturbed.json"
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

# In-domain: agent0->can, agent1->teabox, agent2->hamburg (training distribution)
prepare_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-indomain-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_indomain.yaml"
move_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-indomain-rule"

# Reorder: agent1 acts first, then agent0, then agent2 (out-of-domain order)
prepare_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-reorder-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_reorder.yaml"
move_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-reorder-rule"

# Sync: all 3 agents pick simultaneously, then all place simultaneously
prepare_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-sync-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_sync.yaml"
move_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-sync-rule"

# Composite
prepare_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-composite-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_composite.yaml"
move_eval_video "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-composite-rule"

# --- Visual perturbation evaluation ---

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_indomain.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-indomain-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_reorder.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-reorder-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_sync.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-sync-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_composite.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-composite-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG three_arms_place_objects_123_123 "$YAML_DIR/place_objects_composite_sync.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/ThreeArmsPlaceObjects-rf" "$VIDEO_DIR/ThreeArmsPlaceObjects-rf-composite-sync-perturbed-rule"
