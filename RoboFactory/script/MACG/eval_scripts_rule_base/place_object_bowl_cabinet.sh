#!/bin/bash
set -euo pipefail
# Rule-based evaluation for TwoArmsPlaceObjectBowlCabinet with in-domain and sync instruction modes.
# Each YAML uses mode: "rule_based" with gripper-state transitions.
#
# Usage: bash script/MACG/eval_scripts_rule_base/place_object_bowl_cabinet.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

EVAL_SCRIPT="$ROBOFACTORY_ROOT/script/MACG/two_arms_place_object_bowl_cabinet/eval_place_object_bowl_cabinet.sh"
SCENE_CFG="$ROBOFACTORY_ROOT/configs/table/two_arms_place_object_bowl_cabinet.yaml"
DATA_NUM=1
CKPT_NUM=300
DEBUG=0
YAML_DIR="$ROBOFACTORY_ROOT/script/MACG/two_arms_place_object_bowl_cabinet"
VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"

# --- Reproducibility: replay recorded perturbations -------------------------
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/two_arms_place_object_bowl_cabinet_perturbed.json"
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

# In-domain: agent1 opens drawer, agent0 places object, agent1 places bowl
prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf" "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf-in-domain-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_object_bowl_cabinet "$YAML_DIR/place_object_bowl_cabinet_indomain.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf" "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf-in-domain-rule"

# Sync: agent1 opens drawer, both arms simultaneously pick and place their items
prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf" "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf-sync-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_object_bowl_cabinet "$YAML_DIR/place_object_bowl_cabinet_sync.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf" "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf-sync-rule"

# # --- Visual perturbation evaluation ---
# PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation_place_object_bowl_cabinet.yaml"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_object_bowl_cabinet "$YAML_DIR/place_object_bowl_cabinet_indomain.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf" "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf-in-domain-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_object_bowl_cabinet "$YAML_DIR/place_object_bowl_cabinet_sync.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf" "$VIDEO_DIR/TwoArmsPlaceObjectBowlCabinet-rf-sync-perturbed-rule"
