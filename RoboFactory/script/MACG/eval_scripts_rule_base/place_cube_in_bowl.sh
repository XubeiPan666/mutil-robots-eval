#!/bin/bash
set -euo pipefail
# Rule-based evaluation for TwoArmsCubeInBowl.
# agent1 (right) places bowl at center, then agent0 (left) places cube inside bowl.
#
# Usage: bash script/MACG/eval_scripts_rule_base/place_cube_in_bowl.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

EVAL_SCRIPT="$ROBOFACTORY_ROOT/script/MACG/two_arms_place_cube_in_bowl/eval_place_cube_in_bowl.sh"
SCENE_CFG="$ROBOFACTORY_ROOT/configs/table/two_arms_place_cube_in_bowl.yaml"
DATA_NUM=1
CKPT_NUM=300
DEBUG=0
YAML_DIR="$ROBOFACTORY_ROOT/script/MACG/two_arms_place_cube_in_bowl"
VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"

# --- Reproducibility: replay recorded perturbations -------------------------
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/two_arms_place_cube_in_bowl_perturbed.json"
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

# In-domain: agent1 places bowl first, then agent0 places cube in bowl
prepare_eval_video "$VIDEO_DIR/TwoArmsCubeInBowl-rf" "$VIDEO_DIR/TwoArmsCubeInBowl-rf-in-domain-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_cube_in_bowl "$YAML_DIR/place_cube_in_bowl_indomain.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsCubeInBowl-rf" "$VIDEO_DIR/TwoArmsCubeInBowl-rf-in-domain-rule"

prepare_eval_video "$VIDEO_DIR/TwoArmsCubeInBowl-rf" "$VIDEO_DIR/TwoArmsCubeInBowl-rf-sync-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_cube_in_bowl "$YAML_DIR/place_cube_in_bowl_sync.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsCubeInBowl-rf" "$VIDEO_DIR/TwoArmsCubeInBowl-rf-sync-rule"

# # --- Visual perturbation evaluation ---
# PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation.yaml"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_cube_in_bowl "$YAML_DIR/place_cube_in_bowl_indomain.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsCubeInBowl-rf" "$VIDEO_DIR/TwoArmsCubeInBowl-rf-in-domain-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_cube_in_bowl "$YAML_DIR/place_cube_in_bowl_sync.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsCubeInBowl-rf" "$VIDEO_DIR/TwoArmsCubeInBowl-rf-sync-perturbed-rule"
