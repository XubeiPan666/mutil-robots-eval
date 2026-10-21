#!/bin/bash
set -euo pipefail
# Rule-based evaluation for TwoArmsPlaceBurgerFries with all 4 instruction modes.
# Each YAML uses mode: "rule_based" with gripper-state transitions.
#
# Usage: bash script/MACG/eval_scripts_rule_base/place_burger_fries.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

EVAL_SCRIPT="$ROBOFACTORY_ROOT/script/MACG/two_arms_place_burger_fries/eval_place_burger_fries.sh"
SCENE_CFG="$ROBOFACTORY_ROOT/configs/table/two_arms_place_burger_fries.yaml"
DATA_NUM=1
CKPT_NUM=300
DEBUG=0
YAML_DIR="$ROBOFACTORY_ROOT/script/MACG/two_arms_place_burger_fries"
VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"
PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation_place_burger_fries.yaml"

# --- Reproducibility: replay recorded perturbations -------------------------
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/two_arms_place_burger_fries_perturbed.json"
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

prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-in-domain-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_12 "$YAML_DIR/place_burger_fries_indomain.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-in-domain-rule"

prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-reorder-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_21 "$YAML_DIR/place_burger_fries_reorder.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-reorder-rule"

prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-sync-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_12 "$YAML_DIR/place_burger_fries_sync.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-sync-rule"

prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-syncreorder-rule"
env SHELLOPTS=errexit bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_21 "$YAML_DIR/place_burger_fries_syncreorder.yaml"
move_eval_video "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-syncreorder-rule"

# --- Visual perturbation evaluation (background + lighting + clutter) ---

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_12 "$YAML_DIR/place_burger_fries_indomain.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-in-domain-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_21 "$YAML_DIR/place_burger_fries_reorder.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-reorder-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_12 "$YAML_DIR/place_burger_fries_sync.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-sync-perturbed-rule"

# bash "$EVAL_SCRIPT" "$SCENE_CFG" $DATA_NUM $CKPT_NUM $DEBUG two_arms_place_burger_fries_21 "$YAML_DIR/place_burger_fries_syncreorder.yaml" "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf" "$VIDEO_DIR/TwoArmsPlaceBurgerFries-rf-syncreorder-perturbed-rule"
