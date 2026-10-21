#!/bin/bash
set -euo pipefail
# Rule-based evaluation for TwoArmsStackCube with visual perturbation.
#
# Usage: bash script/MACG/eval_scripts_rule_base/stack_cubes.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"

# --- Clean evaluation (no perturbation) ---
unset REPLAY_MANIFEST_FILE
prepare_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-in-domain-pi0"
env SHELLOPTS=errexit bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_12 script/MACG/two_arms_stack_cube/stack_cubes_indomain.yaml
move_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-in-domain-pi0"
prepare_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-reorder-pi0"
env SHELLOPTS=errexit bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_21 script/MACG/two_arms_stack_cube/stack_cubes_reorder.yaml
move_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-reorder-pi0"

prepare_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-sync-pi0"
env SHELLOPTS=errexit bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_12 script/MACG/two_arms_stack_cube/stack_cubes_sync.yaml
move_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-sync-pi0"

prepare_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-syncreorder-pi0"
env SHELLOPTS=errexit bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_21 script/MACG/two_arms_stack_cube/stack_cubes_syncreorder.yaml
move_eval_video "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-syncreorder-pi0"

# --- Reproducibility: replay recorded perturbations -------------------------
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/two_arms_stack_cube_perturbed.json"
if [ -f "$_MANIFEST" ]; then
    export REPLAY_MANIFEST_FILE="$_MANIFEST"
    echo "[replay] Using manifest: $REPLAY_MANIFEST_FILE"
else
    echo "[replay] WARNING: manifest not found at $_MANIFEST"
    echo "  Perturbations will NOT be reproducible across machines."
    echo "  Run record_manifests.py on the original machine to generate manifests."
    unset REPLAY_MANIFEST_FILE
fi

# --- Visual perturbation evaluation ---
# PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation_stackcube.yaml"
# bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_12 script/MACG/two_arms_stack_cube/stack_cubes_indomain.yaml "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-in-domain-perturbed-pi0"
# bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_21 script/MACG/two_arms_stack_cube/stack_cubes_reorder.yaml "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-reorder-perturbed-pi0"
# bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_21 script/MACG/two_arms_stack_cube/stack_cubes_syncreorder.yaml "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-syncreorder-perturbed-pi0"
# bash script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_12 script/MACG/two_arms_stack_cube/stack_cubes_sync.yaml "$PERTURBATION_CFG"
# mv "$VIDEO_DIR/TwoArmsStackCube-rf" "$VIDEO_DIR/TwoArmsStackCube-rf-sync-perturbed-pi0"
