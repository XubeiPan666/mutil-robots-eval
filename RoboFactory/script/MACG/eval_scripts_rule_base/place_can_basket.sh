#!/bin/bash
set -euo pipefail
# Rule-based evaluation for TwoArmsPlaceCanBasket.
#
# Usage: bash script/MACG/eval_scripts_rule_base/place_can_basket.sh
#   (can be run from any directory)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ROBOFACTORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROBOFACTORY_ROOT" || exit 1

VIDEO_DIR="$ROBOFACTORY_ROOT/eval_video"
PERTURBATION_CFG="$ROBOFACTORY_ROOT/configs/perturbation/visual_perturbation.yaml"

# --- Reproducibility: replay recorded perturbations -------------------------
REPLAY_MANIFEST_DIR="${REPLAY_MANIFEST_DIR:-release/manifests}"
_MANIFEST="${REPLAY_MANIFEST_DIR}/two_arms_place_can_basket_perturbed.json"
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

prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf" "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf-indomain-rule"
env SHELLOPTS=errexit bash script/MACG/two_arms_place_can_basket/eval_place_can_basket.sh configs/table/two_arms_place_can_basket.yaml 1 300 0 two_arms_place_can_basket script/MACG/two_arms_place_can_basket/place_can_basket_indomain.yaml
move_eval_video "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf" "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf-indomain-rule"

prepare_eval_video "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf" "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf-sync-rule"
env SHELLOPTS=errexit bash script/MACG/two_arms_place_can_basket/eval_place_can_basket.sh configs/table/two_arms_place_can_basket.yaml 1 300 0 two_arms_place_can_basket script/MACG/two_arms_place_can_basket/place_can_basket_sync.yaml
move_eval_video "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf" "$VIDEO_DIR/TwoArmsPlaceCanBasket-rf-sync-rule"
