#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/stack_bowls.sh"
bash "$SCRIPT_DIR/stack_cubes.sh"
bash "$SCRIPT_DIR/push_cubes.sh"
bash "$SCRIPT_DIR/place_can_basket.sh"
bash "$SCRIPT_DIR/place_burger_fries.sh"
bash "$SCRIPT_DIR/place_object_cabinet.sh"
bash "$SCRIPT_DIR/handover_bottles.sh"
bash "$SCRIPT_DIR/place_cube_in_bowl.sh"
bash "$SCRIPT_DIR/place_object_bowl_cabinet.sh"
