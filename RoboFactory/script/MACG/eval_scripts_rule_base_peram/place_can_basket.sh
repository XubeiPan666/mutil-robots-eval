#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_with_proxy.sh" \
    bash script/MACG/eval_scripts_rule_base/place_can_basket.sh
