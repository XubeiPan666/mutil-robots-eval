#!/usr/bin/env bash
# Convenience alias for the documented evaluation launcher.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/evaluate.sh" "$@"
