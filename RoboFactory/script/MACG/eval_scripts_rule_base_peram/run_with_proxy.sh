#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <evaluation command> [args ...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOFACTORY_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROXY_HOST="${PERARM_PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PERARM_PROXY_PORT:-31888}"
PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"
ARM0_URL="${ARM0_VLA_SERVER_URL:-http://localhost:20011}"
ARM1_URL="${ARM1_VLA_SERVER_URL:-http://localhost:20012}"
ARM0_HOST="${ARM0_URL#*://}"
ARM0_HOST="${ARM0_HOST%%[:/]*}"
ARM1_HOST="${ARM1_URL#*://}"
ARM1_HOST="${ARM1_HOST%%[:/]*}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROXY_LOG="${PERARM_PROXY_LOG:-/tmp/robofactory_perarm_proxy_${PROXY_PORT}_$$.log}"
PROXY_PID=""

NO_PROXY_HOSTS="$PROXY_HOST,localhost,127.0.0.1,$ARM0_HOST,$ARM1_HOST"
export NO_PROXY="${NO_PROXY_HOSTS}${NO_PROXY:+,$NO_PROXY}"
export no_proxy="${NO_PROXY_HOSTS}${no_proxy:+,$no_proxy}"

cleanup() {
    if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

cd "$ROBOFACTORY_ROOT"

echo "[perarm] arm0: $ARM0_URL"
echo "[perarm] arm1: $ARM1_URL"
echo "[perarm] starting local proxy: $PROXY_URL"
echo "[perarm] proxy log: $PROXY_LOG"

"$PYTHON_BIN" -u "$SCRIPT_DIR/dual_arm_proxy.py" \
    --host "$PROXY_HOST" \
    --port "$PROXY_PORT" \
    >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

ready=0
for _ in {1..30}; do
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "[perarm] proxy exited during startup; inspect $PROXY_LOG" >&2
        exit 1
    fi
    if curl --fail --silent --max-time 3 "$PROXY_URL/health" >/dev/null; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "$ready" -ne 1 ]]; then
    echo "[perarm] proxy or upstream servers did not become healthy; inspect $PROXY_LOG" >&2
    exit 1
fi

echo "[perarm] both upstream servers are healthy"
set +e
VLA_SERVER_URL="$PROXY_URL" "$@"
status=$?
set -e
exit "$status"
