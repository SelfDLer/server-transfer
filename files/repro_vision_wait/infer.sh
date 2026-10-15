#!/usr/bin/env bash
# Run each mode separately so a hang leaves its worker processes available for inspection.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-}
case "$MODE" in
  eager|compile|piecewise|graph) shift ;;
  *) echo 'Usage: bash infer.sh {eager|compile|piecewise|graph} --model /model [--video /video] [other options]' >&2; exit 2 ;;
esac
# Keep async scheduling enabled in every comparison; first run without vision probes.
exec bash "$SCRIPT_DIR/run.sh" --probe off --launch-blocking 0 --async-scheduling \
  --output "$PWD/vision-repro-logs/$MODE" "$@" --execution "$MODE"
