#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${CONDA_PREFIX:-} || ${CONDA_DEFAULT_ENV:-} == base ]]; then
  echo '请先 conda activate 安装脚本创建的环境路径，不要在 base 中运行。' >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
CONSOLE_DIR=${REPRO_CONSOLE_DIR:-"$PWD/vision-repro-logs"}
mkdir -p "$CONSOLE_DIR"
python -u "$SCRIPT_DIR/repro.py" "$@" 2>&1 |
  tee "$CONSOLE_DIR/console-$(date +%Y%m%d-%H%M%S)-$$.log"
