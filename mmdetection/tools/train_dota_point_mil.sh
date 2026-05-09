#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/train_dota_point_mil.sh [GPUS] [WORK_DIR] [-- extra args]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x_dota.py"
GPUS="${1:-1}"
WORK_DIR="${2:-work_dirs/edl_point_mil_r50_fpn_1x_dota}"

EXTRA_ARGS=()
if [[ $# -ge 3 ]]; then
  shift 2
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Config: $CONFIG"
echo "[INFO] GPUs: $GPUS"
echo "[INFO] Work dir: $WORK_DIR"
echo "[INFO] Extra args: ${EXTRA_ARGS[*]:-(none)}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Config not found: $CONFIG"
  exit 1
fi

if [[ "$GPUS" =~ ^[0-9]+$ ]] && [[ "$GPUS" -gt 1 ]]; then
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
else
  python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
fi
