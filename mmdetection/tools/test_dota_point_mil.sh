#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/test_dota_point_mil.sh <checkpoint_path> [GPUS] [-- extra args]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x_dota.py"
CHECKPOINT="${1:-}"
GPUS="${2:-1}"
WORK_DIR="${ROOT_DIR}/work_dirs/edl_point_mil_r50_fpn_1x_dota"

EXTRA_ARGS=()
if [[ $# -ge 3 ]]; then
  shift 2
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint path is required."
  echo "Usage: bash tools/test_dota_point_mil.sh <checkpoint_path> [GPUS] [-- extra args]"
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Config not found: $CONFIG"
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT"
  exit 1
fi

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Checkpoint: $CHECKPOINT"
echo "[INFO] GPUs: $GPUS"
echo "[INFO] Work dir: $WORK_DIR"
echo "[INFO] Extra args: ${EXTRA_ARGS[*]:-(none)}"

mkdir -p "$WORK_DIR"

if [[ "$GPUS" =~ ^[0-9]+$ ]] && [[ "$GPUS" -gt 1 ]]; then
  bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
else
  python tools/test.py "$CONFIG" "$CHECKPOINT" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
fi
