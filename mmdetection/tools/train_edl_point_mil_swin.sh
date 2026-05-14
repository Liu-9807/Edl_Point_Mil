#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/train_edl_point_mil_swin.sh [GPUS] [WORK_DIR] [-- extra args]
# Examples:
#   bash tools/train_edl_point_mil_swin.sh
#   bash tools/train_edl_point_mil_swin.sh 2
#   bash tools/train_edl_point_mil_swin.sh 1 work_dirs/edl_point_mil_swin_t_fpn_1x -- --resume auto

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="mmdet/configs/edl_point_mil/edl_point_mil_swin_t_fpn_1x.py"
GPUS="${1:-1}"
WORK_DIR="${2:-work_dirs/edl_point_mil_swin_t_fpn_1x}"

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
  echo "[INFO] Launch distributed training on $GPUS GPUs"
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
else
  echo "[INFO] Launch single-GPU training"
  python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
fi
