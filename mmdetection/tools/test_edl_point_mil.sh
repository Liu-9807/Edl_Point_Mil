#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/test_edl_point_mil.sh [CHECKPOINT] [GPUS] [-- extra args]
# Examples:
#   bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
#   bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1
#   bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- --show
#   bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 2 -- --show --show-dir output

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x.py"
CHECKPOINT="${1:-}"
GPUS="${2:-1}"
WORK_DIR="${ROOT_DIR}/work_dirs/edl_point_mil_r50_fpn_1x"

# Parse extra args after optional "--" separator.
EXTRA_ARGS=()
if [[ $# -ge 3 ]]; then
  shift 2
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

# Validate inputs
if [[ -z "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint path is required!"
  echo "Usage: bash tools/test_edl_point_mil.sh <checkpoint_path> [gpus] [-- extra args]"
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

# Ensure work_dir exists
mkdir -p "$WORK_DIR"

if [[ "$GPUS" =~ ^[0-9]+$ ]] && [[ "$GPUS" -gt 1 ]]; then
  echo "[INFO] Launch distributed testing on $GPUS GPUs"
  bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
else
  echo "[INFO] Launch single-GPU testing"
  python tools/test.py "$CONFIG" "$CHECKPOINT" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
fi

echo "[INFO] Test completed!"
echo "[INFO] Results saved to: $WORK_DIR"
