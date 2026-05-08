#!/usr/bin/env bash
set -euo pipefail

# Pipeline:
#   1. Train weakly supervised PointMIL model
#   2. Test the weak model and export COCO-format predictions
#   3. Convert predictions into pseudo-label COCO annotations
#   4. Train Faster R-CNN on the pseudo labels
#   5. Evaluate the pseudo-trained Faster R-CNN on the validation set
#
# Usage:
#   bash tools/run_weak_to_pseudo_faster_rcnn.sh [WEAK_GPUS] [PSEUDO_GPUS]
#
# Environment overrides:
#   WEAK_WORK_DIR        default: work_dirs/edl_point_mil_r50_fpn_1x
#   PSEUDO_WORK_DIR      default: work_dirs/faster_rcnn_pseudo_coco
#   PSEUDO_SCORE_THR     default: 0.5
#   PSEUDO_MIN_AREA      default: 1.0
#   DATA_ROOT            default: /home/user/Dataset/YouYu-JiangYong/COCO_youyu-jiangyong
#   SOURCE_ANN           default: $DATA_ROOT/annotations/instances_data.json
#   PSEUDO_ANN           default: $DATA_ROOT/annotations/pseudo_train.json
#   PRED_JSON            default: work_dirs/coco_results/predictions.bbox.json
#   SKIP_WEAK_TRAIN      set to 1 to reuse an existing weak checkpoint
#   SKIP_WEAK_TEST       set to 1 to skip weak-model inference/export
#   SKIP_CONVERT         set to 1 to skip pseudo-label conversion
#   SKIP_PSEUDO_TRAIN    set to 1 to reuse an existing pseudo-training checkpoint
#   SKIP_FINAL_TEST      set to 1 to skip final Faster R-CNN evaluation
#   WEAK_CHECKPOINT      override weak-model checkpoint path
#   PSEUDO_CHECKPOINT    override pseudo-trained Faster R-CNN checkpoint path

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

WEAK_CONFIG="mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x.py"
PSEUDO_CONFIG="configs/edl_point_mil/faster-rcnn_r50_fpn_pseudo_coco.py"

WEAK_GPUS="${1:-1}"
PSEUDO_GPUS="${2:-${WEAK_GPUS}}"

WEAK_WORK_DIR="${WEAK_WORK_DIR:-work_dirs/edl_point_mil_r50_fpn_1x}"
PSEUDO_WORK_DIR="${PSEUDO_WORK_DIR:-work_dirs/faster_rcnn_pseudo_coco}"

DATA_ROOT="${DATA_ROOT:-/home/user/Dataset/YouYu-JiangYong/COCO_youyu-jiangyong}"
SOURCE_ANN="${SOURCE_ANN:-${DATA_ROOT}/annotations/instances_data.json}"
PSEUDO_ANN="${PSEUDO_ANN:-${DATA_ROOT}/annotations/pseudo_train.json}"
PRED_JSON="${PRED_JSON:-${ROOT_DIR}/work_dirs/coco_results/predictions.bbox.json}"
PSEUDO_SCORE_THR="${PSEUDO_SCORE_THR:-0.5}"
PSEUDO_MIN_AREA="${PSEUDO_MIN_AREA:-1.0}"

SKIP_WEAK_TRAIN="${SKIP_WEAK_TRAIN:-0}"
SKIP_WEAK_TEST="${SKIP_WEAK_TEST:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_PSEUDO_TRAIN="${SKIP_PSEUDO_TRAIN:-0}"
SKIP_FINAL_TEST="${SKIP_FINAL_TEST:-0}"

WEAK_CHECKPOINT="${WEAK_CHECKPOINT:-}"
PSEUDO_CHECKPOINT="${PSEUDO_CHECKPOINT:-}"

log() {
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

latest_checkpoint() {
  local work_dir="$1"
  if [[ -f "$work_dir/latest.pth" ]]; then
    echo "$work_dir/latest.pth"
    return 0
  fi

  local latest
  latest="$(ls -1t "$work_dir"/epoch_*.pth "$work_dir"/*.pth 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest" ]]; then
    echo "$latest"
    return 0
  fi

  return 1
}

ensure_file() {
  local path="$1"
  local name="$2"
  [[ -f "$path" ]] || die "$name not found: $path"
}

run_train() {
  local config="$1"
  local gpus="$2"
  local work_dir="$3"

  mkdir -p "$work_dir"
  if [[ "$gpus" -gt 1 ]]; then
    log "Launch distributed training: $config on $gpus GPUs"
    bash tools/dist_train.sh "$config" "$gpus" --work-dir "$work_dir"
  else
    log "Launch single-GPU training: $config"
    python tools/train.py "$config" --work-dir "$work_dir"
  fi
}

run_test() {
  local config="$1"
  local checkpoint="$2"
  local gpus="$3"
  local work_dir="$4"

  mkdir -p "$work_dir"
  if [[ "$gpus" -gt 1 ]]; then
    log "Launch distributed testing: $config on $gpus GPUs"
    bash tools/dist_test.sh "$config" "$checkpoint" "$gpus" --work-dir "$work_dir"
  else
    log "Launch single-GPU testing: $config"
    python tools/test.py "$config" "$checkpoint" --work-dir "$work_dir"
  fi
}

log "Root: $ROOT_DIR"
log "Weak config: $WEAK_CONFIG"
log "Pseudo config: $PSEUDO_CONFIG"
log "Weak GPUs: $WEAK_GPUS"
log "Pseudo GPUs: $PSEUDO_GPUS"
log "Weak work dir: $WEAK_WORK_DIR"
log "Pseudo work dir: $PSEUDO_WORK_DIR"
log "Data root: $DATA_ROOT"
log "Source ann: $SOURCE_ANN"
log "Pseudo ann: $PSEUDO_ANN"
log "Pred json: $PRED_JSON"
log "Pseudo score thr: $PSEUDO_SCORE_THR"
log "Pseudo min area: $PSEUDO_MIN_AREA"

is_int "$WEAK_GPUS" || die "WEAK_GPUS must be an integer, got: $WEAK_GPUS"
is_int "$PSEUDO_GPUS" || die "PSEUDO_GPUS must be an integer, got: $PSEUDO_GPUS"

ensure_file "$WEAK_CONFIG" "Weak config"
ensure_file "$PSEUDO_CONFIG" "Pseudo config"

if [[ "$SKIP_WEAK_TRAIN" != "1" ]]; then
  run_train "$WEAK_CONFIG" "$WEAK_GPUS" "$WEAK_WORK_DIR"
else
  log "Skip weak training"
fi

if [[ -z "$WEAK_CHECKPOINT" ]]; then
  WEAK_CHECKPOINT="$(latest_checkpoint "$WEAK_WORK_DIR")"
fi
ensure_file "$WEAK_CHECKPOINT" "Weak checkpoint"
log "Weak checkpoint: $WEAK_CHECKPOINT"

if [[ "$SKIP_WEAK_TEST" != "1" ]]; then
  run_test "$WEAK_CONFIG" "$WEAK_CHECKPOINT" "$WEAK_GPUS" "$WEAK_WORK_DIR"
else
  log "Skip weak testing/export"
fi

if [[ "$SKIP_CONVERT" != "1" ]]; then
  ensure_file "$PRED_JSON" "Prediction json"
  ensure_file "$SOURCE_ANN" "Source annotation"
  log "Converting pseudo labels"
  python tools/dataset_converters/convert_pseudo_coco_results.py \
    --pred-json "$PRED_JSON" \
    --source-ann "$SOURCE_ANN" \
    --out-ann "$PSEUDO_ANN" \
    --score-thr "$PSEUDO_SCORE_THR" \
    --min-area "$PSEUDO_MIN_AREA"
else
  log "Skip pseudo-label conversion"
fi

if [[ "$SKIP_PSEUDO_TRAIN" != "1" ]]; then
  run_train "$PSEUDO_CONFIG" "$PSEUDO_GPUS" "$PSEUDO_WORK_DIR"
else
  log "Skip pseudo training"
fi

if [[ -z "$PSEUDO_CHECKPOINT" ]]; then
  PSEUDO_CHECKPOINT="$(latest_checkpoint "$PSEUDO_WORK_DIR")"
fi
ensure_file "$PSEUDO_CHECKPOINT" "Pseudo checkpoint"
log "Pseudo checkpoint: $PSEUDO_CHECKPOINT"

if [[ "$SKIP_FINAL_TEST" != "1" ]]; then
  run_test "$PSEUDO_CONFIG" "$PSEUDO_CHECKPOINT" "$PSEUDO_GPUS" "$PSEUDO_WORK_DIR"
else
  log "Skip final evaluation"
fi

log "Pipeline completed"
log "Pseudo labels: $PSEUDO_ANN"
log "Weak outputs: $WEAK_WORK_DIR"
log "Pseudo outputs: $PSEUDO_WORK_DIR"
