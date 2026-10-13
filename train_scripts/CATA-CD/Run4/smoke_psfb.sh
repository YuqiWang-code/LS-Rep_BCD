#!/bin/bash
# CATA-CD v2 — Run4 PSFB 冒烟：真实 v3 cache 上走通 REAL/ZERO/SHUFFLE/NOFEAT。
#
# **只查正确性**（形状/梯度/BN 隔离/部署不变式/显存），不产生机制结论，不得代替 40K。
# 产物写到 *_smoke 树，绝不触碰正式目录。
#
# Usage:
#   bash smoke_psfb.sh <gpu_id> [teacher] [dataset] [steps] [batch]
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
TEACHER="${2:-dinov2}"
DS="${3:-LEVIR}"
STEPS="${4:-6}"
BATCH="${5:-64}"

SMOKE_CKPT="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run4_smoke"
SMOKE_LOG="$LOG_BASE/../Run4_smoke"

if [ ! -f "$CACHE3/$TEACHER/$DS/train/manifest.json" ]; then
  echo "[smoke] FATAL: v3 cache missing for $TEACHER/$DS"; exit 1
fi

# expandable_segments reduces fragmentation on Blackwell; harmless for correctness.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_arm() {  # $1=tag $2=mode $3=extra
  local TAG="$1" MODE="$2" EXTRA="$3"
  local SAVE_DIR="$SMOKE_CKPT/$TAG/$DS"
  local LOG_FILE="$SMOKE_LOG/$TAG/$DS/train_log.txt"
  rm -rf "$SAVE_DIR" "$(dirname "$LOG_FILE")"
  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"
  echo "==================== smoke $TAG (mode=$MODE $EXTRA) ===================="
  python -m models.scripts.train \
    --experiment_id "SMK-$TAG" \
    --dca_mode moe128 --teacher_package "$TEACHER" --aux_task none \
    --channel_mode psfb --teacher_input_mode "$MODE" --v3_teacher "$TEACHER" \
    --teacher_cache_root "$CACHE3" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size "$BATCH" --max_steps "$STEPS" --seed 2333 --num_workers 2 \
    --save_dir "$SAVE_DIR" --log_file "$LOG_FILE" $EXTRA 2>&1 | tail -20
  echo "[smoke] $TAG exit=$?"
  [ -f "$LOG_FILE" ] && grep -E '^(Student Params|Aux Params|Train Params|Infer Params|FLOPs|Temporal swap max error|Deploy max error|Channel Mode|Teacher Input Mode|PSFB[A-Za-z ]*:|F1|IoU):' "$LOG_FILE" | tail -16
  echo
}

run_arm R-real   real    ""
run_arm Z-zero   zero    ""
run_arm S-shuf   shuffle ""
run_arm NF-nofeat real   "--psfb_no_feature"

echo "[smoke] artifacts under $SMOKE_LOG"
