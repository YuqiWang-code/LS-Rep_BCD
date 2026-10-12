#!/bin/bash
# CATA-CD v2 — Run3 CTP 冒烟：用**真实完整 cache**走通 zero/real/shuffle 三个模式。
#
# 只检查正确性（形状/梯度/日志/部署不变式），**不产生机制结论**。
# 产物写到 *_smoke 树，绝不触碰正式目录。
#
# Usage:
#   bash train_scripts/CATA-CD/Run3/smoke_ctp.sh <gpu_id> [teacher] [dataset] [steps]
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
TEACHER="${2:-dinov2}"
DS="${3:-LEVIR}"
STEPS="${4:-5}"
# Smoke may use a smaller batch: this script only checks correctness (shapes,
# gradients, logging, deploy invariants). Real 40K arms always use batch 64 via
# common.sh. A reduced batch here is NOT a mechanism result.
BATCH="${5:-64}"

SMOKE_CKPT="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run3_smoke"
SMOKE_LOG="$LOG_BASE/../Run3_smoke"

if [ ! -f "$CACHE3/$TEACHER/$DS/train/manifest.json" ]; then
  echo "[smoke] FATAL: v3 cache missing for $TEACHER/$DS"
  echo "        run: bash train_scripts/CATA-CD/Run3/build_cache.sh $GPU_ID $TEACHER $DS"
  exit 1
fi

# expandable_segments meaningfully reduces fragmentation on Blackwell; harmless.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

declare -A ARM_FOR=( [zero]=C3-Z-SMOKE [real]=C3-R-SMOKE [shuffle]=C3-S-SMOKE )

for MODE in zero real shuffle; do
  ARM="${ARM_FOR[$MODE]}"
  SAVE_DIR="$SMOKE_CKPT/$ARM/$DS"
  LOG_FILE="$SMOKE_LOG/$ARM/$DS/train_log.txt"
  rm -rf "$SAVE_DIR" "$(dirname "$LOG_FILE")"
  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"

  echo "==================== smoke $MODE ($TEACHER / $DS) ===================="
  python -m models.scripts.train \
    --experiment_id "$ARM" \
    --dca_mode moe128 --teacher_package "$TEACHER" --aux_task none \
    --channel_mode ctp --teacher_input_mode "$MODE" --v3_teacher "$TEACHER" \
    --teacher_cache_root "$CACHE3" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size "$BATCH" --max_steps "$STEPS" --seed 2333 --num_workers 2 \
    --save_dir "$SAVE_DIR" --log_file "$LOG_FILE" 2>&1 | tail -24
  echo "[smoke] $MODE exit=$?"
  if [ -f "$LOG_FILE" ]; then
    echo "--- CTP / params / invariants ---"
    grep -E '^(Student Params|Aux Params|Train Params|Infer Params|FLOPs|Temporal swap max error|Deploy max error|Channel Mode|Teacher Input Mode|F1|IoU|CTP):' "$LOG_FILE" | tail -14
  fi
  echo
done

echo "[smoke] artifacts under $SMOKE_LOG"
