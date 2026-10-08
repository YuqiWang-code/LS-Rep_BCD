#!/bin/bash
# CATA-CD v2 — J1 冒烟测试：用真实 teacher cache 走通四臂的**完整代码路径**
# （含 gate/taskkd 的校准加载、区域构造、辅助头、deploy/swap 校验、TEST block），
# 但只跑 SMOKE_STEPS 步，产物写到 Run2_smoke/，绝不触碰正式目录。
#
# Usage:
#   bash train_scripts/CATA-CD/Run2/smoke_j1.sh <gpu_id> [steps] [dataset]
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
STEPS="${2:-30}"
DS="${3:-SYSU}"

SMOKE_CKPT="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run2_smoke"
SMOKE_LOG="$LOG_BASE/../Run2_smoke"

echo "[smoke] steps=$STEPS dataset=$DS gpu=$GPU_ID"

for ARM in J1-C1 J1-GT J1-GATE J1-TASKKD; do
  CFG="$(exp_cfg "$ARM")"
  DCA_MODE="${CFG%% *}"
  TEACHER="$(echo "$CFG" | awk '{print $2}')"
  AUX_TASK="$(echo "$CFG" | awk '{print $3}')"

  SAVE_DIR="$SMOKE_CKPT/$ARM/$DS"
  LOG_FILE="$SMOKE_LOG/$ARM/$DS/train_log.txt"
  rm -rf "$SAVE_DIR"; rm -rf "$(dirname "$LOG_FILE")"
  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"

  CACHE_ARG=""
  CALIB_ARG=""
  if [ "$TEACHER" != "none" ]; then
    CACHE_ARG="--teacher_cache_root $CACHE_BASE/$TEACHER"
  fi
  if [ "$AUX_TASK" = "gate" ] || [ "$AUX_TASK" = "taskkd" ]; then
    CAL="$CALIB_BASE/${DS}_${TEACHER}.json"
    if [ ! -f "$CAL" ]; then
      echo "[smoke] FATAL missing calibration $CAL"; exit 1
    fi
    CALIB_ARG="--taskkd_calibration $CAL"
  fi

  echo "==================== smoke $ARM (aux=$AUX_TASK teacher=$TEACHER) ===================="
  python -m models.scripts.train \
    --experiment_id "SMOKE-$ARM" \
    --dca_mode "$DCA_MODE" --teacher_package "$TEACHER" --aux_task "$AUX_TASK" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps "$STEPS" --seed 2333 \
    --save_dir "$SAVE_DIR" --log_file "$LOG_FILE" \
    $CACHE_ARG $CALIB_ARG 2>&1 | tail -30
  echo "[smoke] $ARM exit=$?  log=$LOG_FILE"
  if [ -f "$LOG_FILE" ]; then
    echo "[smoke] --- TEST block ---"
    sed -n '/=== TEST RESULTS ===/,/=== END TEST RESULTS ===/p' "$LOG_FILE"
  fi
  echo
done

echo "[smoke] all four arms attempted; artifacts under $SMOKE_LOG"
