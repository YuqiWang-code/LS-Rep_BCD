#!/bin/bash
# CATA-CD v2 — Run1/M1：按 Agent（A-LODO MLP）选择，逐数据集从头训练 M1。
# Agent 选择（outputs/CATA-CD/agent/lodo_report.json）:
#   SYSU -> none, WHU -> none, CDD -> dinov3_lvd, LEVIR -> anysat
# Usage: bash train_scripts/CATA-CD/Run1/run_m1.sh <gpu_id> <DS> [<DS> ...]
set -euo pipefail

GPU_ID="${1:?usage: run_m1.sh <gpu_id> <DS>...}"; shift
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run1"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run1"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/CATA_CD_v2"
PRETRAINED="$PROJ/pre-trained_weights/lwganet_l0_e299.pth"

cd "$PROJ"

declare -A SEL=([SYSU]=none [WHU]=none [CDD]=dinov3_lvd [LEVIR]=anysat)

run_m1() {
  local DS="$1"
  local TEACHER="$2"
  local SAVE_DIR="$CKPT_BASE/$DS"
  local LOG_FILE="$LOG_BASE/$DS/train_log.txt"

  if grep -q '^=== END TEST RESULTS ===' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] M1/$DS already complete"
    return 0
  fi
  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"

  local RESUME_ARG=""
  if [ -f "$SAVE_DIR/last_checkpoint.pth" ]; then
    RESUME_ARG="--resume $SAVE_DIR/last_checkpoint.pth"
  fi
  local CACHE_ARG=""
  if [ "$TEACHER" != "none" ]; then
    CACHE_ARG="--teacher_cache_root $CACHE_BASE/$TEACHER"
  fi

  echo "[M1] $DS (teacher=$TEACHER, gpu=$GPU_ID)"
  python -m models.scripts.train \
    --experiment_id M1 \
    --dca_mode moe128 \
    --teacher_package "$TEACHER" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$DS-CD-256" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps 40000 --seed 2333 \
    --save_dir "$SAVE_DIR" \
    --log_file "$LOG_FILE" \
    $CACHE_ARG $RESUME_ARG
}

set +e
pids=()
for DS in "$@"; do
  run_m1 "$DS" "${SEL[$DS]}" &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || echo "[M1] a job failed"; done
echo "[run_m1] DONE (gpu $GPU_ID: $*)"
