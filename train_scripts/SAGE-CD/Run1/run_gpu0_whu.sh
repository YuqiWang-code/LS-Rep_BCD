#!/bin/bash
# SAGE-CD Run1 — GPU 0 queue: WHU (G0 -> G1 -> G2)
# Usage: bash train_scripts/SAGE-CD/Run1/run_gpu0_whu.sh [gpu_id=0]
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SAGE-CD/Run1"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run1"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/SAGE_DINO3_CD"
PRETRAINED="$PROJ/pre-trained_weights/lwganet_l0_e299.pth"

cd "$PROJ"

run_one() {
  local EXP="$1"
  local DS="$2"
  local DS_FOLDER="$3"

  local SAVE_DIR="$CKPT_BASE/$EXP/$DS"
  local LOG_FILE="$LOG_BASE/$EXP/$DS/train_log.txt"

  if grep -q '^=== END TEST RESULTS ===' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] $EXP/$DS already complete"
    return 0
  fi

  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"

  local RESUME_ARG=""
  if [ -f "$SAVE_DIR/last_checkpoint.pth" ]; then
    RESUME_ARG="--resume $SAVE_DIR/last_checkpoint.pth"
  fi

  local CACHE_ARG=""
  if [ "$EXP" = "G2" ]; then
    CACHE_ARG="--cache_root $CACHE_BASE"
  fi

  python -m models.scripts.train_sage \
    --experiment "$EXP" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$DS_FOLDER" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps 40000 --seed 2333 \
    --save_dir "$SAVE_DIR" \
    --log_file "$LOG_FILE" \
    $CACHE_ARG $RESUME_ARG
}

run_one G0 WHU WHU-CD-256
run_one G1 WHU WHU-CD-256
run_one G2 WHU WHU-CD-256
