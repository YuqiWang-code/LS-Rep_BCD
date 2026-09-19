#!/bin/bash
# RDT-CD Run3 — R3O ablation (OV-only) on GPU 0: SYSU -> WHU
# Usage: bash train_scripts/RDT-CD/Run3/run_gpu0_r3o_sysu_whu.sh [gpu_id=0]
#
# R3O = OV semantic prior only + same Fast/EMA teacher + same advantage audit.
# No BT-SAM structural prior. Runs on GPU 0 (co-tenant usually absent).
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD/Run3"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/Run3"
DATA_BASE="/share_datasets/CD"
SAM_BASE="/share_datasets/CD_teacher_cache/SAMStruct/sam2.1_hiera_large"
OV_BASE="/share_datasets/CD_teacher_cache/OVCDistill/dinov2_vitb14"
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

  # R3O still uses the paired cache path (SAM cache loaded but unused inside
  # BTSAMRDT), keeping the data pipeline identical to R3A/R3.
  python -m models.scripts.train \
    --experiment "$EXP" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$DS_FOLDER" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --sam_cache_root "$SAM_BASE/$DS_FOLDER" \
    --ov_cache_root "$OV_BASE/$DS_FOLDER" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps 40000 --seed 2333 \
    --save_dir "$SAVE_DIR" \
    --log_file "$LOG_FILE" \
    $RESUME_ARG
}

run_one R3O SYSU SYSU-CD-256
run_one R3O WHU  WHU-CD-256
