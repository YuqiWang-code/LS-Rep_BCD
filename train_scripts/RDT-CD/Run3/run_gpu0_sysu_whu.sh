#!/bin/bash
# RDT-CD Run3 — GPU 0 queue: SYSU + WHU (B0 -> R3A -> R3 for each dataset)
# Usage: bash train_scripts/RDT-CD/Run3/run_gpu0_sysu_whu.sh [gpu_id=0]
#
# Phase 1 (primary):  Validate Teacher effectiveness on SYSU / WHU before
#                     expanding to CDD / LEVIR (run_gpu1_cdd_levir.sh).
#
# Checks before launching:
#   1. synthetic smoke  : python models/tools/smoke_dynamic_teacher.py --device cuda --gpu_id 0
#   2. cache validation : python models/tools/validate_teacher_cache.py (per dataset)
#   3. dry run          : R3/SYSU with --max_steps 20
#   4. deploy check     : python models/tools/export_deploy.py
set -euo pipefail

GPU_ID="${1:-1}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD/Run3"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/Run3"
DATA_BASE="/share_datasets/CD"
SAM_BASE="/share_datasets/CD_teacher_cache/SAMStruct/sam2.1_hiera_large"
OV_BASE="/share_datasets/CD_teacher_cache/OVCDistill/dinov2_vitb14"
PRETRAINED="$PROJ/pre-trained_weights/lwganet_l0_e299.pth"

cd "$PROJ"

# ---------------------------------------------------------------------------
# Helper: run one experiment / dataset combination.
#
# Arguments:
#   $1  experiment id  (B0 | R3A | R3)
#   $2  dataset name   (SYSU | WHU | CDD | LEVIR)
#   $3  dataset folder (e.g. SYSU-CD-256)
# ---------------------------------------------------------------------------
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

  # B0 requires no cache args; R3A and R3 require both SAM and OV cache.
  if [ "$EXP" = "B0" ]; then
    python -m models.scripts.train \
      --experiment "$EXP" \
      --dataset_name "$DS" --data_root "$DATA_BASE/$DS_FOLDER" \
      --pretrained --pretrained_path "$PRETRAINED" \
      --gpu_id "$GPU_ID" --batch_size 64 --max_steps 40000 --seed 2333 \
      --save_dir "$SAVE_DIR" \
      --log_file "$LOG_FILE" \
      $RESUME_ARG
  else
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
  fi
}

# ---------------------------------------------------------------------------
# SYSU:  B0 -> R3A -> R3
# ---------------------------------------------------------------------------
run_one B0  SYSU  SYSU-CD-256
run_one R3A SYSU  SYSU-CD-256
run_one R3  SYSU  SYSU-CD-256

# ---------------------------------------------------------------------------
# WHU:   B0 -> R3A -> R3
# ---------------------------------------------------------------------------
run_one B0  WHU  WHU-CD-256
run_one R3A WHU  WHU-CD-256
run_one R3  WHU  WHU-CD-256
