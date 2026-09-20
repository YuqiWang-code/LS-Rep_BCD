#!/bin/bash
# SCTC Run1 — GPU 0 queue: LEVIR + CDD (S0 -> S1 -> S2 for each dataset)
# Usage: bash train_scripts/SCTC/Run1/run_gpu0_levir_cdd.sh [gpu_id=0]
#
# Phase 2: expand to LEVIR / CDD only after Phase 1 (run_gpu0_whu_sysu.sh)
# passes the effectiveness gate (WHU Delta_F1 >= +0.30 and no failure
# criterion triggered). Do NOT launch this script before that gate.
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SCTC/Run1"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SCTC/Run1"
DATA_BASE="/share_datasets/CD"
PRETRAINED="$PROJ/pre-trained_weights/lwganet_l0_e299.pth"

cd "$PROJ"

# ---------------------------------------------------------------------------
# Helper: run one experiment / dataset combination.
#
# Arguments:
#   $1  experiment id  (S0 | S1 | S2)
#   $2  dataset name   (SYSU | WHU | CDD | LEVIR)
#   $3  dataset folder (e.g. LEVIR-CD-256)
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

  python -m models.scripts.train \
    --experiment "$EXP" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$DS_FOLDER" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps 40000 --seed 2333 \
    --save_dir "$SAVE_DIR" \
    --log_file "$LOG_FILE" \
    $RESUME_ARG
}

# ---------------------------------------------------------------------------
# LEVIR: S0 -> S1 -> S2
# ---------------------------------------------------------------------------
run_one S0 LEVIR LEVIR-CD-256
run_one S1 LEVIR LEVIR-CD-256
run_one S2 LEVIR LEVIR-CD-256

# ---------------------------------------------------------------------------
# CDD:   S0 -> S1 -> S2
# ---------------------------------------------------------------------------
run_one S0 CDD CDD-CD-256
run_one S1 CDD CDD-CD-256
run_one S2 CDD CDD-CD-256
