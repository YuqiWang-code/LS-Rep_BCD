#!/bin/bash
# CATA-CD v2 — 噪声底实验：C1 配置（dca=moe128, teacher=none, seed=2333, batch=64, 40K）
# 重复 R1/R2/R3，测量同配置 run-to-run 波动 σ。
# Usage: bash run_noise_floor.sh <gpu_id> <EXP:DS> [<EXP:DS> ...]   e.g. R1:SYSU R1:WHU ...
set -euo pipefail

GPU_ID="${1:?usage: run_noise_floor.sh <gpu_id> <EXP:DS> ...}"; shift
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/noise"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/noise"
DATA_BASE="/share_datasets/CD"
PRETRAINED="$PROJ/pre-trained_weights/lwganet_l0_e299.pth"

cd "$PROJ"

run_one() {
  local EXP="$1"
  local DS="$2"
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

  echo "[noise] $EXP/$DS (gpu=$GPU_ID)"
  python -m models.scripts.train \
    --experiment_id "$EXP" \
    --dca_mode moe128 \
    --teacher_package none \
    --dataset_name "$DS" --data_root "$DATA_BASE/$DS-CD-256" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps 40000 --seed 2333 \
    --save_dir "$SAVE_DIR" \
    --log_file "$LOG_FILE" \
    $RESUME_ARG
}

set +e
# 3-way batches（与原始 C1 的并发度一致）
batch=()
for spec in "$@"; do
  batch+=("$spec")
  if [ "${#batch[@]}" -eq 3 ]; then
    pids=()
    for s in "${batch[@]}"; do
      run_one "${s%%:*}" "${s##*:}" &
      pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || echo "[noise] a job failed"; done
    batch=()
  fi
done
if [ "${#batch[@]}" -gt 0 ]; then
  pids=()
  for s in "${batch[@]}"; do
    run_one "${s%%:*}" "${s##*:}" &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || echo "[noise] a job failed"; done
fi
echo "[run_noise_floor] DONE (gpu $GPU_ID)"
