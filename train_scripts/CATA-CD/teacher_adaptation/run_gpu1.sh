#!/bin/bash
# CATA-CD v2 — GPU 1：先为本卡教师生成 cache，再 3-way 并行训练。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/run_gpu1.sh [gpu_id=1]
set -euo pipefail

GPU_ID="${1:-1}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/common.sh"

# 1) 本卡教师 cache（串行，fail-fast）
for T in remoteclip mars anysat universat radio; do
  bash "$DIR/prepare_teacher_cache.sh" "$T" "$GPU_ID"
done

# 2) 3-way 并行训练
run3() {
  local pids=()
  local spec EXP DS
  for spec in "$@"; do
    EXP="${spec%%:*}"
    DS="${spec##*:}"
    run_one "$EXP" "$DS" "$GPU_ID" &
    pids+=("$!")
  done
  local rc=0 p
  for p in "${pids[@]}"; do
    if ! wait "$p"; then rc=1; echo "[parallel] a job failed (gpu $GPU_ID)"; fi
  done
  return "$rc"
}

set +e
run3 C1:SYSU C1:WHU C1:CDD
run3 C1:LEVIR TV-RCLIP:SYSU TV-RCLIP:WHU
run3 TV-RCLIP:CDD TV-RCLIP:LEVIR TV-MARS:SYSU
run3 TV-MARS:WHU TV-MARS:CDD TV-MARS:LEVIR
run3 TV-ANYSAT:SYSU TV-ANYSAT:WHU TV-ANYSAT:CDD
run3 TV-ANYSAT:LEVIR TV-UNISAT:SYSU TV-UNISAT:WHU
run3 TV-UNISAT:CDD TV-UNISAT:LEVIR TV-RADIO:SYSU
run3 TV-RADIO:WHU TV-RADIO:CDD TV-RADIO:LEVIR

echo "[run_gpu1] ALL DONE"
