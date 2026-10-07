#!/bin/bash
# CATA-CD v2 — GPU 0：先为本卡教师生成 cache，再 3-way 并行训练。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/run_gpu0.sh [gpu_id=0]
set -euo pipefail

GPU_ID="${1:-0}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/common.sh"

# 1) 本卡教师 cache（串行，fail-fast）
for T in sam2 dinov2 dinov3_lvd dinov3_sat; do
  bash "$DIR/prepare_teacher_cache.sh" "$T" "$GPU_ID"
done

# 2) 3-way 并行训练（单个 job 失败不中断整队）
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
run3 C0:SYSU C0:WHU C0:CDD
run3 C0:LEVIR TV-SAM:SYSU TV-SAM:WHU
run3 TV-SAM:CDD TV-SAM:LEVIR TV-D2:SYSU
run3 TV-D2:WHU TV-D2:CDD TV-D2:LEVIR
run3 TV-D3N:SYSU TV-D3N:WHU TV-D3N:CDD
run3 TV-D3N:LEVIR TV-D3S:SYSU TV-D3S:WHU
run3 TV-D3S:CDD TV-D3S:LEVIR

echo "[run_gpu0] ALL DONE"
