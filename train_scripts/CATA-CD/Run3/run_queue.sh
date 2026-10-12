#!/bin/bash
# CATA-CD v2 — Run3 通用队列：一条串行队列，每个臂可属于不同数据集。
#
# 与 run_screen.sh / run_confirm.sh 的区别：后两者要求整条队列同一数据集，
# 而"每卡 2 进程"的排期需要把不同数据集的臂混在一条队列里才能凑成相同轮次。
#
# Usage:
#   bash run_queue.sh <gpu_id> <ARM:DS[,ARM:DS...]>
# 例（每卡 2 条队列、各 2 臂，两条队列同时结束，全程严格 2 进程/卡）：
#   GPU0 A: C3L-Z:LEVIR,C3L-R-D3S:LEVIR
#   GPU0 B: C3L-S-D3S:LEVIR,C3L-R-D2:LEVIR
#   GPU1 A: C3C-Z:CDD,C3C-R-D3S:CDD
#   GPU1 B: C3C-R-D2:CDD,C3L-R-RADIO:LEVIR
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:?gpu id required}"
SPEC="${2:?queue spec required}"

IFS=',' read -r -a ITEMS <<< "$SPEC"
echo "[queue] gpu=$GPU_ID items=${ITEMS[*]}"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

for item in "${ITEMS[@]}"; do
  ARM="${item%%:*}"
  DS="${item##*:}"
  if [ "$ARM" = "$DS" ]; then
    echo "[queue] FATAL: item '$item' must be ARM:DS" >&2
    exit 1
  fi
  echo "==================== queue $ARM / $DS ===================="
  run_one "$ARM" "$DS" "$GPU_ID"
  echo "[queue] $ARM/$DS exit=$?"
done
echo "[queue] gpu=$GPU_ID done"
