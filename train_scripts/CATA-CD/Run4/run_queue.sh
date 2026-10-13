#!/bin/bash
# CATA-CD v2 — Run4 通用队列（ARM:DS 由 ID 自带，故只需列 ID）。
#
# 每卡一个串行队列；"每卡 2 进程"排期用两条队列。与前几轮不同，Run4 的 ID 自带
# 数据集与教师，所以这里只接 ID 列表。
#
# Usage:
#   bash run_queue.sh <gpu_id> <ID1,ID2,...>
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:?gpu id required}"
SPEC="${2:?id list required}"

IFS=',' read -r -a ITEMS <<< "$SPEC"
echo "[queue] gpu=$GPU_ID items=${ITEMS[*]}"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

for EXP in "${ITEMS[@]}"; do
  echo "==================== queue $EXP ===================="
  run_one "$EXP" "$GPU_ID"
  echo "[queue] $EXP exit=$?"
done
echo "[queue] gpu=$GPU_ID done"
