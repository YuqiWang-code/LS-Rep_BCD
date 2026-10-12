#!/bin/bash
# CATA-CD v2 — Run3：等待某个 tmux 会话结束后，再生成下一批 cache。
#
# 用途：把 cache 生成排成链，在训练臂占用显存的同时把空闲容量用满，
# 但**不增加训练进程**（报告 §7.2 要求每卡单训练进程）。
#
# Usage:
#   bash chain_cache.sh <wait_session> <gpu_id> <teachers_csv> <datasets_csv>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

WAIT_SESSION="${1:?wait session required}"
GPU_ID="${2:?gpu id required}"
TEACHERS="${3:?teachers required}"
DATASETS="${4:?datasets required}"

echo "[chain] waiting for tmux session '$WAIT_SESSION' to finish..."
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done
echo "[chain] '$WAIT_SESSION' finished; starting $TEACHERS on $DATASETS (gpu $GPU_ID)"
bash "$HERE/build_cache.sh" "$GPU_ID" "$TEACHERS" "$DATASETS"
echo "[chain] done"
