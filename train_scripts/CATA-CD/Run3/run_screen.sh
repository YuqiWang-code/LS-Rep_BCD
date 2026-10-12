#!/bin/bash
# CATA-CD v2 — Run3 低 σ 筛选阶段（方案 §7.1）：LEVIR + CDD。
#
# 目的：先回答“新通道是否让**同一个域不同教师**的收益拉开”，并对照 zero 反事实。
# 阶段预算：LEVIR 4 run + CDD 3 run = 7 个 40K run ≈ 16.8 GPU·h。
#
# 判据（方案 §5.4）：LEVIR 上 REAL−ZERO 优先观察 ≥+1.0pp F1 且 ΔIoU 同向；
# 同时 REAL−SHUFFLE>0。不理想时先查 critic/梯度/空间，**不要先改 Agent**。
#
# Usage:
#   bash train_scripts/CATA-CD/Run3/run_screen.sh <gpu_id> [arms_csv] [dataset]
# 典型两卡排期：
#   bash .../run_screen.sh 0 C3L-Z,C3L-R-D3S,C3L-S-D3S,C3L-R-D2 LEVIR
#   bash .../run_screen.sh 1 C3C-Z,C3C-R-D3S,C3C-R-D2          CDD
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
ARMS_CSV="${2:-C3L-Z,C3L-R-D3S,C3L-S-D3S,C3L-R-D2}"
DS="${3:-LEVIR}"

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"

echo "[screen] dataset=$DS gpu=$GPU_ID arms=${ARMS[*]}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

for ARM in "${ARMS[@]}"; do
  echo "==================== screen $ARM / $DS ===================="
  run_one "$ARM" "$DS" "$GPU_ID"
  echo "[screen] $ARM/$DS exit=$?"
done
echo "[screen] done"
