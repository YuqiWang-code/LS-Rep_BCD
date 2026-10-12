#!/bin/bash
# CATA-CD v2 — Run3 高 σ 确认阶段（方案 §7.1）：SYSU + WHU。
#
# 只有在低 σ 筛选出现正向信号后才启动。SYSU 需 >1.35pp、WHU 需 >1.59pp 才算
# 超过噪声尺度；未超只能说“未检出”，不得声称有效或无效。
#
# 阶段预算：SYSU 3 run + WHU 3 run = 6 个 40K run ≈ 14.4 GPU·h。
#
# Usage:
#   bash train_scripts/CATA-CD/Run3/run_confirm.sh <gpu_id> [arms_csv] [dataset]
#   bash .../run_confirm.sh 0 C3S-Z,C3S-R-D3S,C3S-R-RADIO SYSU
#   bash .../run_confirm.sh 1 C3W-Z,C3W-R-MARS,C3W-R-SAM2   WHU
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
ARMS_CSV="${2:-C3S-Z,C3S-R-D3S,C3S-R-RADIO}"
DS="${3:-SYSU}"

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"

echo "[confirm] dataset=$DS gpu=$GPU_ID arms=${ARMS[*]}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

for ARM in "${ARMS[@]}"; do
  echo "==================== confirm $ARM / $DS ===================="
  run_one "$ARM" "$DS" "$GPU_ID"
  echo "[confirm] $ARM/$DS exit=$?"
done
echo "[confirm] done"
