#!/bin/bash
# CATA-CD v2 — S-PCG 阶梯（review §5.1）：S0 / S1 / S2 单变量机制表。
#
# 唯一变量 = DCA gate 的输入（专家、width、Decoder、主损失、训练协议全部相同）：
#   S0-C1   legacy 全局 gate（与冻结 C1 同构，3,280,274 参数）
#   S1-STAT 逐尺度 gate，只加对称差分统计 GAP(|a_s-b_s|)
#   S2-PCG  S1 + 对称共性上下文 GAP((a_s+b_s)/2) 与 |GAP(a_s)-GAP(b_s)|
#
# Gate R2：S2 必须**独立优于 S1**（四域 test F1/IoU），否则拒绝该创新。
#
# Usage:
#   bash train_scripts/CATA-CD/Run2/run_spcg.sh <gpu_id> <arms_csv> <datasets_csv>
# 例：
#   bash .../run_spcg.sh 0 S0-C1,S1-STAT,S2-PCG SYSU,WHU
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
ARMS_CSV="${2:-S0-C1,S1-STAT,S2-PCG}"
DS_CSV="${3:-SYSU,WHU}"

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
IFS=',' read -r -a DSS <<< "$DS_CSV"

echo "[spcg] gpu=$GPU_ID arms=${ARMS[*]} datasets=${DSS[*]}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

for DS in "${DSS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    echo "==================== $ARM / $DS ===================="
    run_one "$ARM" "$DS" "$GPU_ID"
    echo "[spcg] $ARM/$DS exit=$?"
  done
done

echo "[spcg] all requested runs done"
