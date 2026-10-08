#!/bin/bash
# CATA-CD v2 — J1 机制判决：SYSU 四臂 40K（review §4.3）。
#
# 四臂唯一变量对照（见 common.sh 的 exp_cfg）：
#   J1-C1 / J1-GT / J1-GATE / J1-TASKKD
#
# Usage:
#   bash train_scripts/CATA-CD/Run2/run_j1_sysu.sh <gpu_id> [arms] [dataset]
#     arms    逗号分隔，默认全部四臂
#     dataset 默认 SYSU
# 例：两块卡并行
#   bash .../run_j1_sysu.sh 0 J1-C1,J1-GT   SYSU
#   bash .../run_j1_sysu.sh 1 J1-GATE,J1-TASKKD SYSU
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
ARMS_CSV="${2:-J1-C1,J1-GT,J1-GATE,J1-TASKKD}"
DS="${3:-SYSU}"

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"

echo "[j1] dataset=$DS gpu=$GPU_ID arms=${ARMS[*]}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

# 前置检查：GATE/TASKKD 需要的校准必须已生成（只读 train，不碰 val/test）。
for ARM in "${ARMS[@]}"; do
  CFG="$(exp_cfg "$ARM")"
  TEACHER="$(echo "$CFG" | awk '{print $2}')"
  AUX="$(echo "$CFG" | awk '{print $3}')"
  if [ "$AUX" = "gate" ] || [ "$AUX" = "taskkd" ]; then
    CAL="$CALIB_BASE/${DS}_${TEACHER}.json"
    if [ ! -f "$CAL" ]; then
      echo "[j1] FATAL: missing calibration $CAL"
      echo "     run: bash train_scripts/CATA-CD/Run2/run_calibration.sh 512 $DS"
      exit 1
    fi
    echo "[j1] calibration OK: $CAL"
  fi
done

for ARM in "${ARMS[@]}"; do
  echo "==================== $ARM / $DS ===================="
  run_one "$ARM" "$DS" "$GPU_ID"
  echo "[j1] $ARM/$DS finished with exit=$?"
done

echo "[j1] all requested arms done"
