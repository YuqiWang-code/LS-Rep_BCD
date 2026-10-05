#!/bin/bash
# CATA-CD v2 — GPU 1 队列：DINOv2 / neutral DINOv3 / RemoteCLIP 教师包。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/run_gpu1.sh [gpu_id=1]
set -euo pipefail

GPU_ID="${1:-1}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
cd "$PROJ"

# shellcheck disable=SC1091
source train_scripts/CATA-CD/teacher_adaptation/common.sh

DATASETS=(SYSU WHU CDD LEVIR)

for DS in "${DATASETS[@]}"; do
  run_one TV-D2    "$DS" "$GPU_ID"
  run_one TV-D3N   "$DS" "$GPU_ID"
  run_one TV-RCLIP "$DS" "$GPU_ID"
done

echo "[run_gpu1] done"
