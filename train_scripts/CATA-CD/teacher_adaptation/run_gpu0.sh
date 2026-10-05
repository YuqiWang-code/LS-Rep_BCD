#!/bin/bash
# CATA-CD v2 — GPU 0 队列：先 C0/C1 四数据集 anchor，再 SAM / DINOv3-SAT / MaRS 教师包。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/run_gpu0.sh [gpu_id=0]
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
cd "$PROJ"

# shellcheck disable=SC1091
source train_scripts/CATA-CD/teacher_adaptation/common.sh

DATASETS=(SYSU WHU CDD LEVIR)

# 1) 干净 anchor C0 与 C1（所有教师效用都相对 C1）
for DS in "${DATASETS[@]}"; do
  run_one C0 "$DS" "$GPU_ID"
  run_one C1 "$DS" "$GPU_ID"
done

# 2) Wave-A 教师能力矩阵（GPU0 负责）
for DS in "${DATASETS[@]}"; do
  run_one TV-SAM  "$DS" "$GPU_ID"
  run_one TV-D3S  "$DS" "$GPU_ID"
  run_one TV-MARS "$DS" "$GPU_ID"
done

echo "[run_gpu0] done"
