#!/bin/bash
# CATA-CD v2 — smoke：学生 C0/C1 参数/交换/部署 校验 + 可选单教师 8-sample cache 冒烟。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/smoke_test.sh [teacher_id]
set -euo pipefail

PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
cd "$PROJ"

python -m models.tools.smoke_cata_v2 --dca none
python -m models.tools.smoke_cata_v2 --dca moe128

if [ "${1:-}" != "" ]; then
  TEACHER_ID="$1"
  echo "[smoke] teacher $TEACHER_ID build + encode"
  python -m models.tools.smoke_cata_v2 --dca moe128 --teacher "$TEACHER_ID" --weight_dir "$PROJ/pre-trained_weights"
fi

echo "[smoke] ALL PASSED"
