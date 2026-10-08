#!/bin/bash
# CATA-CD v2 — 生成 J1 所需的 train-only 冻结校准（review §4.3 / §7.1 step A）。
#
# 只读 train split + train teacher cache，绝不打开 val/test。
# 产出 $LOG_BASE/calibration/<DS>_<TEACHER>.json 及其 .diagnostics.json。
#
# Usage:
#   bash train_scripts/CATA-CD/Run2/run_calibration.sh [max_samples] [dataset ...]
set -uo pipefail
source "$(dirname "$0")/common.sh"

MAX_SAMPLES="${1:-512}"
shift || true
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then DATASETS=(SYSU); fi

OUT="$CALIB_BASE"
mkdir -p "$OUT"

for DS in "${DATASETS[@]}"; do
  CAL="$OUT/${DS}_${J1_TEACHER}.json"
  if [ -f "$CAL" ]; then
    echo "[skip] calibration exists: $CAL"
    continue
  fi
  if [ ! -f "$CACHE_BASE/$J1_TEACHER/$DS/train/manifest.json" ]; then
    echo "[skip] $DS: no cache for $J1_TEACHER"
    continue
  fi
  echo "===== calibration $DS / $J1_TEACHER ====="
  python -m models.tools.audit_taskkd \
    --cache_root "$CACHE_BASE/$J1_TEACHER" \
    --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --dataset_name "$DS" \
    --teacher_package "$J1_TEACHER" \
    --split train --max_samples "$MAX_SAMPLES" \
    --out "$CAL" 2>&1 | tail -12
done
echo "[calibration] done -> $OUT"
