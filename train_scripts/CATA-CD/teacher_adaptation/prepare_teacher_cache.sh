#!/bin/bash
# CATA-CD v2 — 为单个 Teacher Package 在四数据集生成 Compact Cache v2。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh <teacher_id> [gpu_id=0]
set -euo pipefail

TEACHER_ID="${1:?usage: prepare_teacher_cache.sh <teacher_id> [gpu_id]}"
GPU_ID="${2:-0}"

PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/CATA_CD_v2"
WEIGHT_DIR="$PROJ/pre-trained_weights"

cd "$PROJ"

for DS_FOLDER in SYSU-CD-256 WHU-CD-256 CDD-CD-256 LEVIR-CD-256; do
  DS="${DS_FOLDER%-CD-256}"
  if [ -f "$CACHE_BASE/$TEACHER_ID/$DS/train/manifest.json" ]; then
    echo "[cache] $TEACHER_ID/$DS already done, skip"
    continue
  fi
  echo "[cache] generating $TEACHER_ID/$DS"
  python -m models.tools.generate_teacher_cache_v2 \
    --teacher_package "$TEACHER_ID" \
    --weight_dir "$WEIGHT_DIR" \
    --data_root "$DATA_BASE/$DS_FOLDER" \
    --dataset_name "$DS" \
    --cache_root "$CACHE_BASE" \
    --split train \
    --batch_size 8 \
    --gpu_id "$GPU_ID"
done

echo "[prepare-cache] done for $TEACHER_ID"
