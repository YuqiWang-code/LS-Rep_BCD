#!/bin/bash
# SAGE-CD Run1 — generate full-res DINO semantic cache for SYSU + WHU.
# Reuses the FA-SCRD fine-tuned DINOv3 teacher. Run once before training.
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/SAGE_DINO3_CD"
TEACHER_CKPT="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/FA-SCRD/teacher/SYSU/teacher_checkpoint.pth"

cd "$PROJ"

for DS_FOLDER in SYSU-CD-256 WHU-CD-256; do
  DS="${DS_FOLDER%-CD-256}"
  if [ -f "$CACHE_BASE/$DS/train/.done" ]; then
    echo "[sage-cache] $DS already done, skip"
    continue
  fi
  echo "[sage-cache] generating $DS"
  python -m models.tools.generate_sage_cache \
    --teacher_ckpt "$TEACHER_CKPT" \
    --data_root "$DATA_BASE/$DS_FOLDER" \
    --dataset_name "$DS" \
    --cache_root "$CACHE_BASE" \
    --gpu_id "$GPU_ID"
  touch "$CACHE_BASE/$DS/train/.done"
done

echo "[prepare-sage] done"
