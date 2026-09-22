#!/bin/bash
# FA-SCRD Run1 — generate CDD/LEVIR teacher cache (reuse the SYSU-fine-tuned teacher).
# Run once before Phase-2 training. GPU 0 (inference-only, shares with Phase-1 train).
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/FA_SCRD_DINOv3_CD"
TEACHER_CKPT="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/FA-SCRD/teacher/SYSU/teacher_checkpoint.pth"

cd "$PROJ"

for DS_FOLDER in CDD-CD-256 LEVIR-CD-256; do
  DS="${DS_FOLDER%-CD-256}"
  if [ -f "$CACHE_BASE/$DS/train/.done" ]; then
    echo "[cache] $DS already done, skip"
    continue
  fi
  echo "[cache] generating $DS"
  python -m models.tools.generate_teacher_cache \
    --teacher_ckpt "$TEACHER_CKPT" \
    --data_root "$DATA_BASE/$DS_FOLDER" \
    --dataset_name "$DS" \
    --cache_root "$CACHE_BASE" \
    --gpu_id "$GPU_ID"
  touch "$CACHE_BASE/$DS/train/.done"
done

echo "[prepare-cdd-levir] done"
