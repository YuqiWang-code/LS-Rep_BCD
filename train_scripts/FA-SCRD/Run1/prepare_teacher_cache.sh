#!/bin/bash
# FA-SCRD Run1 — prepare teacher (fine-tune on SYSU) + generate SYSU/WHU teacher cache.
# Run ONCE before training. GPU 0.
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
DINO3_WEIGHT="$PROJ/pre-trained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/FA_SCRD_DINOv3_CD"
TEACHER_CKPT="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/FA-SCRD/teacher"

cd "$PROJ"

# 1) Fine-tune the teacher (LoRA + change head) on SYSU.
if [ ! -f "$TEACHER_CKPT/SYSU/teacher_checkpoint.pth" ]; then
  echo "[teacher] fine-tuning on SYSU"
  mkdir -p "$TEACHER_CKPT/SYSU"
  python -m models.tools.train_teacher \
    --weight_path "$DINO3_WEIGHT" \
    --data_root "$DATA_BASE/SYSU-CD-256" \
    --dataset_name SYSU \
    --save_dir "$TEACHER_CKPT/SYSU" \
    --gpu_id "$GPU_ID" --batch_size 4 --max_steps 2000 --seed 2333
else
  echo "[teacher] SYSU teacher checkpoint already exists, skip fine-tuning"
fi

# 2) Generate teacher cache for SYSU and WHU.
for DS_FOLDER in SYSU-CD-256 WHU-CD-256; do
  DS="${DS_FOLDER%-CD-256}"
  if [ -f "$CACHE_BASE/$DS/train/.done" ]; then
    echo "[cache] $DS already done, skip"
    continue
  fi
  echo "[cache] generating $DS"
  python -m models.tools.generate_teacher_cache \
    --teacher_ckpt "$TEACHER_CKPT/SYSU/teacher_checkpoint.pth" \
    --data_root "$DATA_BASE/$DS_FOLDER" \
    --dataset_name "$DS" \
    --cache_root "$CACHE_BASE" \
    --gpu_id "$GPU_ID"
  touch "$CACHE_BASE/$DS/train/.done"
done

echo "[prepare] done"
