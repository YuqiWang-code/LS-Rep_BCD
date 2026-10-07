#!/bin/bash
# CATA-CD v2 — Stage 3：计算 Agent 输入（dataset signature + teacher train-only probe）。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/build_registry_inputs.sh [gpu_id=0]
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/CATA_CD_v2"
SIG_DIR="$CACHE_BASE/signatures"
PROBE_DIR="$CACHE_BASE/probes"

cd "$PROJ"
mkdir -p "$SIG_DIR" "$PROBE_DIR"

DATASETS=(SYSU WHU CDD LEVIR)
TEACHERS=(sam2 dinov2 dinov3_lvd dinov3_sat remoteclip mars anysat universat radio)

echo "[stage3] dataset signatures"
for DS in "${DATASETS[@]}"; do
  if [ -f "$SIG_DIR/$DS.json" ]; then echo "[skip] signature $DS"; continue; fi
  python -m models.tools.build_dataset_signature \
    --data_root "$DATA_BASE/$DS-CD-256" --dataset_name "$DS" \
    --split train --max_samples 2000 --output "$SIG_DIR/$DS.json"
done

echo "[stage3] teacher probes"
for T in "${TEACHERS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    OUT="$PROBE_DIR/${DS}_${T}.json"
    if [ -f "$OUT" ]; then echo "[skip] probe $DS/$T"; continue; fi
    python -m models.tools.probe_teacher_capability \
      --cache_root "$CACHE_BASE/$T" --data_root "$DATA_BASE/$DS-CD-256" \
      --dataset_name "$DS" --teacher_package "$T" --split train \
      --max_samples 256 --output "$OUT"
  done
done

echo "[stage3] ALL DONE"
