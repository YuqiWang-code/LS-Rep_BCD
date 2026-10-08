#!/bin/bash
# CATA-CD v2 — Stage 3 v2：dataset signature v2 + teacher probes（含 P1–P5）。
# v1 的 signatures/probes 目录保持不变，v2 写到 signatures_v2/probes_v2。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/build_registry_inputs_v2.sh [gpu_id=0]
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/CATA_CD_v2"
SIG_DIR="$CACHE_BASE/signatures_v2"
PROBE_DIR="$CACHE_BASE/probes_v2"

cd "$PROJ"
mkdir -p "$SIG_DIR" "$PROBE_DIR"

DATASETS=(SYSU WHU CDD LEVIR)
TEACHERS=(sam2 dinov2 dinov3_lvd dinov3_sat remoteclip mars anysat universat radio)

echo "[stage3v2] dataset signatures (schema v2)"
for DS in "${DATASETS[@]}"; do
  if [ -f "$SIG_DIR/$DS.json" ]; then echo "[skip] signature $DS"; continue; fi
  python -m models.tools.build_dataset_signature \
    --data_root "$DATA_BASE/$DS-CD-256" --dataset_name "$DS" \
    --split train --max_samples 2000 --schema_version 2 --output "$SIG_DIR/$DS.json"
done

echo "[stage3v2] teacher probes (P1-P5, stratified)"
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

echo "[stage3v2] ALL DONE"
