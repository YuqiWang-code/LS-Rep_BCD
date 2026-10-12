#!/bin/bash
# CATA-CD v2 — Run3：增量生成 CTP-Critic 所需的 Teacher Cache v3（E/M 对称对证据）。
#
# 绝不写 v2 树；每个 (teacher, dataset) 单独生成，manifest 最后落盘。
# 每个样本：E = |LN(F_A) - LN(F_B)|，M = 0.5*(LN(F_A)+LN(F_B))，均为原生网格全通道 fp16。
#
# Usage:
#   bash train_scripts/CATA-CD/Run3/build_cache.sh <gpu_id> <teacher_csv> <dataset_csv>
# 例：
#   bash .../build_cache.sh 1 dinov3_sat LEVIR,CDD
#   bash .../build_cache.sh 0 dinov2      LEVIR,CDD
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
TEACHERS_CSV="${2:?teacher list required}"
DS_CSV="${3:?dataset list required}"

IFS=',' read -r -a TEACHERS <<< "$TEACHERS_CSV"
IFS=',' read -r -a DSS <<< "$DS_CSV"

echo "[v3cache] gpu=$GPU_ID teachers=${TEACHERS[*]} datasets=${DSS[*]}"
mkdir -p "$CACHE3"

for T in "${TEACHERS[@]}"; do
  for DS in "${DSS[@]}"; do
    OUT="$CACHE3/$T/$DS/train/manifest.json"
    if [ -f "$OUT" ]; then
      echo "[v3cache] skip (manifest exists): $OUT"
      continue
    fi
    echo "==================== $T / $DS ===================="
    python -m models.tools.generate_teacher_cache_v3 \
      --teacher_package "$T" \
      --dataset_name "$DS" \
      --data_root "$DATA_BASE/$(ds_folder "$DS")" \
      --weight_dir "$PROJ/pre-trained_weights" \
      --out_root "$CACHE3" \
      --batch_size 1 --gpu_id "$GPU_ID" 2>&1 | tail -8
    echo "[v3cache] $T/$DS exit=$?"
  done
done

echo "[v3cache] done; tree:"
find "$CACHE3" -name manifest.json | sed "s#^#  #"
