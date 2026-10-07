#!/bin/bash
# CATA-CD v2 — 为全部 9 个 Teacher Package 在四数据集生成 Compact Cache v2（串行）。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/prepare_all_caches.sh [gpu_id=1]
set -euo pipefail

GPU_ID="${1:-1}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEACHERS=(sam2 dinov2 dinov3_lvd dinov3_sat remoteclip mars anysat universat radio)

for T in "${TEACHERS[@]}"; do
  echo "[cache-all] ===== teacher: $T (gpu $GPU_ID) ====="
  bash "$DIR/prepare_teacher_cache.sh" "$T" "$GPU_ID"
done

echo "[cache-all] ALL DONE"
