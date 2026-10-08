#!/bin/bash
# CATA-CD v2 — J0：同配置同 seed 的可重复性取证（review §4.2）。
#
# 不产生任何正式 40K run，不写 checkpoint，不触碰已有产物。
# 每个数据集在同一张卡上背靠背跑 REPEAT 次 ≤STEPS 步，记录：
#   初始权重 hash / 每步 batch+augmentation hash / RNG hash / loss / grad norm / 参数切片 hash
# 然后逐步比对，报告“第一次分歧出现在第几步、哪个量上”。
#
# Usage:
#   bash train_scripts/CATA-CD/Run2/run_j0.sh <gpu_id> [steps] [repeat]
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
STEPS="${2:-200}"
REPEAT="${3:-2}"
OUT_BASE="$LOG_BASE/j0"
mkdir -p "$OUT_BASE"

# 记录外部并发负载：同卡上还有谁在跑，是解释分歧的首要线索。
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$OUT_BASE/nvidia_smi_before.txt" 2>&1
ps -eo user,pid,pcpu,pmem,cmd --sort=-pmem | grep -E 'python' | grep -v grep \
  > "$OUT_BASE/processes_before.txt" 2>&1

# 可选：通过环境变量追加诊断开关（第二阶段用），默认与生产协议完全一致。
#   J0_CUBLAS_WS=":4096:8"   -> 导出 CUBLAS_WORKSPACE_CONFIG（必须在 python 启动前）
#   J0_EXTRA="--no_tf32"     -> 追加给 audit_reproducibility 的开关
EXTRA="${J0_EXTRA:-}"
if [ -n "${J0_CUBLAS_WS:-}" ]; then
  export CUBLAS_WORKSPACE_CONFIG="$J0_CUBLAS_WS"
  echo "[j0] CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
fi
if [ -n "$EXTRA" ]; then
  echo "[j0] extra flags: $EXTRA"
fi
TAGSUF="${J0_TAG:-}"

for DS in ${J0_DATASETS:-SYSU WHU}; do
  FILES=()
  for i in $(seq 1 "$REPEAT"); do
    TAG=$(printf "%s%s-%d" "$DS" "$TAGSUF" "$i")
    JSON="$OUT_BASE/${TAG}.json"
    echo "===== J0 $TAG (steps=$STEPS, gpu=$GPU_ID) ====="
    python -m models.tools.audit_reproducibility \
      --experiment_id "J0-$TAG" \
      --dataset_name "$DS" \
      --data_root "$DATA_BASE/$(ds_folder "$DS")" \
      --dca_mode moe128 --teacher_package none \
      --pretrained --pretrained_path "$PRETRAINED" \
      --gpu_id "$GPU_ID" --batch_size 64 --steps "$STEPS" --seed 2333 \
      $EXTRA \
      --out "$JSON" 2>&1 | tail -14
    FILES+=("$JSON")
  done
  echo "----- J0 $DS comparison -----"
  python -m models.tools.audit_reproducibility --compare "${FILES[@]}"
done

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$OUT_BASE/nvidia_smi_after.txt" 2>&1
echo "[j0] artifacts -> $OUT_BASE"
