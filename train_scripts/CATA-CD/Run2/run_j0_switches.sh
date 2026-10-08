#!/bin/bash
# CATA-CD v2 — J0 第二阶段：定位“前向一致、反向不一致”的具体来源（review §4.2）。
#
# 第一阶段（run_j0.sh）已证明：同一配置两次运行 **初始权重逐位一致**，
# 但 **第 0 步 grad_norm 就分歧**，而同一批次的 batch/RNG/loss 逐位相同。
# 也就是说：前向确定，反向不确定。
#
# 本脚本逐个打开候选开关，看哪一个能让 step-0 grad_norm 逐位一致：
#   A  baseline                  （复现，含 CUBLAS_WORKSPACE_CONFIG 未设）
#   B  CUBLAS_WORKSPACE_CONFIG   （必须在 python 启动前设置）
#   C  B + cudnn/matmul TF32 关闭
#   D  C + torch.use_deterministic_algorithms(True)（可能对不支持的算子抛错）
#
# 重要：这些是**诊断**配置。review §4.2 明确禁止把某个开关一键打开后直接拿
# 新结果与旧 C1 比较；本脚本只回答“不确定来自哪里”，不改动生产协议。
#
# Usage:
#   bash train_scripts/CATA-CD/Run2/run_j0_switches.sh <gpu_id> [steps] [dataset]
set -uo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID="${1:-0}"
STEPS="${2:-30}"
DS="${3:-SYSU}"
OUT_BASE="$LOG_BASE/j0_switches"
mkdir -p "$OUT_BASE"

DATA="$DATA_BASE/$(ds_folder "$DS")"
COMMON_ARGS="--dataset_name $DS --data_root $DATA --dca_mode moe128 --teacher_package none
             --pretrained --pretrained_path $PRETRAINED --gpu_id $GPU_ID
             --batch_size 64 --steps $STEPS --seed 2333"

run_pair() {
  local NAME="$1"; shift
  local ENV_PREFIX="$1"; shift
  local EXTRA="$1"; shift
  local FILES=()
  for i in 1 2; do
    local JSON="$OUT_BASE/${NAME}-${i}.json"
    echo "===== switch $NAME run $i ====="
    if [ -n "$ENV_PREFIX" ]; then
      env "$ENV_PREFIX" python -m models.tools.audit_reproducibility \
        --experiment_id "J0SW-$NAME-$i" $COMMON_ARGS $EXTRA --out "$JSON" 2>&1 | tail -6
    else
      python -m models.tools.audit_reproducibility \
        --experiment_id "J0SW-$NAME-$i" $COMMON_ARGS $EXTRA --out "$JSON" 2>&1 | tail -6
    fi
    FILES+=("$JSON")
  done
  echo "----- $NAME verdict -----"
  python -m models.tools.audit_reproducibility --compare "${FILES[@]}"
  echo
}

echo "########## A baseline (CUBLAS_WORKSPACE_CONFIG left unset) ##########"
run_pair "A-baseline" "" ""

echo "########## B CUBLAS_WORKSPACE_CONFIG=:4096:8 ##########"
run_pair "B-cublas-ws" "CUBLAS_WORKSPACE_CONFIG=:4096:8" ""

echo "########## C + TF32 disabled ##########"
run_pair "C-no-tf32" "CUBLAS_WORKSPACE_CONFIG=:4096:8" "--no_tf32"

echo "########## D + use_deterministic_algorithms(True) ##########"
run_pair "D-det-algos" "CUBLAS_WORKSPACE_CONFIG=:4096:8" "--no_tf32 --det_algos"

echo "[j0-switches] artifacts -> $OUT_BASE"
echo "[j0-switches] summary table:"
python - "$OUT_BASE" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
for name in ("A-baseline", "B-cublas-ws", "C-no-tf32", "D-det-algos"):
    p = base / f"{name}-1.json"
    if not p.is_file():
        print(f"  {name:14s} (missing)")
        continue
    r1 = json.loads(p.read_text())
    r2 = json.loads((base / f"{name}-2.json").read_text())
    t1, t2 = r1["trace"], r2["trace"]
    n = min(len(t1), len(t2))
    first = None
    for i in range(n):
        if abs(t1[i]["grad_norm"] - t2[i]["grad_norm"]) > 0.0:
            first = i
            break
    init_same = r1["init_hashes"]["state_dict_full"] == r2["init_hashes"]["state_dict_full"]
    print(f"  {name:14s} init_identical={init_same} "
          f"first_grad_divergence={'none' if first is None else first} "
          f"g0=({t1[0]['grad_norm']:.10f},{t2[0]['grad_norm']:.10f})")
PY
