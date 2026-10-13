#!/bin/bash
# CATA-CD v2 — Run4（第四轮：PSFB 训练期特权空间特征桥）共享环境与 run_one 助手。
#
# 方案：docs/temporary/CATA-CD v2 第四轮完整中文研究报告.md
# 机制：在 DCA 后 Decoder 前的 c3 上插入训练期残差旁路，教师 v3 cache 的 E/M 经 P 得到 U，
#       经共享 Decoder 的**影子前向**（functional_call + 克隆 BN buffers）受 GT 监督，
#       并用 64 通道特征迁移把 teacher-free 视图拉向 teacher-conditioned 视图。
#       主预测路径完全不接教师；部署仍 3,280,274 参数 / 3.2432G FLOPs。
#
# 实验 ID 解析（不逐个枚举，避免 20+ 条 case）：
#   D4-<DS>-<TEACHER>-<MODE>
#     DS      : L=LEVIR  C=CDD  S=SYSU  W=WHU
#     TEACHER : D3S=dinov3_sat  D2=dinov2  RAD=radio  MARS=mars  SAM2=sam2  D3L=dinov3_lvd
#     MODE    : Z=zero  R=real  S=shuffle  NF=real 且 lambda_f=0（NOFEAT 对照）
#   例：D4-L-D3S-Z  D4-L-D3S-NF  D4-S-RAD-S

PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run4"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run4"
DATA_BASE="/share_datasets/CD"
CACHE3="/share_datasets/CD_teacher_cache/CATA_CD_v3"
PRETRAINED="$PROJ/pre-trained_weights/lwganet_l0_e299.pth"

ds_folder() {
  case "$1" in
    SYSU) echo "SYSU-CD-256" ;;
    WHU) echo "WHU-CD-256" ;;
    CDD) echo "CDD-CD-256" ;;
    LEVIR) echo "LEVIR-CD-256" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

_exp_ds() {
  case "$1" in
    L) echo "LEVIR" ;; C) echo "CDD" ;; S) echo "SYSU" ;; W) echo "WHU" ;;
    LEVIR|CDD|SYSU|WHU) echo "$1" ;;
    *) echo "unknown dataset code $1" >&2; exit 1 ;;
  esac
}

_exp_teacher() {
  case "$1" in
    D3S) echo "dinov3_sat" ;; D2) echo "dinov2" ;; D3L) echo "dinov3_lvd" ;;
    RAD) echo "radio" ;; MARS) echo "mars" ;; SAM2) echo "sam2" ;;
    *) echo "unknown teacher code $1" >&2; exit 1 ;;
  esac
}

_exp_mode() {
  case "$1" in
    Z) echo "zero" ;; R|NF) echo "real" ;; S) echo "shuffle" ;;
    *) echo "unknown mode code $1" >&2; exit 1 ;;
  esac
}

# exp id -> "dataset teacher mode extra_flags"
exp_cfg() {
  local EXP="$1"
  local P DS T M
  P="$(echo "$EXP" | awk -F'-' '{print $1}')"
  DS="$(echo "$EXP" | awk -F'-' '{print $2}')"
  T="$(echo "$EXP" | awk -F'-' '{print $3}')"
  M="$(echo "$EXP" | awk -F'-' '{print $4}')"
  if [ "$P" != "D4" ] || [ -z "$M" ]; then
    echo "bad Run4 experiment id '$EXP'; expected D4-<DS>-<TEACHER>-<MODE>" >&2
    exit 1
  fi
  local EXTRA=""
  [ "$M" = "NF" ] && EXTRA="--psfb_no_feature"
  echo "$(_exp_ds "$DS") $(_exp_teacher "$T") $(_exp_mode "$M") $EXTRA"
}

run_one() {
  local EXP="$1"
  local GPU_ID="$2"

  local TAG="${RUN_TAG:-}"
  local STEPS="${MAX_STEPS_OVERRIDE:-40000}"

  local CFG DS TEACHER MODE EXTRA
  CFG="$(exp_cfg "$EXP")"
  DS="$(echo "$CFG" | awk '{print $1}')"
  TEACHER="$(echo "$CFG" | awk '{print $2}')"
  MODE="$(echo "$CFG" | awk '{print $3}')"
  EXTRA="$(echo "$CFG" | cut -d' ' -f4-)"

  local SAVE_DIR="$CKPT_BASE$TAG/$EXP/$DS"
  local LOG_FILE="$LOG_BASE$TAG/$EXP/$DS/train_log.txt"

  if [ "$TAG" = "" ] && grep -q '^=== END TEST RESULTS ===' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] $EXP/$DS already complete"
    return 0
  fi
  if [ "$TAG" = "" ] && [ -d "$SAVE_DIR" ] && [ -n "$(ls -A "$SAVE_DIR" 2>/dev/null)" ]; then
    # report §7.4: a fresh arm must start from step 0. Refuse to resume silently.
    echo "[skip] $EXP/$DS: output dir already exists and is not empty -> refuse to resume"
    echo "       (report §7.4 forbids resuming an old arm into a new one; use a new ID)"
    return 1
  fi
  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"

  if [ ! -f "$CACHE3/$TEACHER/$DS/train/manifest.json" ]; then
    echo "[skip] $EXP/$DS: v3 cache missing for $TEACHER/$DS"
    echo "       run: bash train_scripts/CATA-CD/Run3/build_cache.sh <gpu> $TEACHER $DS"
    return 1
  fi

  echo "[run4] $EXP  dataset=$DS teacher=$TEACHER mode=$MODE extra='$EXTRA' steps=$STEPS gpu=$GPU_ID"
  python -m models.scripts.train \
    --experiment_id "$EXP" \
    --dca_mode moe128 --teacher_package "$TEACHER" --aux_task none \
    --channel_mode psfb --teacher_input_mode "$MODE" --v3_teacher "$TEACHER" \
    --teacher_cache_root "$CACHE3" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps "$STEPS" --seed 2333 \
    --save_dir "$SAVE_DIR" --log_file "$LOG_FILE" \
    $EXTRA
}
