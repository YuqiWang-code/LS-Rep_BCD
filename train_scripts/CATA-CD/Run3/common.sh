#!/bin/bash
# CATA-CD v2 — Run3（第三轮：CTP-Critic 知识通道）共享环境与 run_one 助手。
#
# 与 Run1/Run2 完全隔离：自己的 checkpoint / log / cache 根目录，绝不覆盖既有产物。
#
# 机制（第三轮方案 §5）：训练期在 c2 上挂一个**特权评论器**，输入为教师 v3 cache 的
# 多通道对称 E/M；用 teacher_input_mode 做反事实对照：
#   real    同图配对的 U
#   zero    U 在使用处替换为 0（适配器照常前向，唯一变量只有 U）
#   shuffle 取同 batch 内其它样本的 U（幅值分布不变、配对关系破坏）
# 部署图不变：学生仍 3,280,274 参数，无 critic / 无 teacher / 无 cache。

PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run3"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run3"
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

# 实验 ID -> "teacher_input_mode teacher_id"
#   C3L-Z / C3C-Z / C3S-Z / C3W-Z      反事实 zero 臂（同期 CTRL）
#   C3L-R-<T> / C3C-R-<T> / ...         real 臂（真实配对教师）
#   C3L-S-<T>                           shuffle 臂（教师特异性对照）
exp_cfg() {
  case "$1" in
    C3L-Z)        echo "zero dinov3_sat" ;;
    C3L-S-D3S)    echo "shuffle dinov3_sat" ;;
    C3L-R-D3S)    echo "real dinov3_sat" ;;
    C3L-R-D2)     echo "real dinov2" ;;
    C3C-Z)        echo "zero dinov3_sat" ;;
    C3C-R-D3S)    echo "real dinov3_sat" ;;
    C3C-R-D2)     echo "real dinov2" ;;
    C3S-Z)        echo "zero dinov3_sat" ;;
    C3S-R-D3S)    echo "real dinov3_sat" ;;
    C3S-R-RADIO)  echo "real radio" ;;
    C3W-Z)        echo "zero dinov3_sat" ;;
    C3W-R-MARS)   echo "real mars" ;;
    C3W-R-SAM2)   echo "real sam2" ;;
    *) echo "unknown experiment $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local EXP="$1"
  local DS="$2"
  local GPU_ID="$3"

  local TAG="${RUN_TAG:-}"
  local STEPS="${MAX_STEPS_OVERRIDE:-40000}"

  local CFG MODE TEACHER
  CFG="$(exp_cfg "$EXP")"
  MODE="${CFG%% *}"
  TEACHER="${CFG##* }"

  local SAVE_DIR="$CKPT_BASE$TAG/$EXP/$DS"
  local LOG_FILE="$LOG_BASE$TAG/$EXP/$DS/train_log.txt"

  if [ "${SMOKE_CLEAN:-0}" = "1" ]; then
    rm -rf "$SAVE_DIR" "$(dirname "$LOG_FILE")"
  fi
  if [ "$TAG" = "" ] && grep -q '^=== END TEST RESULTS ===' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] $EXP/$DS already complete"
    return 0
  fi
  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"

  if [ ! -f "$CACHE3/$TEACHER/$DS/train/manifest.json" ]; then
    echo "[skip] $EXP/$DS: v3 cache missing for $TEACHER/$DS"
    echo "       run: bash train_scripts/CATA-CD/Run3/build_cache.sh <gpu> $TEACHER $DS"
    return 1
  fi

  local RESUME_ARG=""
  if [ -f "$SAVE_DIR/last_checkpoint.pth" ]; then
    RESUME_ARG="--resume $SAVE_DIR/last_checkpoint.pth"
  fi

  echo "[run3] $EXP/$DS mode=$MODE teacher=$TEACHER steps=$STEPS"
  python -m models.scripts.train \
    --experiment_id "$EXP" \
    --dca_mode moe128 --teacher_package "$TEACHER" --aux_task none \
    --channel_mode ctp --teacher_input_mode "$MODE" --v3_teacher "$TEACHER" \
    --teacher_cache_root "$CACHE3" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps "$STEPS" --seed 2333 \
    --save_dir "$SAVE_DIR" --log_file "$LOG_FILE" \
    $RESUME_ARG
}
