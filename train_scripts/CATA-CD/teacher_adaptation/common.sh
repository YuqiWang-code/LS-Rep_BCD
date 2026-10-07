#!/bin/bash
# CATA-CD v2 — teacher_adaptation 共享环境与 run_one 助手
# 被 run_gpu0.sh / run_gpu1.sh source；不要单独执行。

PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/CATA_CD_v2"
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

# 实验 ID -> dca_mode / teacher_package（唯一变量对照，见 README）
exp_cfg() {
  case "$1" in
    C0)        echo "none none" ;;
    C1)        echo "moe128 none" ;;
    TV-SAM)    echo "moe128 sam2" ;;
    TV-D2)     echo "moe128 dinov2" ;;
    TV-D3N)    echo "moe128 dinov3_lvd" ;;
    TV-D3S)    echo "moe128 dinov3_sat" ;;
    TV-RCLIP)  echo "moe128 remoteclip" ;;
    TV-MARS)   echo "moe128 mars" ;;
    TV-ANYSAT) echo "moe128 anysat" ;;
    TV-UNISAT) echo "moe128 universat" ;;
    TV-RADIO)  echo "moe128 radio" ;;
    *) echo "unknown experiment $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local EXP="$1"
  local DS="$2"
  local GPU_ID="$3"

  local CFG
  CFG="$(exp_cfg "$EXP")"
  local DCA_MODE TEACHER
  DCA_MODE="${CFG%% *}"
  TEACHER="${CFG##* }"

  local SAVE_DIR="$CKPT_BASE/$EXP/$DS"
  local LOG_FILE="$LOG_BASE/$EXP/$DS/train_log.txt"

  if grep -q '^=== END TEST RESULTS ===' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] $EXP/$DS already complete"
    return 0
  fi
  mkdir -p "$SAVE_DIR" "$(dirname "$LOG_FILE")"

  local RESUME_ARG=""
  if [ -f "$SAVE_DIR/last_checkpoint.pth" ]; then
    RESUME_ARG="--resume $SAVE_DIR/last_checkpoint.pth"
  fi

  local CACHE_ARG=""
  if [ "$TEACHER" != "none" ]; then
    CACHE_ARG="--teacher_cache_root $CACHE_BASE/$TEACHER"
    if [ ! -f "$CACHE_BASE/$TEACHER/$DS/train/manifest.json" ]; then
      echo "[skip] $EXP/$DS: cache not ready for $TEACHER (run prepare_teacher_cache.sh $TEACHER first)"
      return 0
    fi
  fi

  python -m models.scripts.train \
    --experiment_id "$EXP" \
    --dca_mode "$DCA_MODE" \
    --teacher_package "$TEACHER" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps 40000 --seed 2333 \
    --save_dir "$SAVE_DIR" \
    --log_file "$LOG_FILE" \
    $CACHE_ARG $RESUME_ARG
}
