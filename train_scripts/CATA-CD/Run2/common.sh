#!/bin/bash
# CATA-CD v2 — Run2（J0 可重复性 + J1 机制判决）共享环境与 run_one 助手。
# 被 run_j0.sh / run_j1_sysu.sh source；不要单独执行。
#
# 与 Run1/teacher_adaptation 的 common.sh 完全隔离：Run2 使用自己的
# checkpoint / log 根目录，绝不覆盖任何既有 run 的产物（review §8.1 要求）。

PROJ="/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CKPT_BASE="/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run2"
LOG_BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run2"
DATA_BASE="/share_datasets/CD"
CACHE_BASE="/share_datasets/CD_teacher_cache/CATA_CD_v2"
PRETRAINED="$PROJ/pre-trained_weights/lwganet_l0_e299.pth"
CALIB_BASE="$LOG_BASE/calibration"

# J1 诊断教师：按 train-only probe 预先冻结，不按 test F1 选择（review §4.3）。
J1_TEACHER="dinov3_lvd"

ds_folder() {
  case "$1" in
    SYSU) echo "SYSU-CD-256" ;;
    WHU) echo "WHU-CD-256" ;;
    CDD) echo "CDD-CD-256" ;;
    LEVIR) echo "LEVIR-CD-256" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

# 实验 ID -> "dca_mode teacher_package aux_task"
# 唯一变量对照（review §4.3 表）：
#   J1-C1     clean anchor，无辅助头
#   J1-GT     +1ch GT 辅助头（样本集 = 全部像素，正负均衡）
#   J1-GATE   J1-GT 同 head/同超参，但只在 teacher∩GT 一致区域 G 上监督，target 仍为 GT
#   J1-TASKKD J1-GATE 同区域 G，target 换成冻结的 teacher 软响应 q_T
exp_cfg() {
  case "$1" in
    J1-C1)     echo "moe128 none        none" ;;
    J1-GT)     echo "moe128 $J1_TEACHER gt" ;;
    J1-GATE)   echo "moe128 $J1_TEACHER gate" ;;
    J1-TASKKD) echo "moe128 $J1_TEACHER taskkd" ;;
    # S-PCG 阶梯（review §5.1）：唯一变量 = DCA gate 的输入。
    #   S0 legacy 全局 gate（与冻结 C1 完全同构，参数量 3,280,274）
    #   S1 逐尺度 gate，只加对称差分统计 GAP(|a_s-b_s|)        (+192 参数)
    #   S2 S1 + 对称共性上下文 GAP((a_s+b_s)/2) 与 |GAP(a_s)-GAP(b_s)| (+32,960 参数)
    S0-C1)     echo "moe128        none none" ;;
    S1-STAT)   echo "moe128_stats  none none" ;;
    S2-PCG)    echo "moe128_sympcg none none" ;;
    *) echo "unknown experiment $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local EXP="$1"
  local DS="$2"
  local GPU_ID="$3"

  # 冒烟/诊断覆盖：默认即生产协议。RUN_TAG=_smoke 会把产物写到 *Run2_smoke 树，
  # MAX_STEPS_OVERRIDE 允许只跑几十步，SMOKE_CLEAN=1 先清空该臂目录。
  local TAG="${RUN_TAG:-}"
  local STEPS="${MAX_STEPS_OVERRIDE:-40000}"

  local CFG DCA_MODE TEACHER AUX_TASK
  CFG="$(exp_cfg "$EXP")"
  DCA_MODE="${CFG%% *}"
  TEACHER="$(echo "$CFG" | awk '{print $2}')"
  AUX_TASK="$(echo "$CFG" | awk '{print $3}')"

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

  local RESUME_ARG=""
  if [ -f "$SAVE_DIR/last_checkpoint.pth" ]; then
    RESUME_ARG="--resume $SAVE_DIR/last_checkpoint.pth"
  fi

  # GATE/TASKKD 需要 teacher 软证据；GT 也需要同一份 cache 以便与 GATE 共享
  # 完全一致的数据流（review §4.3 的“相同空间抽样集合”控制）。
  local CACHE_ARG="" CALIB_ARG=""
  if [ "$TEACHER" != "none" ]; then
    CACHE_ARG="--teacher_cache_root $CACHE_BASE/$TEACHER"
    if [ ! -f "$CACHE_BASE/$TEACHER/$DS/train/manifest.json" ]; then
      echo "[skip] $EXP/$DS: cache not ready for $TEACHER"
      return 0
    fi
  fi
  if [ "$AUX_TASK" = "gate" ] || [ "$AUX_TASK" = "taskkd" ]; then
    CALIB_ARG="--taskkd_calibration $CALIB_BASE/${DS}_${TEACHER}.json"
    if [ ! -f "${CALIB_ARG##* }" ]; then
      echo "[skip] $EXP/$DS: calibration missing -> run audit_taskkd first ($CALIB_ARG)"
      return 0
    fi
  fi

  python -m models.scripts.train \
    --experiment_id "$EXP" \
    --dca_mode "$DCA_MODE" \
    --teacher_package "$TEACHER" \
    --aux_task "$AUX_TASK" \
    --dataset_name "$DS" --data_root "$DATA_BASE/$(ds_folder "$DS")" \
    --pretrained --pretrained_path "$PRETRAINED" \
    --gpu_id "$GPU_ID" --batch_size 64 --max_steps "$STEPS" --seed 2333 \
    --save_dir "$SAVE_DIR" \
    --log_file "$LOG_FILE" \
    $CACHE_ARG $CALIB_ARG $RESUME_ARG
}
