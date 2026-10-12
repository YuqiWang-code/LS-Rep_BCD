#!/bin/bash
# CATA-CD v2 — Run3「每卡 2 进程」筛选排期启动器。
#
# 设计（用户 2026-10-12 决定）：把筛选阶段压满两卡吞吐，**代价是两个 zero CTRL
# 一起重跑**，从而让全部 8 个臂处于**完全相同的并发条件**下：
#
#   GPU0-A: C3L-Z:LEVIR       -> C3L-R-D3S:LEVIR
#   GPU0-B: C3L-S-D3S:LEVIR   -> C3L-R-D2:LEVIR
#   GPU1-A: C3C-Z:CDD         -> C3C-R-D3S:CDD
#   GPU1-B: C3C-R-D2:CDD      -> C3L-R-RADIO:LEVIR
#
# 每条队列 2 个臂、四个队列同时开始与结束 ⇒ 任何时刻恰好 2 个训练进程/卡。
# 第 8 臂 C3L-R-RADIO 同时把 LEVIR 的教师跨度从 2 个扩到 3 个（Agent 需要
# "不同教师在同一个域上可区分"的证据，2 个教师答不了）。
#
# 使用前必须确认：GPU 上没有其它训练/缓存生成进程（2×训练需要 ~23 GiB/卡）。
#
# Usage:
#   bash train_scripts/CATA-CD/Run3/launch_2x.sh fresh   # 先清除这 8 个臂的旧产物再启动
#   bash train_scripts/CATA-CD/Run3/launch_2x.sh         # 直接启动（会按需 resume）
set -uo pipefail
source "$(dirname "$0")/common.sh"

FRESH="${1:-}"
mkdir -p "$LOG_BASE"

SESSIONS="r3g0a r3g0b r3g1a r3g1b"
ARMS_DS="C3L-Z:LEVIR C3L-R-D3S:LEVIR C3L-S-D3S:LEVIR C3L-R-D2:LEVIR C3L-R-RADIO:LEVIR C3C-Z:CDD C3C-R-D3S:CDD C3C-R-D2:CDD"

stop_all() {
  for s in $SESSIONS; do tmux kill-session -t "$s" 2>/dev/null; done
  # 等待显存真正释放，否则新的 2x 训练会 OOM
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [ "$used" -lt 1500 ] && break
    sleep 5
  done
  echo "[2x] gpu after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
}

if [ "$FRESH" = "fresh" ]; then
  echo "[2x] FRESH: stopping current queues and removing this schedule's artifacts"
  stop_all
  for item in $ARMS_DS; do
    ARM="${item%%:*}"; DS="${item##*:}"
    rm -rf "$CKPT_BASE/$ARM/$DS" "$LOG_BASE/$ARM/$DS"
    echo "  wiped $ARM/$DS"
  done
  rm -f "$LOG_BASE"/q_gpu*.log
fi

launch() {  # $1=session $2=gpu $3=spec $4=logname
  tmux kill-session -t "$1" 2>/dev/null
  tmux new-session -d -s "$1" \
    "cd $PROJ && source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep && \
     bash train_scripts/CATA-CD/Run3/run_queue.sh $2 $3 > $LOG_BASE/$4 2>&1"
  echo "[2x] launched $1 (gpu$2): $3 -> $LOG_BASE/$4"
}

launch r3g0a 0 "C3L-Z:LEVIR,C3L-R-D3S:LEVIR"     q_gpu0a.log
launch r3g0b 0 "C3L-S-D3S:LEVIR,C3L-R-D2:LEVIR"  q_gpu0b.log
launch r3g1a 1 "C3C-Z:CDD,C3C-R-D3S:CDD"         q_gpu1a.log
launch r3g1b 1 "C3C-R-D2:CDD,C3L-R-RADIO:LEVIR"  q_gpu1b.log

sleep 20
echo "[2x] tmux:"; tmux ls
echo "[2x] gpu:"; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
