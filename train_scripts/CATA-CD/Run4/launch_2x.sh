#!/bin/bash
# CATA-CD v2 — Run4 启动器。
#
# 阶段（报告 §6.3）：先低 σ 筛选（LEVIR 8 臂 + CDD 3 臂 = 11 run），再高 σ 确认，最后四库主表。
# 每条队列串行；两卡各两条队列即"每卡 2 进程"。
#
# 排期原则（报告 §6.4）：**同域的 CTRL 与 EXP 混排、同步启动**，避免
# "所有 CTRL 独占跑、所有 EXP 挤在拥堵阶段"的混杂。因此这里把每个域的
# Z/R/S/NF 交替分配到两条队列，且每条队列内也交替。
#
# Usage:
#   bash launch_2x.sh            # 启动 Run4 低 σ 筛选排期
#   bash launch_2x.sh status     # 只看当前 tmux/GPU
set -uo pipefail
source "$(dirname "$0")/common.sh"

SESSIONS="r4g0a r4g0b r4g1a r4g1b"

if [ "${1:-}" = "status" ]; then
  tmux ls 2>&1; echo; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
  exit 0
fi

mkdir -p "$LOG_BASE"

# 低 σ 筛选阶段 11 臂，按"同域混排、两卡对称"分配到 4 条队列。
# GPU0：LEVIR D3S 四臂 + CDD D3S 两臂；GPU1：LEVIR D2/RAD 四臂 + CDD D3S 一臂。
Q_G0A="D4-L-D3S-Z,D4-L-D3S-S,D4-C-D3S-Z"
Q_G0B="D4-L-D3S-R,D4-L-D3S-NF,D4-C-D3S-R"
Q_G1A="D4-L-D2-Z,D4-L-D2-R,D4-C-D3S-S"
Q_G1B="D4-L-RAD-Z,D4-L-RAD-R"

launch() {  # $1=session $2=gpu $3=ids $4=logname
  tmux kill-session -t "$1" 2>/dev/null
  tmux new-session -d -s "$1" \
    "cd $PROJ && source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep && \
     bash train_scripts/CATA-CD/Run4/run_queue.sh $2 $3 > $LOG_BASE/$4 2>&1"
  echo "[4x] launched $1 (gpu$2): $3"
}

if [ "${1:-}" = "single" ]; then
  # 每卡一进程（显存不足时的回退，报告 §6.4）
  launch r4g0a 0 "$Q_G0A,$Q_G0B" q_gpu0.log
  launch r4g1a 1 "$Q_G1A,$Q_G1B" q_gpu1.log
else
  launch r4g0a 0 "$Q_G0A" q_gpu0a.log
  launch r4g0b 0 "$Q_G0B" q_gpu0b.log
  launch r4g1a 1 "$Q_G1A" q_gpu1a.log
  launch r4g1b 1 "$Q_G1B" q_gpu1b.log
fi

sleep 20
echo "[4x] tmux:"; tmux ls
echo "[4x] gpu:"; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
