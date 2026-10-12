#!/bin/bash
# CATA-CD v2 — Run3 显存守候启动器。
#
# 背景：RSML-3 上同时跑着 PI 的 CASA-CD（每卡 2 进程 × ~10.5 GiB），每卡只剩
# ~9–11 GiB，而一个 batch-64 的 40K run 峰值需要 ~11 GiB。因此不硬抢，改成
# **守候**：持续探测空闲显存，一旦某张卡连续 N 次超过阈值，就自动把该卡的
# Run3 队列挂到 tmux 里。
#
# 安全性：
#   * 只在**该卡还没有 Run3 训练 tmux**时启动，绝不重复起；
#   * 阈值取 14 GiB（留 ~3 GiB 余量给 CASA-CD 的自然波动）；
#   * 每次探测都记录到 watch.log，便于回溯为什么没启动。
#
# Usage:
#   bash train_scripts/CATA-CD/Run3/watch_and_launch.sh [threshold_GiB] [consecutive] [poll_s]
#   默认：14 GiB、连续 3 次、每 120 秒一次
set -uo pipefail
source "$(dirname "$0")/common.sh"

THRESH="${1:-14}"
NEED="${2:-3}"
POLL="${3:-120}"
WATCH_LOG="$LOG_BASE/watch.log"
mkdir -p "$LOG_BASE"

# 每张卡要跑的队列（低 σ 筛选阶段，方案 §7.1）
QUEUE_0="C3L-Z,C3L-R-D3S,C3L-S-D3S,C3L-R-D2"     # LEVIR
QUEUE_1="C3C-Z,C3C-R-D3S,C3C-R-D2"                # CDD
DS_0="LEVIR"
DS_1="CDD"
SESSION_0="run3_gpu0"
SESSION_1="run3_gpu1"

hits_0=0
hits_1=0

free_gib() {  # $1 = gpu index
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | head -1 | awk '{printf "%.2f", $1/1024}'
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"; }

casacd_running() {
  pgrep -u "$(whoami)" -f "CASA-CD" >/dev/null 2>&1 && echo yes || echo no
}

launch() {  # $1=gpu $2=queue $3=ds $4=session
  local gpu="$1" queue="$2" ds="$3" sess="$4"
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new-session -d -s "$sess" \
    "cd $PROJ && source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep && \
     bash train_scripts/CATA-CD/Run3/run_screen.sh $gpu $queue $ds \
     > $LOG_BASE/screen_gpu${gpu}.log 2>&1"
  log "LAUNCHED $sess on GPU$gpu: $queue / $ds -> $LOG_BASE/screen_gpu${gpu}.log"
}

log "=== watcher start: threshold=${THRESH}GiB consecutive=${NEED} poll=${POLL}s ==="
log "CASA-CD running: $(casacd_running)"

while true; do
  for gpu in 0 1; do
    sess_var="SESSION_$gpu"
    sess="${!sess_var}"
    if tmux has-session -t "$sess" 2>/dev/null; then
      continue                      # this GPU's queue already running
    fi
    f=$(free_gib "$gpu")
    if [ -z "$f" ]; then log "GPU$gpu query failed"; continue; fi
    ok=$(awk -v a="$f" -v b="$THRESH" 'BEGIN{print (a>=b)?1:0}')
    if [ "$ok" = "1" ]; then
      if [ "$gpu" = "0" ]; then hits_0=$((hits_0+1)); h=$hits_0; else hits_1=$((hits_1+1)); h=$hits_1; fi
      log "GPU$gpu free=${f}GiB (>=${THRESH}) hit $h/$NEED"
      if [ "$h" -ge "$NEED" ]; then
        qvar="QUEUE_$gpu"; dsvar="DS_$gpu"
        launch "$gpu" "${!qvar}" "${!dsvar}" "$sess"
        if [ "$gpu" = "0" ]; then hits_0=0; else hits_1=0; fi
      fi
    else
      if [ "$gpu" = "0" ]; then hits_0=0; else hits_1=0; fi
      log "GPU$gpu free=${f}GiB (<${THRESH}) waiting"
    fi
  done

  if tmux has-session -t "$SESSION_0" 2>/dev/null && tmux has-session -t "$SESSION_1" 2>/dev/null; then
    log "both queues launched; watcher exiting"
    break
  fi
  sleep "$POLL"
done
