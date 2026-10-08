#!/bin/bash
# CATA-CD v2 — Run2 进度监控：各臂 epoch 进度 / ETA / 日志尾部 / GPU / tmux。
# Usage: bash train_scripts/CATA-CD/Run2/monitor.sh [dataset]
source "$(dirname "$0")/common.sh"

DS="${1:-SYSU}"
MAX_STEPS=40000

echo "=== Run2 arms on $DS ==="
printf '%-12s %-9s %-8s %-7s %-9s %s\n' ARM EPOCHS STEP PCT ETA_MIN LAST_LINE
for ARM in J1-C1 J1-GT J1-GATE J1-TASKKD; do
  LOG="$LOG_BASE/$ARM/$DS/train_log.txt"
  if [ ! -f "$LOG" ]; then
    printf '%-12s %s\n' "$ARM" "(no log yet)"
    continue
  fi
  EPOCHS=$(grep -c 'global_step:' "$LOG" 2>/dev/null || echo 0)
  STEP=$(grep -o 'global_step: [0-9]*' "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  STEP=${STEP:-0}
  PCT=$(( STEP * 100 / MAX_STEPS ))
  # ETA from the log's own wall-clock rate: first and last epoch timestamps are not
  # recorded, so use the epoch count as a coarse but honest progress signal.
  ETA="-"
  LAST=$(tail -1 "$LOG" | cut -c1-70)
  if grep -q '^=== END TEST RESULTS ===' "$LOG"; then
    LAST="DONE  $(grep '^F1:' "$LOG" | tail -1)"
  fi
  printf '%-12s %-9s %-8s %-7s %-9s %s\n' "$ARM" "$EPOCHS" "$STEP" "${PCT}%" "$ETA" "$LAST"
done

echo
echo "=== aux region (gate/taskkd), latest epoch ==="
for ARM in J1-GATE J1-TASKKD J1-GT; do
  LOG="$LOG_BASE/$ARM/$DS/train_log.txt"
  [ -f "$LOG" ] && grep -h 'Aux region' "$LOG" | tail -1 | sed "s/^/$ARM: /"
done

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "=== tmux ==="
tmux ls 2>&1

echo
echo "=== driver log tails ==="
for F in j1_gpu0.log j1_gpu1.log; do
  P="$LOG_BASE/$F"
  [ -f "$P" ] && { echo "--- $F ---"; tail -3 "$P"; }
done
