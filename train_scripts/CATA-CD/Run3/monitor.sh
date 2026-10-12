#!/bin/bash
# CATA-CD v2 — Run3 进度监控（各臂 epoch/step/F1 + CTP 诊断量 + GPU + tmux）。
# Usage: bash train_scripts/CATA-CD/Run3/monitor.sh [dataset]
source "$(dirname "$0")/common.sh"

DS="${1:-LEVIR}"
MAX_STEPS=40000

echo "=== Run3 arms on $DS ==="
ARMS=()
for d in "$LOG_BASE"/*/; do
  [ -d "$d" ] || continue
  a=$(basename "$d")
  [ -f "$d$DS/train_log.txt" ] && ARMS+=("$a")
done
if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(C3L-Z C3L-R-D3S C3L-S-D3S C3L-R-D2 C3C-Z C3C-R-D3S C3C-R-D2); fi

printf '%-14s %-7s %-8s %-6s %s\n' ARM EPOCHS STEP PCT LAST
for ARM in $(printf '%s\n' "${ARMS[@]}" | sort); do
  LOG="$LOG_BASE/$ARM/$DS/train_log.txt"
  if [ ! -f "$LOG" ]; then
    printf '%-14s %s\n' "$ARM" "(no log yet)"
    continue
  fi
  EPOCHS=$(grep -c 'global_step:' "$LOG" 2>/dev/null || echo 0)
  STEP=$(grep -o 'global_step: [0-9]*' "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  STEP=${STEP:-0}
  PCT=$(( STEP * 100 / MAX_STEPS ))
  LAST=$(tail -1 "$LOG" | cut -c1-58)
  if grep -q '^=== END TEST RESULTS ===' "$LOG"; then
    LAST="DONE F1=$(grep '^F1:' "$LOG" | tail -1 | awk '{print $2}') IoU=$(grep '^IoU:' "$LOG" | tail -1 | awk '{print $2}') mode=$(grep '^Teacher Input Mode:' "$LOG" | tail -1 | awk '{print $4}')"
  fi
  printf '%-14s %-7s %-8s %-6s %s\n' "$ARM" "$EPOCHS" "$STEP" "${PCT}%" "$LAST"
done

echo
echo "=== CTP diagnostics (latest epoch) ==="
for d in "$LOG_BASE"/*/; do
  [ -d "$d" ] || continue
  f="$d$DS/train_log.txt"
  [ -f "$f" ] && grep -h '^CTP:' "$f" | tail -1 | sed "s#^#$(basename $d): #"
done

echo
echo "=== v3 cache tree ==="
find "$CACHE3" -name manifest.json 2>/dev/null | while read -r m; do
  n=$(ls "$(dirname "$m")"/*.pt 2>/dev/null | wc -l)
  echo "  $(echo "$m" | sed "s#$CACHE3/##; s#/manifest.json##"): $n samples"
done

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "=== tmux ==="
tmux ls 2>&1
