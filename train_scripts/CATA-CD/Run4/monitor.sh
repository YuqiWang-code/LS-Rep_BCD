#!/bin/bash
# CATA-CD v2 — Run4 进度监控。
# Usage: bash train_scripts/CATA-CD/Run4/monitor.sh
source "$(dirname "$0")/common.sh"

MAX_STEPS=40000
echo "=== Run4 arms ==="
printf '%-18s %-7s %-7s %-8s %-6s %s\n' EXP DS EPOCHS STEP PCT LAST
ARMS=()
for d in "$LOG_BASE"/*/; do
  [ -d "$d" ] || continue
  a=$(basename "$d")
  [ "$a" = "Run4_smoke" ] && continue
  ARMS+=("$a")
done
if [ ${#ARMS[@]} -eq 0 ]; then
  echo "  (no logs yet)"
fi
for EXP in $(printf '%s\n' "${ARMS[@]}" | sort); do
  for f in "$LOG_BASE/$EXP"/*/train_log.txt; do
    [ -f "$f" ] || continue
    DS=$(basename "$(dirname "$f")")
    EPOCHS=$(grep -c 'global_step:' "$f" 2>/dev/null || echo 0)
    STEP=$(grep -o 'global_step: [0-9]*' "$f" 2>/dev/null | tail -1 | awk '{print $2}')
    STEP=${STEP:-0}
    PCT=$(( STEP * 100 / MAX_STEPS ))
    LAST=$(tail -1 "$f" | cut -c1-52)
    if grep -q '^=== END TEST RESULTS ===' "$f"; then
      LAST="DONE F1=$(grep '^F1:' "$f" | tail -1 | awk '{print $2}') IoU=$(grep '^IoU:' "$f" | tail -1 | awk '{print $2}') mode=$(grep '^Teacher Input Mode:' "$f" | tail -1 | awk '{print $4}')"
    fi
    printf '%-18s %-7s %-7s %-8s %-6s %s\n' "$EXP" "$DS" "$EPOCHS" "$STEP" "${PCT}%" "$LAST"
  done
done

echo
echo "=== PSFB diagnostics (latest epoch per arm) ==="
for f in "$LOG_BASE"/*/*/train_log.txt; do
  [ -f "$f" ] || continue
  line=$(grep -h '^PSFB:' "$f" | tail -1)
  [ -n "$line" ] && echo "  $(basename "$(dirname "$(dirname "$f")")")/$(basename "$(dirname "$f")"): $line"
done

echo
echo "=== 塌缩判据：||HT-H0|| 应显著 > 0；若早期塌到 ~0 则 L_feat 无物可传 ==="
echo
echo "=== v3 cache ==="
find "$CACHE3" -name manifest.json 2>/dev/null | while read -r m; do
  n=$(ls "$(dirname "$m")"/*.pt 2>/dev/null | wc -l)
  echo "  $(echo "$m" | sed "s#$CACHE3/##; s#/manifest.json##"): $n"
done
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== tmux ==="
tmux ls 2>&1
