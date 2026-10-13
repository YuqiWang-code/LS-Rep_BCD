#!/bin/bash
# CATA-CD v2 — Run4 实验 ID 解析自测（不需要 GPU）。
# 校验 exp_cfg 的 <DS>-<TEACHER>-<MODE> 解析，并确认对应 v3 cache 是否存在。
# Usage: bash train_scripts/CATA-CD/Run4/selftest_ids.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"

printf '%-16s %-7s %-12s %-8s %-14s %s\n' EXP DS TEACHER MODE EXTRA CACHE
FAIL=0
for E in D4-L-D3S-Z D4-L-D3S-R D4-L-D3S-S D4-L-D3S-NF \
         D4-L-D2-Z D4-L-D2-R D4-L-RAD-Z D4-L-RAD-R \
         D4-C-D3S-Z D4-C-D3S-R D4-C-D3S-S \
         D4-S-RAD-Z D4-S-RAD-R D4-S-RAD-S D4-S-D3S-R D4-S-D3S-Z \
         D4-W-MARS-Z D4-W-MARS-R D4-W-MARS-S D4-W-D3S-R D4-W-D3S-Z ; do
  CFG="$(exp_cfg "$E")" || { echo "PARSE FAIL $E"; FAIL=1; continue; }
  DS="$(echo "$CFG" | awk '{print $1}')"
  T="$(echo "$CFG" | awk '{print $2}')"
  M="$(echo "$CFG" | awk '{print $3}')"
  X="$(echo "$CFG" | cut -d' ' -f4-)"
  if [ -f "$CACHE3/$T/$DS/train/manifest.json" ]; then C="ready"; else C="MISSING"; fi
  printf '%-16s %-7s %-12s %-8s %-14s %s\n' "$E" "$DS" "$T" "$M" "$X" "$C"
done

echo
echo "=== 各队列涉及的 cache 汇总 ==="
find "$CACHE3" -name manifest.json 2>/dev/null | sed "s#$CACHE3/##; s#/manifest.json##" | sort
echo
echo "FAIL=$FAIL"
exit "$FAIL"
