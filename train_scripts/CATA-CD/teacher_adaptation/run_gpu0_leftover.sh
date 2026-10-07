#!/bin/bash
# CATA-CD v2 — GPU 0 收尾队列：跑剩余的 UNISAT（LEVIR/WHU）。
# Usage: bash train_scripts/CATA-CD/teacher_adaptation/run_gpu0_leftover.sh [gpu_id=0]
set -euo pipefail

GPU_ID="${1:-0}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/common.sh"

set +e
run_one TV-UNISAT LEVIR "$GPU_ID" &
P1=$!
run_one TV-UNISAT WHU "$GPU_ID" &
P2=$!
wait "$P1"
wait "$P2"
echo "[run_gpu0_leftover] DONE"
