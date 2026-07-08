#!/usr/bin/env bash
set -euo pipefail

cd /home/sunrise/ws_pitch_aim

AXIS="${1:-5000}"
STAND_ARG="${2:-}"

EXTRA_ARGS=(
  --axis "${AXIS}"
  --duration 10
  --rate 20
  --heartbeat-hz 10
  --navi-mode
  --pose-mode
  --send-zero-after
)

if [[ "${STAND_ARG}" == "stand" ]]; then
  EXTRA_ARGS+=(--stand)
fi

python3 ./test_hold_pitch.py "${EXTRA_ARGS[@]}"
