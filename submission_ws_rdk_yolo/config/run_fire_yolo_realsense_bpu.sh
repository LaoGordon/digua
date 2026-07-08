#!/usr/bin/env bash
set -euo pipefail
cd /home/sunrise/ws_rdk_yolo
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
python3 /home/sunrise/ws_rdk_yolo/realsense_fire_yolo_bpu.py "$@"
