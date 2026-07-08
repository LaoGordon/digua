#!/usr/bin/env bash
set -eo pipefail
cd /home/sunrise/ws_rdk_yolo
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export LD_LIBRARY_PATH="/usr/hobot/lib:${LD_LIBRARY_PATH:-}"
source /opt/ros/humble/setup.bash
source /home/sunrise/ws_rdk_yolo/install/setup.bash

HBM_RUNTIME_VENDOR_DIR="/home/sunrise/ws_rdk_yolo/third_party"
export PYTHONPATH="${HBM_RUNTIME_VENDOR_DIR}:${PYTHONPATH:-}"

ensure_hbm_runtime() {
  if python3 -c 'import hbm_runtime; from hbm_runtime import HB_HBMRuntime; print(getattr(HB_HBMRuntime, "version", "ok"))' >/tmp/ws_rdk_yolo_hbm_runtime_check.log 2>&1; then
    return 0
  fi
  echo "[INFO] installing local hbm_runtime into ${HBM_RUNTIME_VENDOR_DIR}"
  mkdir -p "${HBM_RUNTIME_VENDOR_DIR}"
  python3 -m pip install --no-build-isolation --target "${HBM_RUNTIME_VENDOR_DIR}" /usr/hobot/lib/hbm_runtime
  python3 -c 'import hbm_runtime; from hbm_runtime import HB_HBMRuntime; print(getattr(HB_HBMRuntime, "version", "ok"))' >/tmp/ws_rdk_yolo_hbm_runtime_check.log 2>&1
}

camera_topics_ready() {
  ros2 topic list 2>/dev/null | grep -Fxq /camera/camera/color/image_raw \
    && ros2 topic list 2>/dev/null | grep -Fxq /camera/camera/depth/image_rect_raw \
    && ros2 topic list 2>/dev/null | grep -Fxq /camera/camera/color/camera_info
}

start_realsense_if_needed() {
  if camera_topics_ready; then
    echo "[INFO] realsense camera topics already available"
    return 0
  fi
  if pgrep -af 'realsense2_camera.*(rs_launch.py|realsense2_camera.launch.py)' >/dev/null; then
    echo "[INFO] realsense2_camera is already running"
  else
    echo "[INFO] starting realsense2_camera"
    nohup bash -lc 'source /opt/ros/humble/setup.bash; export DISPLAY=:0; export XDG_RUNTIME_DIR=/run/user/$(id -u); ros2 launch realsense2_camera rs_launch.py' >/tmp/ws_rdk_yolo_realsense.log 2>&1 &
  fi
  for _ in $(seq 1 20); do
    if camera_topics_ready; then
      echo "[INFO] realsense camera topics are ready"
      return 0
    fi
    sleep 0.5
  done
  echo "[WARN] realsense camera topics are not ready yet"
}

ensure_hbm_runtime
start_realsense_if_needed
ros2 launch fire_yolo_bpu_ros2 fire_yolo_bpu.launch.py
