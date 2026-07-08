#!/bin/bash
set -eo pipefail

WS_LOCALIZATION="$HOME/ws_localization"
WS_LIVOX="$HOME/ws_livox"

FASTLIVO_DIR="$WS_LOCALIZATION/src/FAST-LIVO2"

SENSOR_DELAY="${SENSOR_DELAY:-6}"
FASTLIVO_WARMUP="${FASTLIVO_WARMUP:-2}"
FASTLIVO_SAVE_TIMEOUT="${FASTLIVO_SAVE_TIMEOUT:-60}"
FASTLIVO_SAVE_STATUS_INTERVAL="${FASTLIVO_SAVE_STATUS_INTERVAL:-5}"

PIDS=()

pcd_status() {
  local raw_pcd="$FASTLIVO_DIR/Log/PCD/all_raw_points.pcd"
  local downsampled_pcd="$FASTLIVO_DIR/Log/PCD/all_downsampled_points.pcd"

  if [ -s "$raw_pcd" ]; then
    echo "      raw:         $(du -h "$raw_pcd" | awk '{print $1}')"
  else
    echo "      raw:         not written yet"
  fi

  if [ -s "$downsampled_pcd" ]; then
    echo "      downsampled: $(du -h "$downsampled_pcd" | awk '{print $1}')"
  else
    echo "      downsampled: not written yet"
  fi
}

cleanup() {
  local exit_code=$?
  trap '' INT TERM
  trap - EXIT

  echo ""
  echo "[cleanup] stopping mapping stack..."
  echo "[cleanup] Do not press Ctrl-C again. Waiting for FAST-LIVO2 to flush PCD files."
  for pid in "${PIDS[@]}"; do
    kill -INT "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done

  echo "[cleanup] waiting for FAST-LIVO2 to flush PCD (up to 60s)..."
  for i in $(seq 1 60); do
    if ! pgrep -f "fastlivo_mapping" >/dev/null 2>&1; then
      echo "[cleanup] FAST-LIVO2 exited gracefully (PCD flushed)"
      break
    fi
    sleep 1
  done
  echo "[cleanup] PCD status:"
  pcd_status

  echo "[cleanup] stopping remaining sensor/launch processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  pkill -9 -f "livox_ros_driver2" 2>/dev/null || true
  pkill -9 -f "realsense2_camera" 2>/dev/null || true
  exit $exit_code
}
trap cleanup INT TERM EXIT

source /opt/ros/humble/setup.bash
source "$WS_LIVOX/install/setup.bash"
source "$WS_LOCALIZATION/install/setup.bash"
ulimit -c unlimited  # [7/1] 开core dump抓崩溃栈
# [20260630] 视觉建图用纯 glibc (验证假设: jemalloc+OpenCV 冲突导致 VIO 段错误)
# 假设: jemalloc LD_PRELOAD 拦截 malloc 破坏 OpenCV cv::Mat 的 SIMD 对齐,
#       aarch64 NEON 访问未对齐内存 -> SIGSEGV(-11). 纯激光不碰cv::Mat所以jemalloc不冲突.
# 验证中: 若视觉建图不再崩 -> 确认 jemalloc+OpenCV 冲突
# (建图一次性任务, 在glibc崩溃窗口448s前完成; 导航仍用jemalloc)
# export JEMALLOC_PATH=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# export LD_PRELOAD=$JEMALLOC_PATH
# export MALLOC_ARENA_MAX=2
# export MALLOC_CONF=background_thread:true
# [TEST 20260701] MALLOC_ARENA_MAX=1: 强制 spin 线程和主线程共用同一 arena
# 假设: 双线程并发 malloc 时, 跨 arena 归还 chunk 竞争导致 tcache 损坏
# ARENA_MAX=1 消除跨 arena 迁移 -> 消除竞争源 (代价: 轻微锁竞争, 建图可接受)
# export MALLOC_ARENA_MAX=1  # [7/1] 诊断: 去掉测干净5/12代码
set -u

echo "========================================"
echo "  FAST-LIVO2 Mapping + PCD Save"
echo "========================================"
echo "  mode:    visual (img_en=1) [20260630] + jemalloc"
echo "  pcd:     enabled (press Ctrl-C once to save)"
echo "  output:  $FASTLIVO_DIR/Log/PCD/"
echo "  wait:    ${FASTLIVO_SAVE_TIMEOUT}s max for raw + downsampled PCD flush"
echo "========================================"
echo

# Clean stale processes from previous runs
echo "[pre] cleaning stale processes..."
pkill -9 -f "fastlivo_mapping" 2>/dev/null || true
pkill -9 -f "fastlivo_nav_bridge" 2>/dev/null || true
pkill -9 -f "floor_mapper" 2>/dev/null || true
pkill -9 -f "realsense2_camera" 2>/dev/null || true
pkill -9 -f "livox_ros_driver2" 2>/dev/null || true
sleep 2

# ---------- [1/3] Sensors ----------
echo "[1/3] starting sensors..."
echo "      -> Livox MID360"
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
PIDS+=("$!")

echo "      -> RealSense D435i"
ros2 launch realsense2_camera rs_launch.py &
PIDS+=("$!")

echo "      waiting $SENSOR_DELAY s for sensor streams to stabilize..."
sleep "$SENSOR_DELAY"

# ---------- [2/3] FAST-LIVO2 ----------
echo "[2/3] starting FAST-LIVO2 mapping node..."
cd "$FASTLIVO_DIR"
MAP_LOG_DIR="${MAP_LOG_DIR:-/tmp/map_logs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$MAP_LOG_DIR"
echo "[log] mapping logs -> $MAP_LOG_DIR/fastlivo.log"
ros2 launch fast_livo mapping_mid360.launch.py \
  enable_pcd_save:=True \
  use_rviz:=True \
  use_respawn:=False > "$MAP_LOG_DIR/fastlivo.log" 2>&1 &
PIDS+=("$!")

echo "      waiting $FASTLIVO_WARMUP s for FAST-LIVO2 initialization..."
sleep "$FASTLIVO_WARMUP"

# ---------- [3/3] Wait for Ctrl+C ----------
echo
echo "Mapping started. Move the robot to cover the target area."
echo "Press Ctrl-C once to stop and save PCD files:"
echo "  $FASTLIVO_DIR/Log/PCD/all_raw_points.pcd"
echo "  $FASTLIVO_DIR/Log/PCD/all_downsampled_points.pcd"
echo "After pressing Ctrl-C, wait for the cleanup messages to finish."
echo

wait
