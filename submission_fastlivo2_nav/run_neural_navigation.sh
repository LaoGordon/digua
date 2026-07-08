#!/bin/bash
# =============================================================================
# run_navigation.sh — 2D Navigation on Real Robot
#
# 启动 FAST-LIVO2 定位 + 静态地图 + Nav2 导航栈，输出 /cmd_vel
#
# 前置条件：
#   1. 已通过 run_mapping_real.sh 建图并生成 all_raw_points.pcd
#   2. 已生成静态地图：
#      source install/setup.bash
ulimit -c unlimited  # [7/1] 开core dump抓崩溃栈
#      python3 pcd_to_static_map_v2.py \
#        src/FAST-LIVO2/Log/PCD/all_raw_points.pcd \
#        --output-prefix src/fastlivo2_nav/quadruped_nav_bringup/maps/floor_latest \
#        --z-min 0.2 --z-max 1.8 --resolution 0.05
#   3. 机器人放在与建图起始位置相同的位置（FAST-LIVO2 camera_init 原点）
#
# 架构：
#   LiDAR/Camera/IMU → FAST-LIVO2 → bridge → /odom + /obstacle_points + TF
#   map_server → /map
#   Nav2 → /cmd_vel
#
# 坐标系对齐（重要）：
#   FAST-LIVO2 每次启动在当前位置创建新的 camera_init 坐标系。
#   若机器人未放在建图起始位置，会导致 map 与 odom 错位。
#   通过设置环境变量手动校准 map→odom 偏移：
#     export MAP_TO_ODOM_X=偏移米   export MAP_TO_ODOM_Y=偏移米
#     export MAP_TO_ODOM_YAW=偏移弧度
# =============================================================================
set -e

WS_DIR="$HOME/ws_localization"
WS_LIVOX="$HOME/ws_livox"                                # LiDAR 驱动
FAST_LIVO_DIR="$WS_DIR/src/FAST-LIVO2"
TRANSFER_WS="${TRANSFER_WS:-$HOME/lite3_ros_ws}"          # transfer 包所在工作空间

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[nav]${NC} $1"; }
warn() { echo -e "${YELLOW}[nav]${NC} $1"; }
err()  { echo -e "${RED}[nav]${NC} $1"; }

# ---- 参数 ----
MAP_YAML="${MAP_YAML:-$WS_DIR/src/fastlivo2_nav/quadruped_nav_bringup/maps/floor_20260701_visual.yaml}"
NAV_PARAMS="${NAV_PARAMS:-$WS_DIR/src/fastlivo2_nav/quadruped_nav_bringup/config/neural_nav2_params.yaml}"
ENABLE_TRANSFER="${ENABLE_TRANSFER:-true}"
ENABLE_ROBOT_WAKE="${ENABLE_ROBOT_WAKE:-true}"
LOG_DIR="${LOG_DIR:-/tmp/nav_logs_$(date +%Y%m%d_%H%M%S)}"

# ICP localization parameters
USE_ICP="${USE_ICP:-false}"
REFERENCE_PCD="${REFERENCE_PCD:-$WS_DIR/src/FAST-LIVO2/Log/PCD/all_raw_points.pcd}"
ICP_VOXEL_SIZE="${ICP_VOXEL_SIZE:-0.3}"
ICP_HZ="${ICP_HZ:-1.0}"

# Navigation-specific FAST-LIVO2 config (sliding window enabled)
FASTLIVO_NAV_CONFIG="${FASTLIVO_NAV_CONFIG:-$WS_DIR/install/fast_livo/share/fast_livo/config/nav_mid360.yaml}"
USE_AMCL="${USE_AMCL:-false}"

# [录包 20260624] 录制 MID360 原始数据(LiDAR+IMU)用于离线回放定位崩溃
# 开关: RECORD_BAG=false 默认关(不录包), =true 显式开启录制
# 时长: BAG_DURATION=600 默认录600s; =0 表示跟随脚本生命周期(脚本停则录停)
# 保留上限: MAX_BAGS_KEEP=3 只保留最近3个bag, 老的自动删(防爆盘)
# 磁盘水位: MIN_DISK_GB=5 剩余<5G时拒绝录制并告警(防爆盘)
RECORD_BAG="${RECORD_BAG:-false}"
BAG_DURATION="${BAG_DURATION:-600}"
BAG_TOPICS="${BAG_TOPICS:-/livox/lidar /livox/imu}"
MAX_BAGS_KEEP="${MAX_BAGS_KEEP:-3}"
MIN_DISK_GB="${MIN_DISK_GB:-5}"

PIDS=()
HEARTBEAT_PID=""
BAG_RECORD_PID=""  # 录制进程单独管理(cleanup要给它优雅退出时间)

cleanup() {
    log "Emergency stop — sending sit command..."
    ros2 topic pub --once /simple_cmd transfer_interfaces/msg/MotionSimpleCMD \
        "{cmd_code: 0x21010202, size: 0, type: 0}" 2>/dev/null || true
    sleep 1

    # 录制进程需要优雅退出: SIGINT 后等它把 metadata 落盘(最多60s), 否则 bag 损坏
    if [ -n "$BAG_RECORD_PID" ] && kill -0 "$BAG_RECORD_PID" 2>/dev/null; then
        log "[REC] Stopping bag record gracefully (waiting for metadata flush, up to 60s)..."
        kill -INT "$BAG_RECORD_PID" 2>/dev/null || true
        for i in $(seq 1 60); do
            kill -0 "$BAG_RECORD_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$BAG_RECORD_PID" 2>/dev/null && kill -9 "$BAG_RECORD_PID" 2>/dev/null || true
        log "[REC] bag record stopped."
    fi

    log "Shutting down processes..."
    [ -n "$HEARTBEAT_PID" ] && kill "$HEARTBEAT_PID" 2>/dev/null || true
    for pid in "${PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    log "Done."
}
trap cleanup EXIT INT TERM

# ---- 0. 环境 ----
log "Fixing routing (dog=eth0, lidar=eth1)..."
sudo sysctl -w net.ipv4.conf.eth0.rp_filter=0 > /dev/null 2>&1 || true

log "Cleaning up old processes..."
pkill -9 -f "jetson2app\|jetson2motion" 2>/dev/null || true
pkill -9 -f "fastlivo_mapping" 2>/dev/null || true
pkill -9 -f "livox_ros_driver2" 2>/dev/null || true
pkill -9 -f "realsense2_camera" 2>/dev/null || true
sleep 1

mkdir -p "$LOG_DIR"
log "Logs: $LOG_DIR"

log "Sourcing ROS 2..."
source /opt/ros/humble/setup.bash
source "$WS_LIVOX/install/setup.bash"       # livox_ros_driver2
ulimit -c unlimited  # [7/1] 开core dump抓崩溃栈
source "$WS_DIR/install/setup.bash"         # fast_livo, bridge, nav
ulimit -c unlimited  # [7/1] 开core dump抓崩溃栈
source "$TRANSFER_WS/install/setup.bash"    # transfer, transfer_interfaces
ulimit -c unlimited  # [7/1] 开core dump抓崩溃栈

# ---- 1. 机器人通信 + 唤醒 ----
if [ "$ENABLE_TRANSFER" = "true" ]; then
    log "[1/6] Starting ROS2 ↔ UDP transfer..."
    ros2 launch transfer transfer_launch.py > "$LOG_DIR/transfer.log" 2>&1 &
    PIDS+=("$!")
    sleep 3

    # 后台监听 /simple_cmd 话题，记录所有发出的命令
    log "   -> Recording /simple_cmd topic to $LOG_DIR/simple_cmd.log"
    ros2 topic echo --no-arr --flow-style /simple_cmd transfer_interfaces/msg/MotionSimpleCMD \
        >> "$LOG_DIR/simple_cmd.log" 2>&1 &
    PIDS+=("$!")

    warn "Waiting 10s — move away from the robot!"
    sleep 10

    log "   -> Starting heartbeat first (0x21040001, 2Hz)"
    ros2 topic pub -r 2 /simple_cmd transfer_interfaces/msg/MotionSimpleCMD \
        "{cmd_code: 0x21040001, size: 0, type: 0}" > "$LOG_DIR/heartbeat.log" 2>&1 &
    HEARTBEAT_PID=$!
    sleep 2

    log "   -> Stand/sit (0x21010202)"
    ros2 topic pub --once /simple_cmd transfer_interfaces/msg/MotionSimpleCMD \
        "{cmd_code: 0x21010202, size: 0, type: 0}" 2>&1 | tee "$LOG_DIR/stand.log"
    sleep 3

    log "   -> Navi mode (0x21010C03)"
    ros2 topic pub --once /simple_cmd transfer_interfaces/msg/MotionSimpleCMD \
        "{cmd_code: 0x21010C03, size: 0, type: 0}" 2>&1 | tee "$LOG_DIR/navi.log"
    sleep 1
    sleep 1
fi
# ---- 2. 传感器驱动（必须先于 FAST-LIVO2） ----
log "[2/6] Starting sensors..."
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
PIDS+=("$!")
ros2 launch realsense2_camera rs_launch.py &
PIDS+=("$!")
warn "Waiting 6s for sensor streams to stabilize..."
sleep 6

# ---- 3. 启动 FAST-LIVO2 定位（不保存 PCD） ----
log "[3/6] Starting FAST-LIVO2 localization (nav config: sliding window ON)..."
cd "$FAST_LIVO_DIR"
# 限制 glibc malloc arena 数, 防止 OpenMP 多线程 per-thread arena 泄漏
# (20 线程 × 64MB arena = 8GB, 实测 RSS 暴涨根源). 共享 arena 性能损失极小。
export MALLOC_ARENA_MAX=2
# [缓解修复 20260623] 用 jemalloc 置换 glibc malloc, 显著推迟 corrupted double-linked list 崩溃。
#
# 高度怀疑根因(非定论): glibc malloc 在 OpenMP 多线程 + 高频分配下堆元数据自损坏,
# 导致 corrupted double-linked list (SIGABRT, exit -6)。程序越界写嫌疑已 6 个全排除。
#
# 对照实验(2026-06-23, RDK S100P, MID360, 纯LIO, 按 LiDAR 数据时间戳):
#   - 纯 glibc + buffer-cap: 跑 448s 崩 corrupted (崩前 RSS 突涨 + sync 卡死)
#   - jemalloc + buffer-cap: 跑 1072s 用户手动停(未崩) ← 至少推迟 2.4 倍
#   - 历史 纯 glibc 无 buffer-cap: 150-207s 必崩
# 状态: jemalloc 显著缓解(推迟崩溃), 是否根治未知(样本各1次, jemalloc版只跑到1072s手动停)。
#       需多次长时间跑测验证(若 jemalloc 版每次都稳定跑过 2000s+ 才能判根治)。
#
# 回滚: 注释掉下面 LD_PRELOAD 两行即恢复纯 glibc(崩溃点退回 ~448s)。备份 .bak_20260623_arena。
export JEMALLOC_PATH=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export LD_PRELOAD=$JEMALLOC_PATH
# background_thread: 后台线程做 purge, 降低 RSS。其余用 jemalloc 默认。
export MALLOC_CONF=background_thread:true
log "[MEM] using jemalloc LD_PRELOAD=$JEMALLOC_PATH (缓解 corrupted 崩溃, 治本待验证)"
ros2 launch fast_livo nav_mid360.launch.py \
    enable_pcd_save:=False \
    mid360_params_file:="$FASTLIVO_NAV_CONFIG" \
    use_respawn:=False > "$LOG_DIR/fastlivo.log" 2>&1 &
PIDS+=("$!")
warn "Waiting 15s for FAST-LIVO2 initialization..."
sleep 15


# ---- 4. 桥接 + 静态地图 ----
log "[4/6] Starting bridge + static map..."

BRIDGE_ARGS=""

if [ "$USE_ICP" = "true" ]; then
    log "   ICP localization ENABLED (ref=$REFERENCE_PCD, voxel=$ICP_VOXEL_SIZE, hz=$ICP_HZ)"
    BRIDGE_ARGS="publish_identity_map_to_odom:=false use_icp_localization:=true reference_pcd_path:=$REFERENCE_PCD icp_voxel_size:=$ICP_VOXEL_SIZE icp_hz:=$ICP_HZ"
elif [ "$USE_AMCL" = "true" ]; then
    log "   AMCL localization ENABLED (bridge publishes /scan, AMCL publishes map->odom)"
    BRIDGE_ARGS="publish_identity_map_to_odom:=false publish_laser_scan:=true"
else
    MAP_TO_ODOM_X="${MAP_TO_ODOM_X:-0.0}"
    MAP_TO_ODOM_Y="${MAP_TO_ODOM_Y:-0.0}"
    MAP_TO_ODOM_Z="${MAP_TO_ODOM_Z:-0.0}"
    MAP_TO_ODOM_YAW="${MAP_TO_ODOM_YAW:-0.0}"

    if [ "$MAP_TO_ODOM_X" != "0.0" ] || [ "$MAP_TO_ODOM_Y" != "0.0" ] || [ "$MAP_TO_ODOM_YAW" != "0.0" ]; then
        log "   Using calibrated map->odom: ($MAP_TO_ODOM_X, $MAP_TO_ODOM_Y, $MAP_TO_ODOM_Z) yaw=$MAP_TO_ODOM_YAW"
        BRIDGE_ARGS="publish_identity_map_to_odom:=false map_to_odom_x:=$MAP_TO_ODOM_X map_to_odom_y:=$MAP_TO_ODOM_Y map_to_odom_z:=$MAP_TO_ODOM_Z map_to_odom_yaw:=$MAP_TO_ODOM_YAW"
    fi
fi

ros2 launch fastlivo_nav_bridge bridge.launch.py $BRIDGE_ARGS &
PIDS+=("$!")
ros2 launch quadruped_nav_bringup static_map.launch.py \
    map_yaml:="$MAP_YAML" \
    autostart:=true &
PIDS+=("$!")
sleep 3

# ---- 4.5 AMCL 定位 ----
if [ "$USE_AMCL" = "true" ]; then
    log "[4.5] Starting AMCL localization..."
    ros2 launch quadruped_nav_bringup amcl.launch.py &
    PIDS+=("$!")
    sleep 2
    log "[4.5.1] Publishing initial pose for AMCL..."
    ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: "map"}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}" > /dev/null 2>&1
fi

# ---- 5. Nav2 导航栈 ----
log "[5/6] Starting Nav2..."
ros2 launch nav2_bringup navigation_launch.py \
    params_file:="$NAV_PARAMS" \
    autostart:=true &
PIDS+=("$!")
sleep 5

# ---- 6. RViz2 可视化 ----
NAV_RVIZ="${NAV_RVIZ:-$WS_DIR/src/fastlivo2_nav/quadruped_nav_bringup/rviz/nav_real.rviz}"
if [ "${NAV_RVIZ_ENABLE:-true}" = "true" ] && [ -f "$NAV_RVIZ" ]; then
    log "[6/6] Starting RViz2..."
    rviz2 -d "$NAV_RVIZ" &
    PIDS+=("$!")
fi

# ---- 7. pos_logger 延迟监控 ----
POS_LOGGER="$WS_DIR/src/fastlivo2_nav/pos_logger.py"
if [ -f "$POS_LOGGER" ]; then
    log "Starting pos_logger (latency monitor)..."
    python3 "$POS_LOGGER" >> "$LOG_DIR/pos_logger.log" 2>&1 &
    PIDS+=("$!")
fi

# ---- 8. 系统级监控 (捕获 FAST-LIVO2 被杀瞬间: 内存/CPU/OOM) ----
SYSMON="$WS_DIR/src/fastlivo2_nav/sysmon.py"
if [ -f "$SYSMON" ]; then
    log "Starting sysmon (process death + memory monitor)..."
    python3 "$SYSMON" "$LOG_DIR/sysmon.log" 2>&1 &
    PIDS+=("$!")
fi

# ---- 9. 录制 ROS bag (离线回放定位崩溃用) ----
if [ "$RECORD_BAG" = "true" ]; then
    mkdir -p ~/bags
    # 9.1 磁盘水位检查(防爆盘)
    AVAIL_GB=$(df -BG ~/bags | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
        warn "[REC] 磁盘剩余 ${AVAIL_GB}G < 阈值 ${MIN_DISK_GB}G, 拒绝录制(防爆盘)"
        warn "[REC] 清理旧bag: ls -dt ~/bags/bag_* | tail -n +1 | xargs rm -rf; 或设 RECORD_BAG=false 跳过"
    else
        # 9.2 清理旧bag, 只保留最近 MAX_BAGS_KEEP 个(防爆盘堆积)
        OLD_BAGS=$(ls -dt ~/bags/bag_* 2>/dev/null | tail -n +$((MAX_BAGS_KEEP+1)))
        if [ -n "$OLD_BAGS" ]; then
            log "[REC] 清理旧bag(只保留最近${MAX_BAGS_KEEP}个):"
            echo "$OLD_BAGS" | while read -r b; do
                log "[REC]   删除 $b ($(du -sh "$b" 2>/dev/null | awk '{print $1}'))"
                rm -rf "$b"
            done
        fi
        # 9.3 启动录制
        BAG_DIR=~/bags/bag_$(date +%Y%m%d_%H%M%S)
        log "[REC] 磁盘剩余 ${AVAIL_GB}G OK. Starting ros2 bag record -> $BAG_DIR"
        log "[REC] Topics: $BAG_TOPICS | Duration: ${BAG_DURATION}s (0=跟随脚本)"
        if [ "$BAG_DURATION" = "0" ]; then
            ros2 bag record -o "$BAG_DIR" -s sqlite3 $BAG_TOPICS > "$LOG_DIR/bag_record.log" 2>&1 &
        else
            ros2 bag record -o "$BAG_DIR" -s sqlite3 -d "$BAG_DURATION" $BAG_TOPICS > "$LOG_DIR/bag_record.log" 2>&1 &
        fi
        BAG_RECORD_PID=$!
        # 不进 PIDS[], 由 cleanup 单独优雅退出(给 metadata 落盘时间, 防 bag 损坏)
        log "[REC] bag record PID=$BAG_RECORD_PID (单独管理, cleanup 会等它落盘). Path: $BAG_DIR"
    fi
fi

log "============================================"
log "Navigation stack running. Robot is in NAVI mode."
log ""
log "  /cmd_vel   ← Nav2 → UDP → 机器人运动"
log "  /plan      ← 全局路径 (RViz: Global Path)"
log "  /map       ← 静态地图 (RViz: Map)"
log "  /odom      ← FAST-LIVO2 里程计"
log ""
log "  RViz2 中点击 2D Goal Pose 设置目标点"
log ""
log "  Logs saved to: $LOG_DIR"
if [ "$USE_ICP" = "true" ]; then
    log "  ICP localization active: map->odom corrected by point cloud matching"
else
    warn "Robot must start at same position as mapping run."
    warn "map->odom is identity transform."
fi
log "============================================"

wait
log "Experiment finished. Logs: $LOG_DIR"
log "Check: cat $LOG_DIR/simple_cmd.log"

wait
