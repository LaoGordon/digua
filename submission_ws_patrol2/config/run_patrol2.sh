#!/usr/bin/env bash
set -eo pipefail

cd /home/sunrise/ws_patrol2
source /opt/ros/humble/setup.bash
export PYTHONPATH="/home/sunrise/ws_patrol2/install/patrol2_ros2/lib/python3.10/site-packages:${PYTHONPATH:-}"
exec /home/sunrise/ws_patrol2/install/patrol2_ros2/lib/patrol2_ros2/patrol2_node \
  --ros-args \
  --params-file /home/sunrise/ws_patrol2/src/patrol2_ros2/config/patrol2_params.yaml
