#!/usr/bin/env bash
set -eo pipefail
cd /home/sunrise/ws_pitch_aim
source /opt/ros/humble/setup.bash
source /home/sunrise/ws_pitch_aim/install/setup.bash
ros2 launch fire_pitch_aim_ros2 fire_pitch_aim.launch.py
