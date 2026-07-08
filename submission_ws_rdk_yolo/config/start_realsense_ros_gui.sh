#!/usr/bin/env bash
set -eo pipefail
export DISPLAY=:0
source /opt/ros/humble/setup.bash
DISPLAY=:0 gnome-terminal -- bash -c 'source /opt/ros/humble/setup.bash && ros2 launch realsense2_camera rs_launch.py; exec bash'
