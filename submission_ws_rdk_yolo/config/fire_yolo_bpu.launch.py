import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("fire_yolo_bpu_ros2")
    default_params = os.path.join(package_share, "config", "fire_yolo_bpu_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "model_path",
                default_value="/home/sunrise/ws_rdk_yolo/fire_yolov8n_s100p.hbm",
            ),
            DeclareLaunchArgument("display", default_value="true"),
            Node(
                package="fire_yolo_bpu_ros2",
                executable="fire_yolo_bpu_node",
                name="fire_yolo_bpu_node",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "display": LaunchConfiguration("display"),
                    },
                ],
                output="screen",
            ),
        ]
    )
