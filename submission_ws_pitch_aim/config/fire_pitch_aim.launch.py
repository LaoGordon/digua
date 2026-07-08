import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("fire_pitch_aim_ros2")
    default_params = os.path.join(package_share, "config", "fire_pitch_aim_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("detections_topic", default_value="/fire/detections"),
            DeclareLaunchArgument("image_height", default_value="480"),
            DeclareLaunchArgument("auto_stand", default_value="true"),
            Node(
                package="fire_pitch_aim_ros2",
                executable="fire_pitch_aim_node",
                name="fire_pitch_aim_node",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "image_height": LaunchConfiguration("image_height"),
                        "auto_stand": LaunchConfiguration("auto_stand"),
                    },
                ],
                output="screen",
            ),
        ]
    )
