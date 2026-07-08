from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    params = PathJoinSubstitution([FindPackageShare("patrol2_ros2"), "config", "patrol2_params.yaml"])
    return LaunchDescription(
        [
            Node(
                package="patrol2_ros2",
                executable="patrol2_node",
                name="patrol2_node",
                output="screen",
                parameters=[params],
            )
        ]
    )
