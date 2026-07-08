
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='pump_ros2',
                executable='pump_spray_node',
                name='pump_spray_node',
                output='screen',
                parameters=[{'pin': 37, 'spray_duration_sec': 5.0}],
            )
        ]
    )
