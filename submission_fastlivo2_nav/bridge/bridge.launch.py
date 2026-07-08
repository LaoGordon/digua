from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fastlivo_odom_topic = LaunchConfiguration('fastlivo_odom_topic')
    fastlivo_cloud_topic = LaunchConfiguration('fastlivo_cloud_topic')
    odom_topic = LaunchConfiguration('odom_topic')
    obstacle_topic = LaunchConfiguration('obstacle_topic')
    map_frame = LaunchConfiguration('map_frame')
    odom_frame = LaunchConfiguration('odom_frame')
    base_frame = LaunchConfiguration('base_frame')
    publish_identity_map_to_odom = LaunchConfiguration('publish_identity_map_to_odom')
    map_to_odom_x = LaunchConfiguration('map_to_odom_x')
    map_to_odom_y = LaunchConfiguration('map_to_odom_y')
    map_to_odom_z = LaunchConfiguration('map_to_odom_z')
    map_to_odom_yaw = LaunchConfiguration('map_to_odom_yaw')
    use_icp_localization = LaunchConfiguration('use_icp_localization')
    reference_pcd_path = LaunchConfiguration('reference_pcd_path')
    icp_voxel_size = LaunchConfiguration('icp_voxel_size')
    ref_voxel_size = LaunchConfiguration('ref_voxel_size')
    icp_max_iterations = LaunchConfiguration('icp_max_iterations')
    icp_max_correspondence_distance = LaunchConfiguration('icp_max_correspondence_distance')
    icp_hz = LaunchConfiguration('icp_hz')
    publish_laser_scan = LaunchConfiguration('publish_laser_scan')

    return LaunchDescription([
        DeclareLaunchArgument(
            'fastlivo_odom_topic',
            default_value='/aft_mapped_to_init',
            description='FAST-LIVO2 odometry topic'),
        DeclareLaunchArgument(
            'fastlivo_cloud_topic',
            default_value='/cloud_registered_lidar',
            description='FAST-LIVO2 registered cloud topic'),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Output nav odometry topic'),
        DeclareLaunchArgument(
            'obstacle_topic',
            default_value='/obstacle_points',
            description='Output obstacle pointcloud topic'),
        DeclareLaunchArgument(
            'map_frame',
            default_value='map',
            description='Global map frame'),
        DeclareLaunchArgument(
            'odom_frame',
            default_value='odom',
            description='Odometry frame'),
        DeclareLaunchArgument(
            'base_frame',
            default_value='base',
            description='Robot base frame'),
        DeclareLaunchArgument(
            'publish_identity_map_to_odom',
            default_value='true',
            description='Publish identity map to odom TF'),
        DeclareLaunchArgument(
            'map_to_odom_x',
            default_value='0.0',
            description='X offset from map to odom (m)'),
        DeclareLaunchArgument(
            'map_to_odom_y',
            default_value='0.0',
            description='Y offset from map to odom (m)'),
        DeclareLaunchArgument(
            'map_to_odom_z',
            default_value='0.0',
            description='Z offset from map to odom (m)'),
        DeclareLaunchArgument(
            'map_to_odom_yaw',
            default_value='0.0',
            description='Yaw rotation from map to odom (rad)'),
        DeclareLaunchArgument(
            'use_icp_localization',
            default_value='false',
            description='Enable ICP-based localization against reference PCD'),
        DeclareLaunchArgument(
            'reference_pcd_path',
            default_value='',
            description='Path to reference PCD map for ICP matching'),
        DeclareLaunchArgument(
            'icp_voxel_size',
            default_value='0.3',
            description='Voxel size for live cloud downsampling (m)'),
        DeclareLaunchArgument(
            'ref_voxel_size',
            default_value='0.1',
            description='Voxel size for reference cloud downsampling (m)'),
        DeclareLaunchArgument(
            'icp_max_iterations',
            default_value='30',
            description='Max ICP iterations'),
        DeclareLaunchArgument(
            'icp_max_correspondence_distance',
            default_value='2.0',
            description='Max correspondence distance for ICP (m)'),
        DeclareLaunchArgument(
            'icp_hz',
            default_value='1.0',
            description='ICP processing frequency (Hz)'),
        DeclareLaunchArgument(
            'publish_laser_scan',
            default_value='false',
            description='Publish /scan LaserScan for AMCL'),
        Node(
            package='fastlivo_nav_bridge',
            executable='fastlivo_nav_bridge_node',
            name='fastlivo_nav_bridge',
            output='screen',
            parameters=[
                {
                    'fastlivo_odom_topic': fastlivo_odom_topic,
                    'fastlivo_cloud_topic': fastlivo_cloud_topic,
                    'odom_topic': odom_topic,
                    'obstacle_topic': obstacle_topic,
                    'map_frame': map_frame,
                    'odom_frame': odom_frame,
                    'base_frame': base_frame,
                    'publish_identity_map_to_odom': publish_identity_map_to_odom,
                    'map_to_odom_x': map_to_odom_x,
                    'map_to_odom_y': map_to_odom_y,
                    'map_to_odom_z': map_to_odom_z,
                    'map_to_odom_yaw': map_to_odom_yaw,
                    'use_icp_localization': use_icp_localization,
                    'reference_pcd_path': reference_pcd_path,
                    'icp_voxel_size': icp_voxel_size,
                    'ref_voxel_size': ref_voxel_size,
                    'icp_max_iterations': icp_max_iterations,
                    'icp_max_correspondence_distance': icp_max_correspondence_distance,
                    'icp_hz': icp_hz,
                    'publish_laser_scan': publish_laser_scan,
                }
            ],
        )
    ])
