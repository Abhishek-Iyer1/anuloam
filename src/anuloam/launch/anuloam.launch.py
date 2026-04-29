import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('anuloam'), 'config', 'anuloam.yaml'
    )

    return LaunchDescription([
        Node(
            package='anuloam',
            executable='imu_preintegration',
            name='imu_preintegration',
            output='screen',
            parameters=[params],
        ),
        Node(
            package='anuloam',
            executable='feature_extraction',
            name='feature_extraction',
            output='screen',
            parameters=[params],
        ),
        # Node(
        #     package='anuloam',
        #     executable='global_map',
        #     name='global_map',
        #     output='screen',
        # ),
    ])
