from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='anuloam',
            executable='imu_preintegration',
            name='imu_preintegration',
            output='screen',
        ),
        Node(
            package='anuloam',
            executable='global_map',
            name='global_map',
            output='screen',
        ),
    ])
