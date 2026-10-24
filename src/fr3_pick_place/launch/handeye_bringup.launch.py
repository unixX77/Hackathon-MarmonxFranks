import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    moveit = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'), 'launch', 'moveit.launch.py')
    camera = os.path.join(
        get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.3.10'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        # The camera starts first so RViz's Camera display finds the topic.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera),
            launch_arguments={'enable_color': 'true', 'enable_depth': 'false'}.items()),
        TimerAction(period=5.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(moveit),
            launch_arguments={
                'robot_ip': LaunchConfiguration('robot_ip'),
                'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
            }.items())]),
    ])
