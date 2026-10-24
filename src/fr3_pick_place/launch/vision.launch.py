"""Start MoveIt and observation worker; run the interactive runner separately."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory, get_package_prefix


def generate_launch_description():
    share = get_package_share_directory('fr3_pick_place')
    moveit = get_package_share_directory('franka_fr3_moveit_config')
    worker = os.path.join(get_package_prefix('fr3_pick_place'), 'lib', 'fr3_pick_place', 'vision_perception.py')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(share, 'config', 'vision_demo.yaml')),
        DeclareLaunchArgument('start_moveit', default_value='true'),
        DeclareLaunchArgument('use_fake_hardware', default_value='true'),
        DeclareLaunchArgument('robot_ip', default_value='dont-care'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('perception_python', default_value='/usr/bin/python3',
            description='Python with ROS modules and, for real perception, teammate SAM3 dependencies'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(moveit, 'launch', 'moveit.launch.py')),
            condition=IfCondition(LaunchConfiguration('start_moveit')),
            launch_arguments={'robot_ip': LaunchConfiguration('robot_ip'),
                'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
                'load_gripper': 'true', 'use_rviz': LaunchConfiguration('use_rviz')}.items()),
        ExecuteProcess(cmd=[LaunchConfiguration('perception_python'), worker, '--config', LaunchConfiguration('config')],
                       output='screen'),
    ])
