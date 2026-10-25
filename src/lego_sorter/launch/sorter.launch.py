from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

def generate_launch_description():
    config=LaunchConfiguration('config')
    root=Path('/home/tanay/Documents/Hackathon/project-planner')
    return LaunchDescription([
        DeclareLaunchArgument('config',default_value=str(Path(get_package_share_directory('lego_sorter'))/'config/sorter.yaml')),
        SetEnvironmentVariable('ROS_LOG_DIR',str(root/'runtime/ros_logs')),
        SetEnvironmentVariable('PYTHONDONTWRITEBYTECODE','1'),
        SetEnvironmentVariable('PYTHONNOUSERSITE','1'),
        SetEnvironmentVariable('ROS_HOME',str(root/'runtime/ros_home')),
        *[Node(package='lego_sorter',executable=name,parameters=[{'config':config}],output='screen') for name in ('perception','manipulation','coordinator')]])
