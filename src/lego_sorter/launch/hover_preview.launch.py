"""Standalone saved-pose viewer: RViz only, without hardware or motion nodes."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro


def generate_launch_description():
    description = Path(get_package_share_directory('franka_description'))
    mappings = {'hand': 'true', 'ee_id': 'franka_hand'}
    urdf = xacro.process_file(
        str(description / 'robots/fr3/fr3.urdf.xacro'), mappings=mappings).toxml()
    srdf = xacro.process_file(
        str(description / 'robots/fr3/fr3.srdf.xacro'), mappings=mappings).toxml()
    config = Path(__file__).resolve().parents[1] / 'config/hover_state_preview.rviz'
    return LaunchDescription([
        SetEnvironmentVariable('PYTHONNOUSERSITE', '1'),
        Node(package='rviz2', executable='rviz2', name='hover_preview_rviz',
             output='screen', arguments=['-d', str(config)],
             parameters=[{'robot_description': ParameterValue(urdf, value_type=str),
                          'robot_description_semantic': ParameterValue(srdf, value_type=str)}]),
    ])
