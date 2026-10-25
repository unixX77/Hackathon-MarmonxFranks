import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """MoveIt + RViz + the Franka Hand + the table for pose-driven pick and place. No brick, no camera.

    Real robot: robot_ip:=...   Simulation: use_fake_hardware:=true (adds the simulated hand and the
    position controller the fake arm needs)."""
    moveit = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'), 'launch', 'moveit.launch.py')
    share = get_package_share_directory('fr3_pose_pick')
    lib = os.path.join(get_package_prefix('fr3_pose_pick'), 'lib', 'fr3_pose_pick')
    config = os.path.join(share, 'config', 'pose_pick.yaml')
    sim_arm_config = os.path.join(share, 'config', 'sim_arm_controller.yaml')
    fake = LaunchConfiguration('use_fake_hardware')
    simulated = IfCondition(fake)
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.3.10'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(moveit),
            launch_arguments={
                'robot_ip': LaunchConfiguration('robot_ip'),
                'use_fake_hardware': fake,
                'use_rviz': LaunchConfiguration('use_rviz'),
            }.items()),
        # spawn_scene.py waits for MoveIt itself; the delay just lets RViz open first.
        TimerAction(period=8.0, actions=[
            ExecuteProcess(cmd=[os.path.join(lib, 'spawn_scene.py'), '--config', config, '--no-brick'],
                           output='screen')]),
        ExecuteProcess(cmd=[os.path.join(lib, 'sim_gripper.py')], output='screen', condition=simulated),
        ExecuteProcess(cmd=[os.path.join(lib, 'sim_arm_controller.sh'), sim_arm_config],
                       output='screen', condition=simulated),
    ])
