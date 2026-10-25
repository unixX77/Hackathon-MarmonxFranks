import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    moveit = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'), 'launch', 'moveit.launch.py')
    config = os.path.join(get_package_share_directory('fr3_visual_pick'), 'config', 'lego.yaml')
    spawn = os.path.join(get_package_prefix('fr3_visual_pick'), 'lib', 'fr3_visual_pick', 'spawn_scene.py')
    spawn_cmd = [spawn, '--config', config]
    sim_world = os.path.join(get_package_prefix('fr3_visual_pick'), 'lib', 'fr3_visual_pick', 'sim_world.py')
    sim_arm = os.path.join(get_package_prefix('fr3_visual_pick'), 'lib', 'fr3_visual_pick', 'sim_arm_controller.sh')
    sim_arm_config = os.path.join(get_package_share_directory('fr3_visual_pick'), 'config',
                                  'sim_arm_controller.yaml')
    fake = LaunchConfiguration('use_fake_hardware')
    world = LaunchConfiguration('sim_world')
    # sim_world only ever starts together with fake hardware
    run_world = IfCondition(PythonExpression(["'", world, "' == 'true' and '", fake, "' == 'true'"]))
    static_brick = IfCondition(PythonExpression(
        ["'", LaunchConfiguration('spawn_brick'), "' == 'true' and '", world, "' == 'false'"]))
    no_brick = UnlessCondition(PythonExpression(
        ["'", LaunchConfiguration('spawn_brick'), "' == 'true' and '", world, "' == 'false'"]))
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.3.10'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('spawn_brick', default_value='true',
                              description='Also add a static virtual red brick to the planning scene'),
        DeclareLaunchArgument('sim_world', default_value='false',
                              description='Fake camera + hand + movable brick (needs use_fake_hardware:=true)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(moveit),
            launch_arguments={
                'robot_ip': LaunchConfiguration('robot_ip'),
                'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
                'use_rviz': LaunchConfiguration('use_rviz'),
            }.items()),
        # spawn_scene.py waits for MoveIt itself; the delay just lets RViz open first.
        TimerAction(period=8.0, actions=[
            ExecuteProcess(cmd=spawn_cmd, output='screen', condition=static_brick),
            ExecuteProcess(cmd=spawn_cmd + ['--no-brick'], output='screen', condition=no_brick),
        ]),
        ExecuteProcess(cmd=[sim_world, '--config', config], output='screen', condition=run_world),
        ExecuteProcess(cmd=[sim_arm, sim_arm_config], output='screen', condition=run_world),
    ])
