#  Copyright (c) 2026 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the 'License');
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an 'AS IS' BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Unified Gazebo launch test for all franka_gazebo_bringup bring-up launch files.

This single file is parametrized over (launch_file, launch_arguments) tuples,
replacing the previous per-launch-file test scripts. Each case includes the
launch file, waits for the system to come up, and asserts that no unexpected
[ERROR] messages were logged.

Runs on Humble with Ignition Fortress (CLI ``ign``), so teardown targets the
``ign gazebo`` process family rather than ``gz sim``.
"""

import subprocess
import unittest

from launch import (
    actions,
    launch_description_sources,
    LaunchDescription,
    substitutions,
)
import launch_ros.substitutions
import launch_testing
import launch_testing.actions
import rclpy

TEST_DURATION = 5.0  # sec

# Known-benign ERROR-level messages that do NOT indicate a real failure.
# Each entry is a substring; any ERROR line matching an entry is excluded from
# the assertion.  Keep this list tight — every entry must have a justification.
KNOWN_BENIGN_ERRORS = [
    # gz_ros2_control plugin races with robot_state_publisher during startup.
    # The service appears shortly after and the system functions normally.
    'robot_state_publisher service not available',
]

# Server-only, headless Gazebo args for fast setup/teardown of the launch files
# that expose a 'gz_args' argument.
HEADLESS_GZ_ARGS = 'empty.sdf -r -s --headless-rendering'

# Each entry: (launch_file, {launch_arguments})
#
# gravity_compensation_example_controller and joint_position_example_controller
# are intentionally omitted from the arm cases: they need state/command
# interfaces (force/torque, fr3/robot_time) that only the real hardware
# interface exports, so they fail deterministically under GazeboSimSystem.
params = [
    # --- Franka arm example controllers (sim-capable controllers only) ---
    (
        'gazebo_franka_arm_example_controller.launch.py',
        {
            'robot_type': 'fr3',
            'controller': 'joint_impedance_example_controller',
            'gz_args': HEADLESS_GZ_ARGS,
            'rviz': 'false',
        },
    ),
    (
        'gazebo_franka_arm_example_controller.launch.py',
        {
            'robot_type': 'fr3',
            'controller': 'joint_velocity_example_controller',
            'gz_args': HEADLESS_GZ_ARGS,
            'rviz': 'false',
        },
    ),
    # --- TMR example controller ---
    (
        'gazebo_tmr_example_controller.launch.py',
        {
            'gz_args': HEADLESS_GZ_ARGS,
            'rviz': 'false',
        },
    ),
    # --- Mobile robot (tmrv0_2) with chained swerve stack in mobile_base ns ---
    # This launch file does not expose a 'gz_args' argument (it builds gz_args
    # internally from the 'world' argument), so headless/server flags cannot be
    # injected here. RViz is disabled via 'use_rviz'.
    (
        'gazebo_mobile_robot.launch.py',
        {
            'with_sensors': 'false',
            'world': '',
            'use_rviz': 'false',
        },
    ),
]


def ensure_gz_sim_not_running():
    """
    Kill any remaining Gazebo/Ignition and controller processes.

    On CI, Ignition and the controller_manager may not shut down in time
    between parametrized tests, causing the next test to fail because
    controllers are still active. We forcefully kill them here.
    See https://github.com/ros2/launch/issues/545 for details.
    On Humble (Ignition Fortress) the sim process is 'ign gazebo', not 'gz sim'.
    """
    patterns = [
        '^gz sim',        # Gazebo Garden+ (gz-sim)
        'ign gazebo',     # Ignition Fortress (Humble)
        'ignition',       # Ignition sub-processes
        'ruby.*ign',      # Ruby launcher for Ignition
        'gzserver',       # Classic Gazebo server (fallback)
        'controller_manager',    # Lingering controller_manager nodes
        'robot_state_publisher',  # Lingering robot_state_publisher nodes
    ]
    for pattern in patterns:
        subprocess.run(['pkill', '-9', '-f', pattern], check=False)


@launch_testing.parametrize('launch_file, launch_args', params)
def generate_test_description(launch_file, launch_args):
    """Generate the test launch description for a given launch file."""
    ensure_gz_sim_not_running()

    launch_description = actions.IncludeLaunchDescription(
        launch_description_sources.PythonLaunchDescriptionSource(
            substitutions.PathJoinSubstitution(
                [
                    launch_ros.substitutions.FindPackageShare(
                        'franka_gazebo_bringup'
                    ),
                    'launch',
                    launch_file,
                ]
            )
        ),
        launch_arguments=launch_args.items(),
    )

    test_description = (
        LaunchDescription(
            [
                launch_description,
                actions.TimerAction(
                    period=TEST_DURATION,
                    actions=[launch_testing.actions.ReadyToTest()],
                ),
            ],
        ),
        {'launch_description': launch_description},
    )
    return test_description


class TestGazeboLaunch(unittest.TestCase):
    """Verify that the Gazebo bring-up launch files start without errors."""

    @classmethod
    def setUpClass(cls):
        """Initialize the ROS context."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown the ROS context and clean up lingering sim processes."""
        rclpy.shutdown()
        ensure_gz_sim_not_running()

    def test_has_no_error(self, proc_output):
        """
        Check that no unexpected ERROR messages appear in launch output.

        Lines matching KNOWN_BENIGN_ERRORS are excluded — these are transient
        startup races that resolve on their own and do not affect functionality.
        """
        error_lines = []
        for event in proc_output:
            if not event.from_stderr:
                continue
            text = event.text.decode('utf-8', errors='replace')
            for line in text.splitlines():
                if 'ERROR' not in line:
                    continue
                if any(pattern in line for pattern in KNOWN_BENIGN_ERRORS):
                    continue
                error_lines.append(line)

        assert not error_lines, (
            'Found unexpected [ERROR] log messages in launch output:\n'
            + '\n'.join(error_lines)
        )
