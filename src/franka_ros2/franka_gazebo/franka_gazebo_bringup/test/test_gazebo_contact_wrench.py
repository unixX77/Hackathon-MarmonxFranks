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
Gazebo contact-response test for the estimated external wrench.

Brings up an fr3 with the gravity_compensation_example_controller (which logs the
stiffness-frame external wrench O_F_ext_hat_K every 100 cycles), settles it, then
injects a known world-frame +Z force on fr3_link7 via Gazebo's ApplyLinkWrench
system and verifies the estimate reports the contact in the stiffness frame K. The
controller logs K_F_ext_hat_K, the base-frame external wrench rotated into the
stiffness frame K by O_R_EE^T. That rotation is pure (orthonormal), so it preserves
magnitude but redistributes a base-frame +Z push across all three K-frame axes
according to the fr3 ready-pose orientation. The primary, pose-robust check is
therefore on the force-triple NORM (must stay ~= the applied force); the per-axis
split is corroborated against the measured ready-pose distribution. The at-rest
baseline (~0) is covered by the existing launch suite; this exercises the loaded
path the baseline cannot.
"""

import math
import os
import re
import shutil
import statistics
import subprocess
import time
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
import launch_testing.markers
import rclpy

# On Ignition Fortress the command-line tool is `ign` (installed at /usr/bin). Locate
# its directory so the test process can find it regardless of how PATH is set up.
GZ_TOOLS_BIN = os.path.dirname(shutil.which('ign') or '/usr/bin/ign')

# World name from empty.sdf and the link the spike characterized the estimate on.
WORLD_NAME = 'empty'
LINK_NAME = 'fr3_link7'

# Ignition Fortress' entity/system/add service resolves the target by entity id, not
# by name+type. The world is the first entity the runner creates from empty.sdf, so it
# has id 1; the robot model is spawned afterwards and never takes this id.
WORLD_ENTITY_ID = 1

# Applied world-frame +Z force (N) and the acceptance bands for the measured K-frame
# force. The controller logs the stiffness-frame wrench (tcp_wrench_ = K_F_ext_hat_K),
# which is the base-frame external wrench rotated into frame K by O_R_EE^T. That
# rotation is pure, so it conserves magnitude: the force-triple norm must stay ~= the
# applied 30 N regardless of pose. This norm band is the primary gate.
#
# The individual K-frame components depend on the fr3 ready-pose orientation; a base
# +Z push does NOT read as a pure Z force in K. The measured wrench follows the reaction
# convention (it opposes the applied load, push +x -> read -x). Measured ready-pose
# distribution under the +Z push: F_x ~= -22, F_y ~= +22, F_z ~= -3 N. The per-axis
# EXPECTED constants corroborate that split with a wide tolerance (measured on the
# Jazzy/gz twin; Fortress kinematics are the same ready pose so these should be
# near-identical, tightened later if needed).
APPLIED_FORCE_Z = 30.0
CONTACT_FORCE_NORM_MIN = 25.0
CONTACT_FORCE_NORM_MAX = 35.0
CONTACT_FORCE_X_EXPECTED = -22.0
CONTACT_FORCE_Y_EXPECTED = 22.0
CONTACT_FORCE_Z_EXPECTED = -3.0
# Per-axis tolerance (N) around the expected K-frame distribution under load.
CONTACT_PER_AXIS_TOLERANCE = 8.0
# At-rest tolerance (N): all axes must sit inside +/- this band with no load applied.
NEAR_ZERO_TOLERANCE = 3.0

# Number of trailing wrench samples to take the per-axis median over. A single tail
# sample can land on a sub-second overshoot; the median of the last few is robust to it.
WRENCH_MEDIAN_WINDOW = 10

# Seconds to let Gazebo, the controller_manager and the controller spin up and the
# arm settle at its home pose before the test body runs.
SPINUP_DURATION = 12.0
# Seconds to wait for the first logged wrench line before giving up.
FIRST_LOG_TIMEOUT = 30.0
# Seconds to let the injected force propagate and the estimate settle.
SETTLE_AFTER_INJECTION = 4.0

# Matches: External wrench (stiffness frame): F=[x, y, z] T=[...]
WRENCH_PATTERN = re.compile(
    r'External wrench.*?F=\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]'
)


def ensure_gz_sim_not_running():
    """
    Kill any remaining Gazebo and related ROS processes between tests.

    On CI, Gazebo and the controller_manager may not shut down in time between
    tests, causing the next test to fail because controllers are still active.
    See https://github.com/ros2/launch/issues/545 for details.
    """
    subprocess.run(['pkill', '-2', '-f', 'ign gazebo'], check=False)
    time.sleep(2)
    subprocess.run(['pkill', '-9', '-f', 'ign gazebo'], check=False)
    subprocess.run(['pkill', '-9', '-f', 'ruby.*ign'], check=False)
    subprocess.run(['pkill', '-9', '-f', 'controller_manager'], check=False)
    subprocess.run(['pkill', '-9', '-f', 'robot_state_publisher'], check=False)
    time.sleep(2)


def gz_environment():
    """Return an environment with the Gazebo tools prepended to PATH."""
    environment = dict(os.environ)
    environment['PATH'] = GZ_TOOLS_BIN + os.pathsep + environment.get('PATH', '')
    return environment


def load_apply_link_wrench_system():
    """Load the ApplyLinkWrench system into the running world at runtime."""
    request = (
        'entity: {{ id: {world_entity_id} }}, '
        'plugins: {{ filename: "ignition-gazebo-apply-link-wrench-system", '
        'name: "ignition::gazebo::systems::ApplyLinkWrench" }}'
    ).format(world_entity_id=WORLD_ENTITY_ID)
    subprocess.run(
        [
            'ign', 'service', '-s', '/world/{}/entity/system/add'.format(WORLD_NAME),
            '--reqtype', 'ign_msgs.EntityPlugin_V',
            '--reptype', 'ign_msgs.Boolean',
            '--req', request,
            '--timeout', '5000',
        ],
        env=gz_environment(),
        check=True,
    )


def apply_persistent_force_z(force_z):
    """Publish a persistent world-frame +Z force on the target link."""
    message = (
        'entity: {{ name: "{link}", type: LINK }}, '
        'wrench: {{ force: {{ x: 0, y: 0, z: {force} }} }}'
    ).format(link=LINK_NAME, force=force_z)
    subprocess.run(
        [
            'ign', 'topic', '-t', '/world/{}/wrench/persistent'.format(WORLD_NAME),
            '-m', 'ign_msgs.EntityWrench',
            '-p', message,
        ],
        env=gz_environment(),
        check=True,
    )


def event_text(output_event):
    """Decode a captured process IO event to text."""
    text = getattr(output_event, 'text', None)
    if text is None:
        return str(output_event)
    if isinstance(text, bytes):
        return text.decode('utf-8', errors='replace')
    return str(text)


def median_wrench_force(proc_output):
    """
    Return the per-axis median of the most recent logged wrench forces, or None.

    Reading the single last sample can catch a sub-second overshoot in the sim
    dynamics; the median of the trailing window rejects those transients and reflects
    the steady-state estimate.
    """
    samples = []
    for output_event in proc_output:
        for match in WRENCH_PATTERN.finditer(event_text(output_event)):
            samples.append(
                (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                )
            )
    if not samples:
        return None
    window = samples[-WRENCH_MEDIAN_WINDOW:]
    return (
        statistics.median(axis_x for axis_x, _, _ in window),
        statistics.median(axis_y for _, axis_y, _ in window),
        statistics.median(axis_z for _, _, axis_z in window),
    )


@launch_testing.markers.keep_alive
def generate_test_description():
    """Bring up fr3 with the gravity_compensation controller in an empty world."""
    ensure_gz_sim_not_running()

    launch_description = actions.IncludeLaunchDescription(
        launch_description_sources.PythonLaunchDescriptionSource(
            substitutions.PathJoinSubstitution(
                [
                    launch_ros.substitutions.FindPackageShare(
                        'franka_gazebo_bringup'
                    ),
                    'launch',
                    'gazebo_franka_arm_example_controller.launch.py',
                ]
            )
        ),
        launch_arguments={
            'robot_type': 'fr3',
            'controller': 'gravity_compensation_example_controller',
            'gz_args': 'empty.sdf -r -s --headless-rendering',
            'rviz': 'false',
        }.items(),
    )

    test_description = (
        LaunchDescription(
            [
                launch_description,
                actions.TimerAction(
                    period=SPINUP_DURATION,
                    actions=[launch_testing.actions.ReadyToTest()],
                ),
            ],
        ),
        {'launch_description': launch_description},
    )
    return test_description


class TestGazeboContactWrench(unittest.TestCase):
    """Verify the estimated external wrench reports an injected contact force."""

    @classmethod
    def setUpClass(cls):
        """Initialize the ROS context."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown the ROS context."""
        rclpy.shutdown()
        ensure_gz_sim_not_running()

    def test_contact_response(self, proc_output):
        """At rest the estimate is ~0; under a base +Z push the K-frame norm is ~30 N."""
        # The controller must be active and logging before we trust any reading.
        # rclcpp loggers write to stderr; the baseline and contact reads below
        # consume the whole proc_output, so this gate must watch stderr too.
        assert proc_output.waitFor(
            'External wrench', timeout=FIRST_LOG_TIMEOUT, stream='stderr'
        ), 'Controller never logged an external wrench'

        at_rest = median_wrench_force(proc_output)
        assert at_rest is not None, 'No external-wrench reading parsed at rest'
        for axis_value in at_rest:
            assert abs(axis_value) <= NEAR_ZERO_TOLERANCE, (
                'At-rest external wrench not near zero: {}'.format(at_rest)
            )

        load_apply_link_wrench_system()
        time.sleep(2.0)
        apply_persistent_force_z(APPLIED_FORCE_Z)
        time.sleep(SETTLE_AFTER_INJECTION)

        under_contact = median_wrench_force(proc_output)
        assert under_contact is not None, 'No external-wrench reading parsed under load'

        force_x, force_y, force_z = under_contact
        force_norm = math.sqrt(force_x**2 + force_y**2 + force_z**2)
        # Primary, pose-robust gate: the pure rotation into frame K conserves magnitude,
        # so the force-triple norm must land near the applied push.
        assert CONTACT_FORCE_NORM_MIN <= force_norm <= CONTACT_FORCE_NORM_MAX, (
            'Force norm {} N outside [{}, {}] N under a +{} N push (F={})'.format(
                force_norm, CONTACT_FORCE_NORM_MIN, CONTACT_FORCE_NORM_MAX,
                APPLIED_FORCE_Z, under_contact
            )
        )
        # Per-axis corroboration against the expected ready-pose K-frame distribution.
        assert abs(force_x - CONTACT_FORCE_X_EXPECTED) <= CONTACT_PER_AXIS_TOLERANCE, (
            'X force {} N off expected {} N by more than +/- {} N'.format(
                force_x, CONTACT_FORCE_X_EXPECTED, CONTACT_PER_AXIS_TOLERANCE
            )
        )
        assert abs(force_y - CONTACT_FORCE_Y_EXPECTED) <= CONTACT_PER_AXIS_TOLERANCE, (
            'Y force {} N off expected {} N by more than +/- {} N'.format(
                force_y, CONTACT_FORCE_Y_EXPECTED, CONTACT_PER_AXIS_TOLERANCE
            )
        )
        assert abs(force_z - CONTACT_FORCE_Z_EXPECTED) <= CONTACT_PER_AXIS_TOLERANCE, (
            'Z force {} N off expected {} N by more than +/- {} N'.format(
                force_z, CONTACT_FORCE_Z_EXPECTED, CONTACT_PER_AXIS_TOLERANCE
            )
        )
