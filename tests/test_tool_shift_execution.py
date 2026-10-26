"""Relative shift guards tested with fake actions; never connect to a robot."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from plan_tool_shift import execute_shift, main
finally:
    sys.path.pop(0)


class ShiftExecutionTests(unittest.TestCase):
    def setUp(self):
        try:
            from geometry_msgs.msg import Pose
            from moveit_msgs.msg import RobotTrajectory
            from trajectory_msgs.msg import JointTrajectoryPoint
            import rclpy.action
        except ImportError:
            self.skipTest('Source ROS to test relative shift execution guards')
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.target = Pose()
        self.target.position.x, self.target.position.z = .4, .09
        self.target.orientation.x = 1.
        self.target.orientation.w = 0.
        self.trajectory = RobotTrajectory()
        joint = self.trajectory.joint_trajectory
        joint.joint_names = [f'fr3_joint{i}' for i in range(1, 8)]
        joint.points = [JointTrajectoryPoint(positions=[0.]*7), JointTrajectoryPoint(positions=[.01]*7)]
        joint.points[-1].time_from_start.sec = 2
        self.feedback = SimpleNamespace(stationary=Mock(), fresh=Mock(return_value=np.zeros(7)),
                                        guard_moved=False, lock=threading.Lock())
        self.buffer = Mock()
        self.buffer.lookup_transform.return_value.transform = SimpleNamespace(
            translation=SimpleNamespace(x=.4, y=0., z=.09),
            rotation=SimpleNamespace(x=1., y=0., z=0., w=0.))
        from builtin_interfaces.msg import Time
        self.buffer.lookup_transform.return_value.header.stamp = Time()
        self.node = Mock()
        self.node.get_clock.return_value.now.return_value.nanoseconds = 0
        self.health = Mock()
        self.health.check.return_value = dict(robot_mode=2)
        self.health.snapshot.return_value = dict(robot_mode=2)
        self.client = Mock()
        self.client.wait_for_server.return_value = True
        self.config = dict(base_frame='fr3_link0', ee_frame='fr3_hand_tcp',
                           arrival_abort=.003, arrival_angle_deg=2., feedback_max_age_s=.5)

    def run_shift(self):
        with patch('rclpy.action.ActionClient', return_value=self.client), \
             patch('lego_sorter.hardware.RobotHealth', return_value=self.health), \
             patch('lego_sorter.hardware.SafeMover.execute') as execute:
            self.execute = execute
            execute_shift(self.node, self.buffer, self.feedback, self.config,
                          self.trajectory, self.target, self.output)

    def test_moved_robot_refuses_before_sending_action(self):
        self.feedback.guard_moved = True
        with self.assertRaisesRegex(RuntimeError, 'moved during'):
            self.run_shift()
        self.execute.assert_not_called()
        self.client.destroy.assert_called_once()
        self.assertEqual(json.loads((self.output / 'shift_execution.json').read_text())['status'],
                         'REFUSED_BEFORE_EXECUTION')

    def test_changed_start_refuses_before_sending_action(self):
        self.feedback.fresh.return_value = np.full(7, .02)
        with self.assertRaisesRegex(RuntimeError, 'no longer matches'):
            self.run_shift()
        self.execute.assert_not_called()

    def test_unhealthy_robot_refuses_before_sending_action(self):
        self.health.wait_ready.side_effect = RuntimeError('Physical Franka robot time is not advancing')
        with self.assertRaisesRegex(RuntimeError, 'not advancing'):
            self.run_shift()
        self.execute.assert_not_called()

    def test_valid_shift_uses_shared_adapter_and_checks_arrival(self):
        self.run_shift()
        self.execute.assert_called_once_with('RELATIVE_HOVER_SHIFT', self.trajectory)
        self.assertEqual(json.loads((self.output / 'shift_execution.json').read_text())['status'], 'COMPLETE')

    def test_bad_arrival_is_reported_as_failure(self):
        self.buffer.lookup_transform.return_value.transform.translation.z = .10
        with self.assertRaisesRegex(RuntimeError, 'arrival check failed'):
            self.run_shift()
        self.assertEqual(json.loads((self.output / 'shift_execution.json').read_text())['status'],
                         'INTERRUPTED_OR_FAILED')

    def test_execution_requires_explicit_signed_distance(self):
        with patch.object(sys, 'argv', ['plan_tool_shift.py', '--execute']), \
             patch('sys.stderr'), self.assertRaises(SystemExit) as stopped:
            main()
        self.assertEqual(stopped.exception.code, 2)
