"""Replay must plan only by default, and must never execute after plan failure."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

try:
    import rclpy.action
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('handeye', ROOT / 'scripts/handeye_calibrate.py')
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


@unittest.skipUnless(ROS_AVAILABLE, 'Source ROS Humble for message-construction tests')
class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.c = yaml.safe_load((ROOT / 'config/handeye.yaml').read_text())
        self.collector = SimpleNamespace(c=self.c, node=object())
        self.source = {'samples': [{'joint_positions': [0.] * 7}, {'joint_positions': [.1] * 7}]}
        self.client = MagicMock()
        self.client.wait_for_server.return_value = True

    def test_default_only_plans_and_seeds_next_plan(self):
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = h.JOINTS
        trajectory.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.] * 7)]
        with patch.object(h, 'collision_scene'), patch('rclpy.action.ActionClient', return_value=self.client) as clients:
            with patch.object(h, 'action', return_value=SimpleNamespace(planned_trajectory=trajectory)) as action:
                h.replay(self.collector, self.source, '/tmp/unused_replay_test.json', False)
                self.assertEqual(clients.call_count, 1)
                self.assertEqual(clients.call_args.args[2], 'move_action')
                for call in action.call_args_list:
                    self.assertTrue(call.args[2].planning_options.plan_only)
                    self.assertEqual(call.args[2].request.max_velocity_scaling_factor, .03)
                second = action.call_args_list[1].args[2].request
                self.assertEqual(list(second.start_state.joint_state.name), h.JOINTS)

    def test_planning_failure_prevents_execution_even_when_requested(self):
        with patch.object(h, 'collision_scene'), patch('rclpy.action.ActionClient', return_value=self.client) as clients:
            with patch.object(h, 'action', side_effect=RuntimeError('planning failed')):
                with self.assertRaisesRegex(RuntimeError, 'planning failed'):
                    h.replay(self.collector, self.source, '/tmp/unused_replay_test.json', True)
                self.assertEqual(clients.call_count, 1)
                self.assertEqual(clients.call_args.args[2], 'move_action')

    def test_missing_collision_scene_prevents_motion_clients(self):
        with patch.object(h, 'collision_scene', side_effect=RuntimeError('geometry missing')):
            with patch('rclpy.action.ActionClient') as clients:
                with self.assertRaisesRegex(RuntimeError, 'geometry missing'):
                    h.replay(self.collector, self.source, '/tmp/unused_replay_test.json', True)
                clients.assert_not_called()


if __name__ == '__main__':
    unittest.main()
