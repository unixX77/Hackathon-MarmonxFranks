"""Mock identity guards, fake feedback routing and finger animation tests."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import fake_execution as f
import pick
from builtin_interfaces.msg import Time
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Float64MultiArray


def description(plugin='mock_components/GenericSystem', names=f.ARM):
    joints = ''.join(f'<joint name="{n}"/>' for n in names)
    return f'<robot name="test"><ros2_control name="arm" type="system"><hardware><plugin>{plugin}</plugin></hardware>{joints}</ros2_control></robot>'


def component(plugin='mock_components/GenericSystem', active=3, names=f.ARM):
    return SimpleNamespace(class_type=plugin, state=SimpleNamespace(id=active),
                           command_interfaces=[SimpleNamespace(name=f'{n}/position') for n in names])


class IdentityGuards(unittest.TestCase):
    def test_only_complete_mock_descriptions_are_accepted(self):
        f.verify_mock_description(description())
        for urdf in (description('franka_hardware/FrankaHardwareInterface'),
                     description(names=f.ARM[:-1]), '<robot name="empty"/>'):
            with self.assertRaises(f.Intervention):
                f.verify_mock_description(urdf)

    def test_only_active_unique_mock_components_are_accepted(self):
        f.verify_mock_components([component()])
        for components in ([component('franka_hardware/FrankaHardwareInterface')],
                           [component(active=2)], [component(names=f.ARM[:-1])],
                           [component(), component()], []):
            with self.assertRaises(f.Intervention):
                f.verify_mock_components(components)

    def test_real_description_rejected_before_creating_command_publisher(self):
        task = SimpleNamespace(robot_description=description('franka_hardware/FrankaHardwareInterface'), node=MagicMock())
        with self.assertRaises(f.Intervention):
            f.FakeExecution(task)
        task.node.create_publisher.assert_not_called()

    def test_real_and_fake_execute_flags_are_mutually_exclusive(self):
        with patch.object(sys, 'argv', ['pick.py', '--config', 'unused', '--camera-target', 'unused', '--execute', '--fake-execute']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                pick.run_cli()
        self.assertEqual(result.exception.code, 2)


class FeedbackTests(unittest.TestCase):
    def backend(self):
        b = object.__new__(f.FakeExecution)
        b.task = SimpleNamespace(node=MagicMock())
        b.task.node.get_clock().now().nanoseconds = int(10e9)
        b.controller_state = JointTrajectoryControllerState()
        b.controller_state.header.stamp.sec = 10
        b.controller_state.joint_names = f.ARM
        b.controller_state.error.positions = [.001] * 7
        b.received = time.monotonic()
        return b

    def test_controller_feedback_drives_fake_tracking_without_native_franka_state(self):
        b = self.backend()
        self.assertEqual(b.tracking_error(), .001)
        b.received -= 1
        with self.assertRaises(f.Intervention):
            b.tracking_error()

    def test_incomplete_and_nonfinite_feedback_are_rejected(self):
        b = self.backend()
        b.controller_state.error.positions = []
        with self.assertRaises(f.Intervention):
            b.check_health()
        b.controller_state.error.positions = [float('nan')]*7
        with self.assertRaises(f.Intervention):
            b.check_health()

    def test_fake_hand_never_calls_real_hand_actions(self):
        p = object.__new__(pick.Picker)
        p.fake_backend = MagicMock()
        p.c = dict(open_width=.04, grasp_width=.015, gripper_speed=.01)
        with patch.object(pick.Task, 'hand') as real_hand:
            p.command_hand()
            p.command_hand(close=True)
        real_hand.assert_not_called()
        self.assertEqual(p.fake_backend.hand.call_args_list[0].args, (.04, .01))
        self.assertEqual(p.fake_backend.hand.call_args_list[1].args, (.015, .01))


class FingerAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / 'franka_ros2/franka_gripper/scripts/fake_gripper_state_publisher.py'
        spec = importlib.util.spec_from_file_location('test_fake_fingers', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def publisher(self):
        p = object.__new__(self.module.FakeGripperStatePublisher)
        p.width = p.target_width = .07
        p.speed = .01
        p.previous_time = 0.
        p.joint_names = ['fr3_finger_joint1', 'fr3_finger_joint2']
        p.publisher_ = MagicMock()
        p.get_clock = MagicMock()
        p.get_clock.return_value.now.return_value.to_msg.return_value = Time(sec=10)
        p.get_logger = MagicMock()
        return p

    def test_width_command_animates_both_fingers_at_configured_speed(self):
        p = self.publisher()
        p.on_command(Float64MultiArray(data=[.02, .01]))
        with patch.object(self.module.time, 'monotonic', return_value=.1):
            p.publish_state()
        message = p.publisher_.publish.call_args.args[0]
        self.assertAlmostEqual(sum(message.position), .069)
        self.assertAlmostEqual(message.position[0], message.position[1])
        self.assertAlmostEqual(message.velocity[0], -.005)

    def test_invalid_width_commands_leave_existing_target_unchanged(self):
        p = self.publisher()
        for data in ([.2, .01], [.02, 0.], [.02], [float('nan'), .01]):
            p.on_command(Float64MultiArray(data=data))
            self.assertEqual(p.target_width, .07)


if __name__ == '__main__':
    unittest.main()
