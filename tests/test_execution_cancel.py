"""Exercise cancellation with fake futures; never connect to a robot."""
from concurrent.futures import Future
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from lego_sorter.hardware import SafeMover


class ExecutionCancellationTests(unittest.TestCase):
    def setUp(self):
        try:
            from action_msgs.msg import GoalStatus
        except ImportError:
            self.skipTest('Source ROS to test the shared trajectory action adapter')
        self.result = Future()
        self.handle = Mock(accepted=True)
        self.handle.get_result_async.return_value = self.result
        def cancel():
            if not self.result.done():
                self.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
            done = Future()
            done.set_result(None)
            return done
        self.handle.cancel_goal_async.side_effect = cancel
        self.pending = Future()
        self.client = Mock()
        self.client.send_goal_async.return_value = self.pending
        self.runner = SimpleNamespace(cancel=threading.Event(), feedback=Mock())

    def test_keyboard_interrupt_cancels_an_accepted_goal(self):
        self.pending.set_result(self.handle)
        self.runner.feedback.fresh.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            SafeMover.action(self.runner, self.client, object(), 1)
        self.handle.cancel_goal_async.assert_called_once()
        self.assertTrue(self.runner.cancel.is_set())

    def test_interrupt_during_acknowledgement_cancels_late_accepted_goal(self):
        with patch('lego_sorter.hardware.wait_future', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                SafeMover.action(self.runner, self.client, object(), 1)
        self.pending.set_result(self.handle)
        self.handle.cancel_goal_async.assert_called_once()

    def test_robot_health_failure_during_action_cancels_goal(self):
        self.pending.set_result(self.handle)
        self.runner.health = Mock()
        self.runner.health.check.side_effect = [None, RuntimeError('Physical robot time is not advancing')]
        with self.assertRaisesRegex(RuntimeError, 'not advancing'):
            SafeMover.action(self.runner, self.client, object(), 1)
        self.handle.cancel_goal_async.assert_called_once()
        self.assertTrue(self.runner.cancel.is_set())

    def test_robot_health_failure_before_action_sends_no_goal(self):
        self.runner.health = Mock()
        self.runner.health.check.side_effect = RuntimeError('Franka reflex')
        with self.assertRaisesRegex(RuntimeError, 'reflex'):
            SafeMover.action(self.runner, self.client, object(), 1)
        self.client.send_goal_async.assert_not_called()
