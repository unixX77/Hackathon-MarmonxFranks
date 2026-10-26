"""Physical-state health checks without ROS or robot connections."""
import unittest
from lego_sorter.hardware import RobotHealthStatus


class RobotHealthTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.health = RobotHealthStatus(.5, clock=lambda: self.now)

    def advance(self):
        self.health.observe(10., 2, [])
        self.now = .1
        self.health.observe(10.1, 2, [])

    def test_advancing_physical_time_in_active_control_is_healthy(self):
        self.advance()
        self.assertEqual(self.health.check()['robot_mode'], 2)

    def test_fresh_publication_with_frozen_robot_time_is_rejected(self):
        self.advance()
        for t in (.2, .4, .8):
            self.now = t
            self.health.observe(10.1, 2, [])
        self.assertEqual(self.health.snapshot()['publication_age_s'], 0.)
        with self.assertRaisesRegex(RuntimeError, 'not advancing'):
            self.health.check()

    def test_reflex_and_current_errors_are_rejected(self):
        self.advance()
        self.health.observe(10.2, 4, ['communication_constraints_violation'])
        with self.assertRaisesRegex(RuntimeError, 'communication_constraints_violation'):
            self.health.check()
        self.health.observe(10.3, 4, [])
        with self.assertRaisesRegex(RuntimeError, 'not in active control'):
            self.health.check()

    def test_old_or_missing_publication_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'No fresh'):
            self.health.check()
        self.advance()
        self.now = 1.
        with self.assertRaisesRegex(RuntimeError, 'No fresh'):
            self.health.check()

    def test_first_sample_is_not_evidence_of_advancing_robot_time(self):
        self.health.observe(10., 2, [])
        with self.assertRaisesRegex(RuntimeError, 'not advancing'):
            self.health.check()

    def test_readonly_capture_can_use_guiding_feedback_without_admitting_motion(self):
        self.advance()
        self.health.observe(10.2, 3, [])
        self.assertEqual(self.health.check_feedback()['robot_mode'], 3)
        with self.assertRaisesRegex(RuntimeError, 'not in active control'):
            self.health.check()
