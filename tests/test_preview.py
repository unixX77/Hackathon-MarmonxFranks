"""Timing validation uses dictionaries and never contacts ROS or a robot."""
import copy
import math
import unittest

from lego_sorter.preview import audit_display, align_display_start, PreviewSampler, horizontal_tool_shift


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.display = dict(trajectory=[dict(joint_trajectory=dict(joint_names=['joint'], points=[
            dict(positions=[0.], time_from_start=dict(sec=0, nanosec=0)),
            dict(positions=[1.], time_from_start=dict(sec=45, nanosec=743231498))]))])

    def test_seconds_and_nanoseconds_are_preserved(self):
        before = copy.deepcopy(self.display)
        report = audit_display(self.display)['trajectories'][0]
        self.assertEqual(report['last_stamp_s'], 45.743231498)
        self.assertEqual(report['fixed_005s_nominal_duration_s'], .1)
        self.assertEqual(self.display, before)

    def test_horizontal_tool_x_shift_rotates_into_base_xy_and_holds_z(self):
        position = [.4, 0., .09]
        yaw = math.radians(21)
        quaternion = [math.cos(yaw/2), math.sin(yaw/2), 0., 0.]
        result = horizontal_tool_shift(position, quaternion, .02)
        self.assertAlmostEqual(result[0], .4 + .02*math.cos(yaw))
        self.assertAlmostEqual(result[1], .02*math.sin(yaw))
        self.assertEqual(result[2], .09)
        self.assertEqual(position, [.4, 0., .09])
        reverse = horizontal_tool_shift(position, quaternion, -.02)
        self.assertAlmostEqual(reverse[0], .4 - .02*math.cos(yaw))
        self.assertAlmostEqual(reverse[1], -.02*math.sin(yaw))
        self.assertEqual(reverse[2], .09)

    def test_shift_rejects_large_distance_and_wrong_tool_orientation(self):
        for distance, quaternion in ((.026, [1., 0., 0., 0.]),
                                     (.02, [0., 0., 0., 1.]),
                                     (float('nan'), [1., 0., 0., 0.])):
            with self.subTest(distance=distance, quaternion=quaternion):
                with self.assertRaises(ValueError):
                    horizontal_tool_shift([.4, 0., .09], quaternion, distance)

    def test_sampler_uses_elapsed_seconds_and_preserves_original(self):
        before = copy.deepcopy(self.display)
        sampler = PreviewSampler(self.display)
        self.assertEqual(sampler.state(0)['joint_state']['position'], [0.])
        self.assertAlmostEqual(sampler.state(sampler.duration/2)['joint_state']['position'][0], .5)
        self.assertEqual(sampler.state(sampler.duration)['joint_state']['position'], [1.])
        self.assertEqual(sampler.state(100)['joint_state']['position'], [1.])
        self.assertEqual(self.display, before)

    def test_sampler_uses_waypoint_velocities_for_cubic_interpolation(self):
        for point in self.display['trajectory'][0]['joint_trajectory']['points']:
            point['velocities'] = [0.]
        sampler = PreviewSampler(self.display)
        self.assertAlmostEqual(sampler.state(sampler.duration/4)['joint_state']['position'][0], .15625)

    def test_sampler_rejects_nonfinite_time(self):
        sampler = PreviewSampler(self.display)
        with self.assertRaises(ValueError):
            sampler.state(float('nan'))

    def test_duplicate_or_backwards_timestamps_are_rejected(self):
        for stamp in (dict(sec=0, nanosec=0), dict(sec=-1, nanosec=0),
                      dict(sec=0, nanosec=10**9)):
            with self.subTest(stamp=stamp):
                self.display['trajectory'][0]['joint_trajectory']['points'][1]['time_from_start'] = stamp
                with self.assertRaises(ValueError):
                    audit_display(self.display)

    def test_corrupt_joint_data_is_rejected(self):
        for positions in ([], [float('nan')], [0., 1.]):
            with self.subTest(positions=positions):
                self.display['trajectory'][0]['joint_trajectory']['points'][1]['positions'] = positions
                with self.assertRaises(ValueError):
                    audit_display(self.display)

    def test_digest_detects_changed_timing(self):
        before = audit_display(self.display)['display_sha256']
        self.display['trajectory'][0]['joint_trajectory']['points'][1]['time_from_start']['sec'] = 4
        self.assertNotEqual(audit_display(self.display)['display_sha256'], before)

    def test_endpoint_start_state_is_corrected_without_changing_path_or_fingers(self):
        self.display['trajectory_start'] = dict(is_diff=True, joint_state=dict(
            name=['joint', 'finger'], position=[1., .04], velocity=[.1, .0]))
        before = copy.deepcopy(self.display)
        fixed, adjustment = align_display_start(self.display)
        self.assertEqual(adjustment, 1.)
        self.assertEqual(fixed['trajectory_start']['joint_state']['position'], [0., .04])
        self.assertEqual(fixed['trajectory_start']['joint_state']['velocity'], [])
        self.assertFalse(fixed['trajectory_start']['is_diff'])
        self.assertEqual(fixed['trajectory'], before['trajectory'])
        self.assertEqual(self.display, before)
