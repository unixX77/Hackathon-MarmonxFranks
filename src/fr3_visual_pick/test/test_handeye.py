"""Offline checks of transform direction, observability, quality gates and board pose."""
import copy
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('handeye', ROOT / 'scripts/handeye_calibrate.py')
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


class HandeyeTest(unittest.TestCase):
    def setUp(self):
        self.c = yaml.safe_load((ROOT / 'config/handeye.yaml').read_text())
        # Synthetic fixture, independent of unfinished physical print measurements.
        self.c['board']['marker_length'] = 0.035
        self.c['board']['marker_separation'] = 0.010
        rng = np.random.default_rng(42)
        self.x = h.matrix(Rotation.from_euler('xyz', [.2, -.1, .3]).as_matrix(), [.04, -.02, .09])
        target = h.matrix(Rotation.from_euler('xyz', [.1, .2, -.1]).as_matrix(), [.5, .1, .02])
        self.data = {'camera_frame': 'camera_color_optical_frame', 'samples': [],
                     'T_mount_camera': h.matrix(Rotation.from_euler('xyz', [-1.57, 0, -1.57]).as_matrix(), [.01, 0, 0]).tolist()}
        for _ in range(25):
            robot = h.matrix(Rotation.from_rotvec(rng.normal(0, .4, 3)).as_matrix(),
                             rng.uniform([.3, -.2, .3], [.6, .2, .6]))
            camera_target = np.linalg.inv(self.x) @ np.linalg.inv(robot) @ target
            self.data['samples'].append({'T_base_ee': robot.tolist(),
                                         'T_camera_target': camera_target.tolist()})

    def test_recovers_camera_to_ee_and_mount_direction(self):
        result = h.solve(self.data, self.c)
        self.assertEqual(result['status'], 'passed_quality_gates')
        np.testing.assert_allclose(result['T_ee_camera'], self.x, atol=1e-9)
        expected = self.x @ np.linalg.inv(self.data['T_mount_camera'])
        np.testing.assert_allclose(result['T_ee_mount'], expected, atol=1e-9)
        self.assertIn('--child-frame-id camera_link', result['mount_tf_command_after_validation'])

    def test_rejects_translation_only_and_single_axis(self):
        for single_axis in (False, True):
            data = copy.deepcopy(self.data)
            for i, s in enumerate(data['samples']):
                t = np.array(s['T_base_ee'])
                t[:3, :3] = Rotation.from_euler('z', i * .04 if single_axis else 0).as_matrix()
                s['T_base_ee'] = t.tolist()
            with self.assertRaisesRegex(ValueError, 'Insufficient rotation'):
                h.solve(data, self.c)

    def test_moving_target_fails_quality_gates(self):
        data = copy.deepcopy(self.data)
        data['samples'][5]['T_camera_target'][0][3] += .15
        result = h.solve(data, self.c)
        self.assertEqual(result['status'], 'REJECTED')
        self.assertNotIn('mount_tf_command_after_validation', result)

    def test_invalid_and_too_few_samples(self):
        with self.assertRaises(ValueError):
            h.rigid(np.zeros((4, 4)))
        self.data['samples'] = self.data['samples'][:4]
        with self.assertRaises(ValueError):
            h.solve(self.data, self.c)

    def test_unmeasured_board_gap_is_rejected(self):
        self.c['board']['marker_separation'] = None
        with self.assertRaisesRegex(ValueError, 'board.marker_separation'):
            h.make_board(self.c)

    def test_validation_uses_original_reference_without_refitting(self):
        result = h.solve(self.data, self.c)
        checked = h.validate_result(self.data, result, self.c)
        self.assertEqual(checked['status'], 'passed_quality_gates')
        changed = copy.deepcopy(self.data)
        for s in changed['samples']:
            s['T_camera_target'][0][3] += .1
        self.assertEqual(h.validate_result(changed, result, self.c)['status'], 'REJECTED')

    def test_rendered_board_pose(self):
        _, board = h.make_board(self.c)
        b = self.c['board']
        width = b['columns'] * b['marker_length'] + (b['columns'] - 1) * b['marker_separation']
        height = b['rows'] * b['marker_length'] + (b['rows'] - 1) * b['marker_separation']
        image = board.draw((round(width * 4000), round(height * 4000))) if hasattr(board, 'draw') else board.generateImage((round(width * 4000), round(height * 4000)))
        k = np.array([[900., 0, 640], [0, 900., 480], [0, 0, 1]])
        rot = Rotation.from_euler('xyz', [2.8, .2, .05]).as_matrix()
        expected = h.matrix(rot, [-.1, .08, .55])
        corners = np.array([[0, height, 0], [width, height, 0], [width, 0, 0], [0, 0, 0]], float)
        projected = cv2.projectPoints(corners, cv2.Rodrigues(rot)[0], expected[:3, 3], k, None)[0].reshape(4, 2).astype(np.float32)
        src = np.array([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]], np.float32)
        rendered = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, projected), (1280, 960), borderValue=255)
        info = {'k': k.flatten().tolist(), 'd': [0.] * 5, 'distortion_model': 'plumb_bob'}
        recovered, error, markers = h.detect_target(rendered, info, self.c)
        self.assertGreaterEqual(markers, 5)
        self.assertLess(error, 1.)
        self.assertLess(np.linalg.norm(recovered[:3, 3] - expected[:3, 3]), .003)
        self.assertLess(Rotation.from_matrix(rot.T @ recovered[:3, :3]).magnitude(), .03)
        with self.assertRaises(ValueError):
            h.detect_target(np.full((960, 1280), 255, np.uint8), info, self.c)


if __name__ == '__main__':
    unittest.main()
