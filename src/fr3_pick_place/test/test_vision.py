"""Deterministic math, measurement gates and workflow tests without hardware/SAM3."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'vision_vendor'))
import vision_core as core
from vision_pick_place import VisionTask, pose_matrix, ros_pose
from top_layer_bricks import apply_size_prior
from top_layer_depth import filter_top
from vision_perception import Perception


def config():
    return core.validate_config(yaml.safe_load((ROOT / 'config/vision_demo.yaml').read_text()))


class VisionMath(unittest.TestCase):
    def test_real_template_refuses_unmeasured_values(self):
        c = yaml.safe_load((ROOT / 'config/vision_real.yaml').read_text())
        with self.assertRaises(ValueError):
            core.validate_config(c)

    def test_pose_roundtrip_and_camera_chain(self):
        obj = core.matrix({'xyz_m': [.1, .2, .3], 'quaternion_xyzw': [0., 0., 0., 1.]})
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        camera[:3, 3] = [1., 0., 0.]
        np.testing.assert_allclose((camera @ obj)[:3, 3], [.8, .1, .3])
        np.testing.assert_allclose(pose_matrix(ros_pose(obj)), obj)

    def test_teammate_fitter_distinguishes_top_and_box_centres(self):
        instance = np.zeros((512, 512), np.uint16)
        instance[200:232, 200:262] = 1
        report = {'table_normal': [0., 0., -1.], 'table_offset_m': .5}
        calibration = {'aligned_depth_intrinsics': {'K': [[1000., 0., 256.], [0., 1000., 256.], [0., 0., 1.]]}}
        args = SimpleNamespace(brick_size_mm=[30., 15., 10.], size_tol=.3)
        _, bricks = apply_size_prior(instance, [{'id': 1, 'height_m': .01}],
            [{'id': 1, 'colour': 'red', 'class_id': 1, 'quality': 'good'}], calibration, report, args)
        self.assertEqual(bricks[0]['status'], 'brick')
        self.assertAlmostEqual(bricks[0]['centre_cam_m'][2], .49)
        self.assertAlmostEqual(bricks[0]['box_centre_cam_m'][2], .495)

    def test_grasp_offset_and_bounded_correction(self):
        c = config()
        ref = core.Refinement(c)
        obj = ref.observe(dict(c['mock']['object_base'], colour='red'))
        current = ref.grasp_pose(obj)
        current[0, 3] -= .04
        current[2, 3] += .05
        decision, target = ref.next_pose(current, obj)
        self.assertEqual(decision, 'correct')
        self.assertAlmostEqual(target[0, 3]-current[0, 3], .01)
        self.assertEqual(target[2, 3], current[2, 3])
        self.assertAlmostEqual(ref.grasp_pose(obj)[2, 3], .378)

    def test_adapter_runs_teammate_geometry_and_restores_quality_metrics(self):
        worker = object.__new__(Perception)
        worker.c = config()
        worker.bridge = MagicMock()
        worker.model = worker.processor = object()
        worker.device, worker.dtype = 'cpu', None
        instance = np.zeros((512, 512), np.uint16)
        instance[200:232, 200:262] = 1
        depth = np.full((512, 512), 500, np.uint16)
        depth[instance == 1] = 490
        worker.bridge.imgmsg_to_cv2.side_effect = [np.zeros((512, 512, 3), np.uint8), depth]
        record = {'id': 1, 'colour': 'red', 'class_id': 1, 'quality': 'good',
                  'score': .9, 'touches_border': False}
        worker.segment = MagicMock(return_value=(None, instance, [record], None, None))
        worker.filter_top, worker.fit = filter_top, apply_size_prior
        transform = np.diag([1., -1., -1., 1.])
        transform[2, 3] = .9
        image = SimpleNamespace(width=512, height=512)
        info = SimpleNamespace(width=512, height=512, d=[0.]*5,
                               k=[1000., 0., 256., 0., 1000., 256., 0., 0., 1.])
        stamp, result = worker.infer((123, 0., image, image, info, transform))
        self.assertEqual(stamp, 123)
        self.assertAlmostEqual(result['xyz_m'][2], .405, places=3)
        self.assertEqual(result['colour'], 'red')

    def test_stability_required_before_final_descent(self):
        c = config()
        ref = core.Refinement(c)
        detection = dict(c['mock']['object_base'], colour='red')
        for i in range(3):
            obj = ref.observe(detection)
            current = ref.grasp_pose(obj)
            current[2, 3] += .05
            decision, _ = ref.next_pose(current, obj)
            self.assertEqual(decision, 'ready' if i == 2 else 'observe')

    def test_target_loss_and_iteration_budget(self):
        c = config()
        ref = core.Refinement(c)
        detection = dict(c['mock']['object_base'], colour='red')
        ref.observe(detection)
        moved = copy.deepcopy(detection)
        moved['xyz_m'][0] += .1
        with self.assertRaisesRegex(RuntimeError, 'lost'):
            ref.observe(moved)
        ref = core.Refinement(c)
        for _ in range(c['refinement']['max_iterations']):
            ref.observe(detection)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            ref.observe(detection)

    def test_half_turn_ambiguity_keeps_selected_jaw_orientation(self):
        c = config()
        ref = core.Refinement(c)
        first = dict(c['mock']['object_base'], colour='red')
        a = ref.observe(first)
        flipped = copy.deepcopy(first)
        flipped['quaternion_xyzw'] = [0., 0., 1., 0.]
        b = ref.observe(flipped)
        self.assertAlmostEqual(core.angle(a, b), 0.)

    def test_rejects_colour_quality_as_substitute_for_pose_metrics(self):
        c = config()
        report = {'table_median_residual_m': .001, 'table_inlier_fraction': .9, 'table_normal': [0., 0., 1.]}
        b = {'id': 1, 'status': 'brick', 'quality': 'good'}
        with self.assertRaisesRegex(RuntimeError, 'score'):
            core.normalize_bricks([b], report, np.eye(4), c)

    def test_too_long_final_descent_is_rejected(self):
        c = config()
        ref = core.Refinement(c)
        for _ in range(3):
            obj = ref.observe(dict(c['mock']['object_base'], colour='red'))
        current = ref.grasp_pose(obj)
        current[2, 3] += .2
        with self.assertRaisesRegex(RuntimeError, 'descent'):
            ref.next_pose(current, obj)


class Workflow(unittest.TestCase):
    def test_support_contact_is_specific_and_rejects_penetration(self):
        from moveit_msgs.msg import PlanningScene
        task = object.__new__(VisionTask)
        task.config, task.table_object_id = config(), 'fr3_vision_table'
        task.scene = MagicMock(return_value=PlanningScene())
        task.apply_scene = MagicMock()
        obj = core.matrix(task.config['mock']['object_base'])
        task.support_contact(True, obj)
        acm = task.apply_scene.call_args.args[0].allowed_collision_matrix
        self.assertEqual(set(acm.entry_names), {'fr3_vision_brick', 'fr3_vision_table'})
        self.assertTrue(acm.entry_values[0].enabled[1])
        obj[2, 3] -= .01
        with self.assertRaisesRegex(RuntimeError, 'penetrates'):
            task.support_contact(True, obj)

    def test_grasp_failure_never_attaches_or_lifts(self):
        c = config()
        task = object.__new__(VisionTask)
        task.config, task.execute, task.hardware = c, True, False
        task.node = MagicMock()
        task.motion = MagicMock()
        task.update_object = MagicMock()
        task.observe = MagicMock(return_value=dict(c['mock']['object_base'], colour='red'))
        current = core.matrix(c['mock']['object_base']) @ core.matrix(c['object_tcp_grasp'])
        current[2, 3] += .05
        task.measured_pose = MagicMock(return_value=ros_pose(current))
        task.vertical_motion = MagicMock()
        task.attach = MagicMock()
        def hand(close=False):
            if close:
                raise RuntimeError('grasp failed')
        task.hand = hand
        with self.assertRaisesRegex(RuntimeError, 'grasp failed'):
            task.run()
        task.attach.assert_not_called()
        self.assertEqual(task.vertical_motion.call_count, 1)  # descent only


if __name__ == '__main__':
    unittest.main()
