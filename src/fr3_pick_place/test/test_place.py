"""Offline placement geometry, shared-picker integration and failure policy."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import place
import pick
import pick_place_final as final
from pick_core import DescentStopped, Intervention, PickFailure, Policy
from pick_place import block_object
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class FakePlace:
    def __init__(self):
        self.hardware = True
        self.events = []
        self.descents_fail = 0
        self.release_fails = False
        self.retreat_fails = False
        self.pose_valid = True

    def status(self, state, message):
        self.events.append(state)

    def require_held(self):
        self.events.append('require_held')

    def lift_held(self):
        self.events.append('lift')

    def place_above(self):
        return [.4, .1, .1]

    def place_release(self):
        return [.4, .1, 0.]

    def find_place_approach(self, target):
        self.events.append('find_transfer')
        return 'transfer'

    def align_place(self, target):
        self.events.append('align')

    def move_arm(self, plan, label, vertical=False):
        self.events.append(label)
        if label == 'Placement descent' and self.descents_fail:
            self.descents_fail -= 1
            raise DescentStopped('confirmed stop')

    def plan_place_descent(self):
        self.events.append('plan_descent')
        return 'descent'

    def verify(self, target):
        return dict(accepted=self.pose_valid)

    def recover_place(self, target):
        self.events.append('recover_held')

    def reject_place_approach(self):
        self.events.append('reject')

    def release_held(self):
        self.events.append('release')
        if self.release_fails:
            raise Intervention('uncertain release')

    def retreat_released(self, height):
        self.events.append('retreat_empty')
        if self.retreat_fails:
            raise Intervention('retreat blocked after release')


class WorkflowTests(unittest.TestCase):
    def test_success_lifts_transfers_verifies_releases_then_retreats(self):
        a = FakePlace()
        self.assertEqual(place.PlacementWorkflow(a, Policy()).run(), 'PLACED')
        for before, after in [('lift', 'Transfer above placement'), ('align', 'plan_descent'),
                              ('Placement descent', 'release'), ('release', 'retreat_empty')]:
            self.assertLess(a.events.index(before), a.events.index(after))

    def test_descent_failure_recovers_with_grasp_and_retries(self):
        a = FakePlace()
        a.descents_fail = 1
        place.PlacementWorkflow(a, Policy()).run()
        self.assertEqual(a.events.count('recover_held'), 1)
        self.assertEqual(a.events.count('release'), 1)
        self.assertLess(a.events.index('recover_held'), a.events.index('release'))

    def test_exhausted_descent_budget_never_releases(self):
        a = FakePlace()
        a.descents_fail = 10
        with self.assertRaises(PickFailure):
            place.PlacementWorkflow(a, Policy()).run()
        self.assertEqual(a.events.count('Placement descent'), 2)
        self.assertEqual(a.events.count('recover_held'), 2)
        self.assertNotIn('release', a.events)

    def test_release_failure_never_attempts_empty_hand_retreat(self):
        a = FakePlace()
        a.release_fails = True
        with self.assertRaises(Intervention):
            place.PlacementWorkflow(a, Policy()).run()
        self.assertNotIn('retreat_empty', a.events)
        self.assertEqual(a.events.count('release'), 1)

    def test_retreat_failure_never_repeats_release_or_grasp(self):
        a = FakePlace()
        a.retreat_fails = True
        with self.assertRaises(Intervention):
            place.PlacementWorkflow(a, Policy()).run()
        self.assertEqual(a.events.count('release'), 1)
        self.assertEqual(a.events.count('Placement descent'), 1)

    def test_bad_release_pose_retreats_without_opening(self):
        a = FakePlace()
        a.pose_valid = False
        with self.assertRaises(PickFailure):
            place.PlacementWorkflow(a, Policy(descent_retries=0)).run()
        self.assertIn('recover_held', a.events)
        self.assertNotIn('release', a.events)

    def test_combined_uses_imported_pick_workflow_then_captures_held_state(self):
        a = FakePlace()
        a.capture_held = MagicMock()
        with patch.object(place, 'PickWorkflow') as picking, patch.object(place, 'PlacementWorkflow') as placing:
            placing.return_value.run.return_value = 'PLACED'
            self.assertEqual(place.PickPlaceWorkflow(a, Policy()).run(), 'PLACED')
        picking.return_value.run.assert_called_once()
        a.capture_held.assert_called_once()
        placing.return_value.run.assert_called_once()

    def test_pick_failure_does_not_attach_or_place(self):
        a = FakePlace()
        a.capture_held = MagicMock()
        with patch.object(place, 'PickWorkflow') as picking, patch.object(place, 'PlacementWorkflow') as placing:
            picking.return_value.run.side_effect = PickFailure('failed pick')
            with self.assertRaises(PickFailure):
                place.PickPlaceWorkflow(a, Policy()).run()
        a.capture_held.assert_not_called()
        placing.assert_not_called()


class GeometryAndSceneTests(unittest.TestCase):
    def task(self):
        return pick.load_config(Path(__file__).resolve().parent.parent / 'config/task.yaml', camera_mode=True)

    def target(self):
        return place.PlaceTarget([.4, .1, -.01525])

    def adapter(self):
        a = object.__new__(place.PlacePicker)
        a.c = self.task()
        a.c['attached_block_pose'] = dict(xyz_m=[.001, .002, .009], quaternion_xyzw=[1., 0., 0., 0.])
        a.hardware, a.holding, a.blocked, a.phase = False, False, False, 'held'
        a.place_target = self.target()
        a.policy = Policy()
        a.node = MagicMock()
        a.log_event = MagicMock()
        a.status = MagicMock()
        a.joints = dict.fromkeys(pick.ARM, 0.)
        a.robot_ok = MagicMock()
        return a

    def test_release_tcp_uses_captured_grip_offset_not_old_pick_position(self):
        target = self.target()
        attachment = dict(xyz_m=[.001, .002, .009], quaternion_xyzw=[1., 0., 0., 0.])
        tcp = place.release_tcp(target, attachment)
        self.assertAlmostEqual(tcp[0], .399)
        self.assertAlmostEqual(tcp[1], .102)
        self.assertAlmostEqual(tcp[2], -.00425)
        reconstructed = place.object_pose(tcp, [1., 0., 0., 0.], attachment)
        for got, expected in zip(reconstructed['xyz_m'], [.4, .1, -.01325]):
            self.assertAlmostEqual(got, expected)
        self.assertAlmostEqual(abs(reconstructed['quaternion_xyzw'][3]), 1.)

    def test_invalid_frames_heights_and_nonfinite_clearances_are_rejected(self):
        base = dict(frame_id='fr3_link0', block_center_xyz_m=[.4, .1, -.01525])
        place.PlaceTarget.from_config(base, self.task())
        for update in (dict(frame_id='camera_link'), dict(block_center_xyz_m=[.4, .1, -.1]),
                       dict(release_clearance_m=float('nan')), dict(approach_clearance_m=-1),
                       dict(block_center_xyz_m=[.4, float('inf'), 0.])):
            with self.assertRaises(ValueError):
                place.PlaceTarget.from_config(dict(base, **update), self.task())

    def test_scene_modes_and_unknown_grasp_gate(self):
        a = self.adapter()
        self.assertEqual(a.scene_mode(), (True, False))
        a.require_held()
        a.phase = 'released'
        self.assertEqual(a.scene_mode(), (False, True))
        with self.assertRaises(Intervention):
            a.require_held()

    def test_virtual_release_scene_restores_attachment_on_failure(self):
        a = self.adapter()
        a.c['placed_block_pose'] = dict(xyz_m=[.4, .1, -.01325], quaternion_xyzw=[0., 0., 0., 1.])
        a.apply_object_scene = MagicMock()
        with self.assertRaisesRegex(RuntimeError, 'retreat test'):
            with a.preview_release_scene():
                self.assertEqual(a.phase, 'released')
                raise RuntimeError('retreat test')
        self.assertEqual(a.phase, 'held')
        scenes = [c.args[0] for c in a.apply_object_scene.call_args_list]
        self.assertEqual(len(scenes), 3)
        self.assertEqual(scenes[0].robot_state.attached_collision_objects[0].object.operation, CollisionObject.REMOVE)
        self.assertEqual(scenes[1].world.collision_objects[0].operation, CollisionObject.REMOVE)
        self.assertEqual(scenes[2].robot_state.attached_collision_objects[0].object.operation, CollisionObject.ADD)

    def test_world_object_retains_actual_release_pose_and_orientation(self):
        c = self.task()
        c['placed_block_pose'] = dict(xyz_m=[.4, .1, -.01325], quaternion_xyzw=[0., 0., 1., 0.])
        p = block_object(c, placed=True).primitive_poses[0]
        self.assertEqual([p.position.x, p.position.y, p.position.z], c['placed_block_pose']['xyz_m'])
        self.assertEqual([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w], [0., 0., 1., 0.])

    def test_release_failure_keeps_uncertain_holding_state_and_never_detaches(self):
        a = self.adapter()
        a.verify = MagicMock(return_value=dict(accepted=True))
        p = Pose()
        p.orientation.x, p.orientation.w = 1., 0.
        a.state_pose = MagicMock(return_value=(RobotState(), p))
        a.apply_object_scene = MagicMock()
        with patch.object(place, 'confirm'), patch.object(place.Task, 'hand', side_effect=RuntimeError('lost result')):
            with self.assertRaises(Intervention):
                a.release_held()
        self.assertEqual(a.phase, 'release_uncertain')
        self.assertTrue(a.holding)
        self.assertTrue(a.blocked)
        a.apply_object_scene.assert_not_called()

    def test_preview_release_updates_world_state_without_claiming_hardware_grasp(self):
        a = self.adapter()
        a.verify = MagicMock(return_value=dict(accepted=True))
        p = Pose()
        p.orientation.x, p.orientation.w = 1., 0.
        a.state_pose = MagicMock(return_value=(RobotState(), p))
        a.apply_object_scene = MagicMock()
        with patch.object(place, 'confirm'):
            a.release_held()
        self.assertFalse(a.holding)
        self.assertEqual(a.phase, 'released')
        scene = a.apply_object_scene.call_args.args[0]
        self.assertEqual(scene.robot_state.attached_collision_objects[0].object.operation, CollisionObject.REMOVE)
        self.assertEqual(scene.world.collision_objects[0].operation, CollisionObject.ADD)
        self.assertAlmostEqual(a.joints['fr3_finger_joint1'], a.c['open_width']/2)

    def test_final_program_reuses_shared_cli_without_invoking_pick_main(self):
        with patch.object(final, 'run_cli', return_value=0) as run:
            self.assertEqual(final.main(), 0)
        self.assertIs(run.call_args.kwargs['picker_class'], place.PlacePicker)
        self.assertIs(run.call_args.kwargs['workflow_class'], place.PickPlaceWorkflow)
        self.assertEqual(run.call_args.kwargs['node_name'], 'pick_place_final')

    def test_lift_lookahead_failure_restores_pick_context(self):
        a = self.adapter()
        a.phase = 'empty'
        a.c.pop('attached_block_pose')
        original = dict(a.joints)
        t = RobotTrajectory()
        t.joint_trajectory.joint_names = pick.ARM
        t.joint_trajectory.points = [JointTrajectoryPoint(positions=[.2]*7)]
        descent = pick.Plan(t, RobotState(), [.3, 0., 0.], True)
        def capture(joints):
            a.c['attached_block_pose'] = dict(xyz_m=[0., 0., .009], quaternion_xyzw=[1., 0., 0., 0.])
        a.capture_grasp_transform = MagicMock(side_effect=capture)
        p = Pose()
        p.orientation.x, p.orientation.w = 1., 0.
        a.state_pose = MagicMock(return_value=(RobotState(), p))
        a.vertical = MagicMock()
        a.vertical.plan.side_effect = RuntimeError('incomplete lift')
        with self.assertRaisesRegex(RuntimeError, 'incomplete lift'):
            a.check_pick_candidate(None, descent)
        self.assertEqual(a.phase, 'empty')
        self.assertEqual(a.joints, original)
        self.assertNotIn('attached_block_pose', a.c)
        self.assertNotIn('carrying_orientation_xyzw', a.c)
        self.assertTrue(a.vertical.plan.call_args.kwargs['carrying'])

    def test_empty_attestation_removes_only_our_attachment_after_operator_removal(self):
        from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
        a = self.adapter()
        a.hardware = True
        a.scene_client = object()
        a.vertical = MagicMock()
        scene = PlanningScene()
        our = AttachedCollisionObject()
        our.object.id = 'fr3_demo_block'
        camera = AttachedCollisionObject()
        camera.object.id = 'camera'
        scene.robot_state.attached_collision_objects = [our, camera]
        a.vertical.call.return_value = SimpleNamespace(scene=scene)
        a.apply_object_scene = MagicMock()
        a.confirm_empty_context()
        a.apply_object_scene.assert_called_once()
        removed = a.apply_object_scene.call_args.args[0].robot_state.attached_collision_objects
        self.assertEqual([o.object.id for o in removed], ['fr3_demo_block'])
        self.assertEqual(a.phase, 'empty')
        self.assertNotIn('attached_block_pose', a.c)


if __name__ == '__main__':
    unittest.main()
