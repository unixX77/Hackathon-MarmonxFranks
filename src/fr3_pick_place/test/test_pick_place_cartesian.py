"""Offline tests of Humble request construction and Cartesian rejection gates."""
import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE / 'pick_place_cartesian.py').exists() else HERE.parent / 'scripts'))
import pick_place_cartesian as c
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import PlanningScene, RobotTrajectory, AttachedCollisionObject, ContactInformation
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionFK, GetPositionIK, GetStateValidity
from trajectory_msgs.msg import JointTrajectoryPoint
from rcl_interfaces.msg import ParameterValue, ParameterType


def pose(x=.45, z=.2):
    p = Pose()
    p.position.x, p.position.z = x, z
    p.orientation.w = 1.
    return p


def trajectory():
    t = RobotTrajectory()
    t.joint_trajectory.joint_names = c.ARM
    for i, q in enumerate((0., .05)):
        p = JointTrajectoryPoint(positions=[q] + [0.] * 6, velocities=[0.] * 7,
                                 accelerations=[0.] * 7)
        p.time_from_start.sec = i * 2
        t.joint_trajectory.points.append(p)
    return t


def make_planner():
    p = object.__new__(c.CartesianPlanner)
    p.limits = c.Limits()
    p.bounds = dict.fromkeys(c.ARM, (-1., 1.))
    p.controller_parameters = None
    p.cartesian, p.fk_client, p.validity, p.apply = object(), object(), object(), object()
    p.task = SimpleNamespace(
        c=dict(table_size=[.6, .6, .05], table_center=[.4, 0., 0.],
               pick=[.45, 0., .15], place=[.45, .2, .15], block_size=[.03, .015, .02],
               velocity_scale=.03, acceleration_scale=.02),
        joints=dict.fromkeys(c.ARM, 0.), scene_client=object(), node=MagicMock())
    original = PlanningScene()
    original.world.collision_objects = [c.box('unrelated', [.1]*3, [1.]*3)]
    camera = AttachedCollisionObject(link_name=c.TCP)
    camera.object = c.box('handeye_camera', [.1]*3, [0.]*3, c.TCP)
    original.robot_state.attached_collision_objects.append(camera)
    p.original = original
    p.solution = trajectory()
    p.fraction = 1.
    p.collision = False
    p.requests = []
    def call(client, req, timeout=30):
        p.requests.append((client, copy.deepcopy(req)))
        if client is p.task.scene_client:
            return SimpleNamespace(scene=copy.deepcopy(original))
        if client is p.apply:
            return ApplyPlanningScene.Response(success=True)
        if client is p.fk_client:
            q = dict(zip(req.robot_state.joint_state.name, req.robot_state.joint_state.position))
            r = GetPositionFK.Response()
            r.error_code.val = c.MoveItErrorCodes.SUCCESS
            r.fk_link_names = [c.TCP]
            ps = PoseStamped(pose=pose(z=.2-q['fr3_joint1']))
            ps.header.frame_id = c.BASE
            r.pose_stamped = [ps]
            return r
        if client is p.validity:
            return GetStateValidity.Response(valid=not p.collision)
        if client is p.cartesian:
            r = GetCartesianPath.Response(fraction=p.fraction, solution=copy.deepcopy(p.solution))
            r.error_code.val = c.MoveItErrorCodes.SUCCESS
            return r
        raise AssertionError('Unexpected service')
    p.call = MagicMock(side_effect=call)
    return p


class CartesianChecks(unittest.TestCase):
    def test_support_touch_accepts_only_numerical_zero_depth_table_block_contact(self):
        contact = ContactInformation(contact_body_1=c.TABLE, contact_body_2=c.BLOCK, depth=0.)
        response = GetStateValidity.Response(valid=False, contacts=[contact])
        self.assertTrue(c.is_support_touch(response, True))
        self.assertFalse(c.is_support_touch(response, False))
        response.contacts[0].depth = 1e-8
        self.assertFalse(c.is_support_touch(response, True))
        response.contacts[0].depth = float('nan')
        self.assertFalse(c.is_support_touch(response, True))
        response.contacts[0].depth = 0.
        response.contacts[0].contact_body_1 = 'fr3_hand'
        self.assertFalse(c.is_support_touch(response, True))
        response.contacts = []
        self.assertFalse(c.is_support_touch(response, True))

    def test_vertical_waypoints_scaling_timing_and_scene_restore(self):
        p = make_planner()
        result, start = p.plan([.45, 0., .15])
        req = next(req for client, req in p.requests if client is p.cartesian)
        self.assertEqual((req.header.frame_id, req.link_name), (c.BASE, c.TCP))
        self.assertTrue(req.avoid_collisions)
        self.assertEqual(req.max_velocity_scaling_factor, .03)
        self.assertEqual(req.max_acceleration_scaling_factor, .02)
        self.assertGreater(len(req.waypoints), 20)
        for w in req.waypoints:
            self.assertEqual((w.position.x, w.position.y, w.orientation.w), (.45, 0., 1.))
        self.assertAlmostEqual(req.waypoints[-1].position.z, .15)
        self.assertEqual(result, p.solution)  # validated timing/geometry are not replaced afterwards
        self.assertEqual(list(start.joint_state.position), [0.]*7)
        applied = [req.scene for client, req in p.requests if client is p.apply]
        self.assertEqual(len(applied), 2)
        self.assertEqual({o.id for o in applied[-1].world.collision_objects}, {c.TABLE, c.BLOCK})
        self.assertTrue(all(o.operation == c.CollisionObject.REMOVE
                            for o in applied[-1].world.collision_objects))

    def test_partial_and_nan_fraction_rejected_and_scene_restored(self):
        for fraction in (.98, float('nan')):
            p = make_planner()
            p.fraction = fraction
            with self.assertRaisesRegex(RuntimeError, 'Incomplete Cartesian'):
                p.plan([.45, 0., .15])
            self.assertEqual(sum(client is p.apply for client, req in p.requests), 2)

    def test_partial_path_reports_collision_probe_but_never_returns_it(self):
        p = make_planner()
        p.fraction = .55
        p.solution.joint_trajectory.points[-1].positions[0] = .025
        original_call = p.call.side_effect
        probes = []
        def call(client, req, timeout=30):
            if isinstance(req, GetPositionIK.Request):
                probes.append(copy.deepcopy(req))
                r = GetPositionIK.Response(solution=copy.deepcopy(req.ik_request.robot_state))
                r.error_code.val = c.MoveItErrorCodes.SUCCESS
                return r
            if isinstance(req, GetStateValidity.Request):
                return GetStateValidity.Response(valid=False, contacts=[ContactInformation(
                    contact_body_1='fr3_hand', contact_body_2=c.BLOCK)])
            return original_call(client, req, timeout)
        p.call.side_effect = call
        with self.assertRaisesRegex(RuntimeError, "collision contacts=.*fr3_hand.*fr3_demo_block"):
            p.plan([.45, 0., .15])
        self.assertEqual(len(probes), 1)
        self.assertFalse(probes[0].ik_request.avoid_collisions)
        self.assertEqual(probes[0].ik_request.ik_link_name, c.TCP)
        self.assertEqual(sum(client is p.cartesian for client, req in p.requests), 1)
        self.assertEqual(sum(client is p.apply for client, req in p.requests), 2)

    def test_collision_rejected_and_scene_restored(self):
        p = make_planner()
        p.collision = True
        with self.assertRaisesRegex(RuntimeError, 'Collision'):
            p.plan([.45, 0., .15])
        self.assertEqual(sum(client is p.apply for client, req in p.requests), 2)

    def test_carried_block_in_state_and_camera_preserved(self):
        p = make_planner()
        _, start = p.plan([.45, 0., .15], carrying=True)
        attached = {o.object.id for o in start.attached_collision_objects}
        self.assertEqual(attached, {'handeye_camera', c.BLOCK})
        diff = next(req.scene for client, req in p.requests if client is p.apply)
        self.assertFalse(any(o.id == c.BLOCK for o in diff.world.collision_objects))

    def test_descent_then_lift_restores_scene_without_removing_absent_objects(self):
        for existing_block in (False, True):
            p = make_planner()
            world = {o.id: copy.deepcopy(o) for o in p.original.world.collision_objects}
            if existing_block:
                world[c.BLOCK] = c.box(c.BLOCK, [.03, .015, .0095], [.4, 0., .1])
            before = copy.deepcopy(world)
            normal_call = p.call.side_effect
            def strict_call(client, req, timeout=30):
                if client is p.task.scene_client:
                    scene = copy.deepcopy(p.original)
                    scene.world.collision_objects = list(copy.deepcopy(world).values())
                    return SimpleNamespace(scene=scene)
                if client is p.apply:
                    p.requests.append((client, copy.deepcopy(req)))
                    for obj in req.scene.world.collision_objects:
                        if obj.operation == c.CollisionObject.REMOVE:
                            if obj.id not in world:
                                return ApplyPlanningScene.Response(success=False)
                            del world[obj.id]
                        else:
                            world[obj.id] = copy.deepcopy(obj)
                    return ApplyPlanningScene.Response(success=True)
                return normal_call(client, req, timeout)
            p.call.side_effect = strict_call
            p.plan([.45, 0., .15])
            self.assertEqual(world, before)
            _, start = p.plan([.45, 0., .15], carrying=True)
            self.assertEqual(world, before)
            self.assertIn(c.BLOCK, [o.object.id for o in start.attached_collision_objects])
            self.assertIn('handeye_camera', [o.object.id for o in start.attached_collision_objects])

    def test_cleanup_failure_keeps_original_install_error_visible(self):
        p = make_planner()
        normal_call = p.call.side_effect
        def fail_apply(client, req, timeout=30):
            if client is p.apply:
                return ApplyPlanningScene.Response(success=False)
            return normal_call(client, req, timeout)
        p.call.side_effect = fail_apply
        with self.assertRaisesRegex(RuntimeError, 'Original failure: Could not install Cartesian collision scene'):
            p.plan([.45, 0., .15], carrying=True)

    def test_unaligned_start_rejected_before_cartesian_request(self):
        p = make_planner()
        with self.assertRaisesRegex(RuntimeError, 'not aligned'):
            p.plan([.46, 0., .15])
        self.assertFalse(any(client is p.cartesian for client, req in p.requests))

    def test_joint_bounds_checked_between_trajectory_states(self):
        p = make_planner()
        p.bounds['fr3_joint1'] = (-.1, .02)
        with self.assertRaisesRegex(RuntimeError, 'joint limits'):
            p.plan([.45, 0., .15])

    def test_joint_jump_and_missing_timing_rejected(self):
        t = trajectory()
        t.joint_trajectory.points[-1].positions[0] = .5
        with self.assertRaisesRegex(RuntimeError, 'discontinuity'):
            list(c.timed_samples(t, c.Limits()))
        t = trajectory()
        t.joint_trajectory.points[-1].time_from_start.sec = 0
        with self.assertRaisesRegex(RuntimeError, 'timing'):
            list(c.timed_samples(t, c.Limits()))

    def test_spline_lateral_overshoot_rejected_even_with_aligned_waypoints(self):
        t = trajectory()
        # Add a middle point on the desired line but with lateral joint velocity.
        # The resulting cubic spline leaves the line between endpoints.
        mid = JointTrajectoryPoint(positions=[.025]+[0.]*6, velocities=[0., .02]+[0.]*5)
        mid.time_from_start.sec = 1
        t.joint_trajectory.points.insert(1, mid)
        for p in t.joint_trajectory.points:
            p.accelerations = []
        def fk(q):
            return pose(x=.45+q['fr3_joint2'], z=.2-q['fr3_joint1'])
        with self.assertRaisesRegex(RuntimeError, 'vertical corridor'):
            c.validate_vertical(t, pose(), .15, fk, c.Limits())

    def test_orientation_drift_and_incomplete_height_rejected(self):
        def fk(q):
            p = pose(z=.2-q['fr3_joint1'])
            angle = q['fr3_joint1']  # final rotation about 2.9 degrees
            p.orientation.z, p.orientation.w = math.sin(angle/2), math.cos(angle/2)
            return p
        with self.assertRaisesRegex(RuntimeError, 'vertical corridor'):
            c.validate_vertical(trajectory(), pose(), .15, fk, c.Limits())
        with self.assertRaisesRegex(RuntimeError, 'requested height'):
            c.validate_vertical(trajectory(), pose(), .14,
                                lambda q: pose(z=.2-q['fr3_joint1']), c.Limits())

    def test_joint_limits_use_model_soft_bounds(self):
        xml = '<robot>' + ''.join(
            f'<joint name="{n}"><limit lower="-2" upper="2"/>'
            '<safety_controller soft_lower_limit="-1" soft_upper_limit="1"/></joint>'
            for n in c.ARM) + '</robot>'
        self.assertEqual(c.joint_bounds(xml), dict.fromkeys(c.ARM, (-1., 1.)))

    def test_live_cartesian_start_drift_prevents_execution(self):
        p = make_planner()
        _, start = p.plan([.45, 0., .15])
        p.check_execution_start(dict.fromkeys(c.ARM, 0.), start)
        moved = dict.fromkeys(c.ARM, 0.)
        moved['fr3_joint1'] = .002
        with self.assertRaisesRegex(RuntimeError, 'Measured TCP'):
            p.check_execution_start(moved, start)

    def test_unvalidated_controller_interpolation_is_rejected(self):
        p = make_planner()
        p.controller_parameters = object()
        p.call = MagicMock(return_value=SimpleNamespace(values=[ParameterValue(
            type=ParameterType.PARAMETER_STRING, string_value='none')]))
        with self.assertRaisesRegex(RuntimeError, 'interpolation mode'):
            p.check_interpolation()

    def test_restore_failure_prevents_returning_a_plan(self):
        p = make_planner()
        call = p.call.side_effect
        applied = []
        def fail_restore(client, req, timeout=30):
            if client is p.apply:
                applied.append(req)
                if len(applied) == 2:
                    return ApplyPlanningScene.Response(success=False)
            return call(client, req, timeout)
        p.call.side_effect = fail_restore
        with self.assertRaisesRegex(RuntimeError, 'restore the planning scene'):
            p.plan([.45, 0., .15])


if __name__ == '__main__':
    unittest.main()
