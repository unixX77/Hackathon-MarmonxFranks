"""Offline policy, kinematics and stop/recovery checks. No robot connections."""
import copy
from contextlib import nullcontext
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from pick_core import ARM, DescentStopped, Intervention, Kinematics, PickFailure, PickWorkflow, Policy, seeds
import pick
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AllowedCollisionMatrix, MoveItErrorCodes, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from franka_msgs.msg import FrankaRobotState


GOOD = dict(accepted=True, xyz_m=0., angle_deg=0., joint_rad=0.)


class Adapter:
    def __init__(self):
        self.events = []
        self.errors = [GOOD]
        self.fail_descents = 0
        self.fail_planning = 0
        self.intervene = False
        self.closed = False

    def status(self, state, message):
        self.events.append(state)

    def pregrasp(self):
        return [.3, 0., .1]

    def grasp_target(self):
        return [.3, 0., 0.]

    def find_approach(self, target):
        self.events.append('find')
        return 'approach'

    def reject_approach(self):
        self.events.append('reject')

    def open_hand(self):
        self.events.append('open')

    def close_hand(self):
        self.closed = True
        self.events.append('close')

    def move_arm(self, plan, label, vertical=False):
        self.events.append(label)
        if vertical and self.intervene:
            raise Intervention('protective stop')
        if vertical and self.fail_descents:
            self.fail_descents -= 1
            raise DescentStopped('tracking error; cancellation confirmed')

    def verify(self, target):
        return self.errors.pop(0) if len(self.errors) > 1 else self.errors[0]

    def plan_descent(self):
        self.events.append('plan_descent')
        if self.fail_planning:
            self.fail_planning -= 1
            raise PickFailure('incomplete Cartesian path')
        return 'descent'

    def recover(self, target):
        self.events.append('retreat')


class WorkflowTests(unittest.TestCase):
    def test_success_orders_lookahead_alignment_descent_and_grasp(self):
        a = Adapter()
        self.assertEqual(PickWorkflow(a, Policy()).run(), 'HOLDING_OBJECT')
        self.assertLess(a.events.index('find'), a.events.index('open'))
        self.assertLess(a.events.index('Approach'), a.events.index('plan_descent'))
        self.assertLess(a.events.index('Descent'), a.events.index('close'))

    def test_no_improvement_exhausts_correction_without_descending(self):
        a = Adapter()
        a.errors = [dict(accepted=False, xyz_m=.001, angle_deg=.6, joint_rad=.001)]
        with self.assertRaisesRegex(PickFailure, 'no improvement'):
            PickWorkflow(a, Policy()).run()
        self.assertEqual(a.events.count('Approach correction'), 1)
        self.assertNotIn('Descent', a.events)
        self.assertFalse(a.closed)

    def test_improving_correction_can_succeed(self):
        a = Adapter()
        a.errors = [dict(accepted=False, xyz_m=.001, angle_deg=.6, joint_rad=.001), GOOD]
        PickWorkflow(a, Policy()).run()
        self.assertTrue(a.closed)
        self.assertEqual(a.events.count('Approach correction'), 1)

    def test_excessive_error_never_corrects(self):
        a = Adapter()
        a.errors = [dict(accepted=False, xyz_m=.02, angle_deg=.6, joint_rad=.001)]
        with self.assertRaises(PickFailure):
            PickWorkflow(a, Policy()).run()
        self.assertNotIn('Approach correction', a.events)

    def test_failed_descent_retreats_and_retries_once(self):
        a = Adapter()
        a.fail_descents = 1
        PickWorkflow(a, Policy()).run()
        self.assertEqual(a.events.count('Descent'), 2)
        self.assertEqual(a.events.count('retreat'), 1)
        self.assertEqual(a.events.count('reject'), 1)
        self.assertTrue(a.closed)

    def test_repeated_descent_failure_has_finite_budget_and_no_grasp(self):
        a = Adapter()
        a.fail_descents = 10
        with self.assertRaisesRegex(PickFailure, 'retry budget exhausted'):
            PickWorkflow(a, Policy()).run()
        self.assertEqual(a.events.count('Descent'), 2)
        self.assertEqual(a.events.count('retreat'), 2)
        self.assertFalse(a.closed)

    def test_hardware_fault_never_retreats_or_retries(self):
        a = Adapter()
        a.intervene = True
        with self.assertRaises(Intervention):
            PickWorkflow(a, Policy()).run()
        self.assertNotIn('retreat', a.events)
        self.assertFalse(a.closed)

    def test_grasp_pose_failure_retreats_without_closing(self):
        a = Adapter()
        a.errors = [GOOD, dict(accepted=False, xyz_m=.005, angle_deg=0., joint_rad=0.)]
        with self.assertRaises(PickFailure):
            PickWorkflow(a, Policy(descent_retries=0)).run()
        self.assertIn('retreat', a.events)
        self.assertFalse(a.closed)

    def test_descent_planning_failure_reselects_before_any_descent(self):
        a = Adapter()
        a.fail_planning = 1
        PickWorkflow(a, Policy()).run()
        self.assertIn('Alternative approach', a.events)
        self.assertNotIn('retreat', a.events)

    def test_planning_fault_does_not_reselect(self):
        a = Adapter()
        a.plan_descent = MagicMock(side_effect=Intervention('scene unavailable'))
        with self.assertRaises(Intervention):
            PickWorkflow(a, Policy()).run()
        self.assertEqual(a.events.count('find'), 1)


def synthetic_urdf():
    joints = []
    for i, name in enumerate(ARM):
        parent = 'fr3_link0' if i == 0 else f'link{i}'
        axis = ('0 0 1', '0 1 0', '1 0 0')[i % 3]
        joints.append(f'<joint name="{name}" type="revolute"><parent link="{parent}"/>'
                      f'<child link="link{i+1}"/><origin xyz=".1 .02 .04" rpy=".1 .2 .3"/>'
                      f'<axis xyz="{axis}"/></joint>')
    joints.append('<joint name="tcp" type="fixed"><parent link="link7"/>'
                  '<child link="fr3_hand_tcp"/><origin xyz="0 0 .1034" rpy="0 0 -.7854"/></joint>')
    return '<robot name="test">' + ''.join(joints) + '</robot>'


class KinematicsTests(unittest.TestCase):
    def test_jacobian_matches_finite_differences_including_tool_offset(self):
        model = Kinematics(synthetic_urdf())
        joints = dict(zip(ARM, [.1, -.4, .2, -1., .3, .7, .4]))
        t, jac, sigma = model.evaluate(joints, length=.3)
        for i, name in enumerate(ARM):
            shifted = dict(joints)
            shifted[name] += 1e-7
            other, _, _ = model.evaluate(shifted, length=.3)
            np.testing.assert_allclose((other[:3, 3]-t[:3, 3])/1e-7/.3, jac[:3, i], atol=1e-6)
            rdot = (other[:3, :3]-t[:3, :3])/1e-7 @ t[:3, :3].T
            np.testing.assert_allclose([rdot[2, 1], rdot[0, 2], rdot[1, 0]], jac[3:, i], atol=1e-6)
        self.assertGreater(sigma, 0.)

    def test_varied_seeds_are_unique_and_within_limits(self):
        current = dict.fromkeys(ARM, 0.)
        bounds = dict.fromkeys(ARM, (-1., 1.))
        result = list(seeds(current, bounds, 8))
        self.assertEqual(result[0], current)
        self.assertEqual(len({tuple(q[n] for n in ARM) for q in result}), 8)
        self.assertTrue(all(-1. < q[n] < 1. for q in result for n in ARM))

    def test_policy_rejects_invalid_or_misspelled_options(self):
        for config in ({'ik_attempts': 0}, {'corrections': 100}, {'min_sigma': float('nan')},
                       {'monitor_period_s': -1}, {'corections': 2}, {'ik_attempts': True}):
            with self.assertRaises(ValueError):
                Policy.from_config(config)


class ExecutionTests(unittest.TestCase):
    def test_initialized_picker_preserves_gripper_clients_and_runs_preview_workflow(self):
        # Exercise the real Task/StepwiseTask constructors: they assign move and
        # grasp action clients, which must never shadow workflow methods.
        node = MagicMock()
        config = pick.load_config(Path(__file__).resolve().parent.parent / 'config/task.yaml', camera_mode=True)
        q = dict.fromkeys(ARM, 0.)
        scene = SimpleNamespace(scene=SimpleNamespace(allowed_collision_matrix=AllowedCollisionMatrix()))
        vertical = MagicMock()
        vertical.call.return_value = SimpleNamespace(values=[SimpleNamespace(string_value=synthetic_urdf())])
        with patch('pick_place.ActionClient'), patch.object(pick.Task, 'wait', return_value=scene), \
                patch('pick_place_stepwise.CartesianPlanner', return_value=vertical), \
                patch.object(pick.StepwiseTask, 'settled_snapshot', return_value=q):
            a = pick.Picker(node, config, False, None, Policy(), '/robot_state')
        self.assertIsNone(a.move)
        self.assertIsNone(a.grasp)
        self.assertTrue(callable(a.move_arm))
        self.assertEqual(a.grasp_target(), config['pick'])
        self.assertAlmostEqual(a.pregrasp()[2], config['pick'][2]+config['clearance'])
        t = RobotTrajectory()
        t.joint_trajectory.joint_names = ARM
        t.joint_trajectory.points = [JointTrajectoryPoint(positions=[.1]*7)]
        a.find_approach = MagicMock(return_value=pick.Plan(t, RobotState(), a.pregrasp()))
        a.plan_descent = MagicMock(return_value=pick.Plan(t, RobotState(), a.grasp_target(), True))
        a.verify = MagicMock(return_value=GOOD)
        with patch.object(pick, 'confirm'), patch.object(a, 'action') as actions:
            self.assertEqual(PickWorkflow(a, Policy()).run(), 'PREVIEW_COMPLETE')
        actions.assert_not_called()
        self.assertFalse(a.holding)  # preview never claims a physical grasp

    def picker(self):
        a = object.__new__(pick.Picker)
        a.hardware = True
        a.node = MagicMock()
        a.robot = FrankaRobotState()
        a.robot.robot_mode = FrankaRobotState.ROBOT_MODE_IDLE
        a.robot.header.stamp.sec = 10
        a.node.get_clock().now().nanoseconds = int(10e9)
        a.robot_received = pick.time.monotonic()
        a.policy = Policy()
        a.log_event = MagicMock()
        a.status = MagicMock()
        a.joints = dict.fromkeys(ARM, 0.)
        a.settled_snapshot = MagicMock(return_value=a.joints)
        return a

    def test_robot_fault_contact_and_stale_feedback_inhibit_motion(self):
        a = self.picker()
        a.robot_ok()
        a.robot.robot_mode = FrankaRobotState.ROBOT_MODE_REFLEX
        with self.assertRaises(Intervention):
            a.robot_ok()
        a.robot.robot_mode = FrankaRobotState.ROBOT_MODE_IDLE
        a.robot.collision_indicators.is_joint_contact = [1.] * 7
        with self.assertRaises(Intervention):
            a.robot_ok()
        a.robot.collision_indicators.is_joint_contact = [0.] * 7
        a.robot_received -= 1.
        with self.assertRaises(Intervention):
            a.robot_ok()

    def test_cancellation_requires_terminal_success_or_cancel_then_settled_healthy_state(self):
        a = self.picker()
        a.wait = MagicMock(side_effect=[SimpleNamespace(), SimpleNamespace(status=pick.GoalStatus.STATUS_CANCELED)])
        a.cancel_and_settle(MagicMock(), MagicMock())
        a.settled_snapshot.assert_called_once()
        a.wait.side_effect = [SimpleNamespace(), SimpleNamespace(status=pick.GoalStatus.STATUS_ABORTED)]
        with self.assertRaises(Intervention):
            a.cancel_and_settle(MagicMock(), MagicMock())

    def test_grasp_rechecks_pose_after_operator_prompt(self):
        a = self.picker()
        a.c = dict(pick=[.3, 0., 0.])
        a.verify = MagicMock(return_value=dict(accepted=False))
        with patch.object(pick, 'confirm'), patch.object(pick.Task, 'hand') as hand:
            with self.assertRaises(Intervention):
                a.close_hand()
        hand.assert_not_called()

    def test_recovery_rejects_sideways_unknown_state_without_motion(self):
        a = self.picker()
        a.holding = False
        p = Pose()
        p.orientation.x = 1.
        p.orientation.w = 0.
        p.position.x = .4
        a.state_pose = MagicMock(return_value=(RobotState(), p))
        a.move_arm = MagicMock()
        with self.assertRaises(Intervention):
            a.recover([.3, 0., .1])
        a.move_arm.assert_not_called()

    def test_uncertain_grasp_blocks_reopening_and_new_requests(self):
        a = self.picker()
        a.c = dict(pick=[.3, 0., 0.])
        a.verify = MagicMock(return_value=dict(accepted=True))
        with patch.object(pick, 'confirm'), patch.object(pick.Task, 'hand', side_effect=RuntimeError('lost action result')):
            with self.assertRaises(Intervention):
                a.close_hand()
        self.assertTrue(a.holding)
        self.assertTrue(a.blocked)

    def test_monitor_uses_native_joint_tracking_feedback(self):
        a = self.picker()
        a.robot.measured_joint_state.name = ARM
        a.robot.measured_joint_state.position = [0.] * 7
        a.robot.desired_joint_state.name = ARM
        a.robot.desired_joint_state.position = [.05] * 7
        a.latest_stamp = 10.
        a.latest = dict.fromkeys(ARM, 0.)
        a.model = MagicMock()
        a.model.evaluate.return_value = (np.eye(4), None, .5)
        with self.assertRaisesRegex(PickFailure, 'tracking error'):
            a._monitor(SimpleNamespace(vertical=False))

    def test_preview_moves_virtual_joints_without_sending_execution(self):
        a = self.picker()
        a.hardware = False
        a.preview = MagicMock()
        a.executor = MagicMock()
        t = RobotTrajectory()
        t.joint_trajectory.joint_names = ARM
        t.joint_trajectory.points = [JointTrajectoryPoint(positions=[.1]*7)]
        plan = pick.Plan(t, RobotState(), [.3, 0., .1])
        with patch.object(pick, 'confirm'):
            a.move_arm(plan, 'Preview')
        a.executor.send_goal_async.assert_not_called()
        self.assertEqual(a.joints, dict.fromkeys(ARM, .1))

    def test_candidate_lookahead_rejects_bad_descent_then_restores_live_start(self):
        a = self.picker()
        a.hardware = False
        a.c = dict(pick=[.3, 0., 0.], open_width=.04)
        a.commands = []
        a.failed_approaches = []
        a.ik_candidates = MagicMock(return_value=[dict.fromkeys(ARM, .1), dict.fromkeys(ARM, .2)])
        a.audit = MagicMock(return_value=(1., {}))
        a.vertical = MagicMock()
        t = RobotTrajectory()
        t.joint_trajectory.joint_names = ARM
        t.joint_trajectory.points = [JointTrajectoryPoint(positions=[.2]*7)]
        a.vertical.plan.side_effect = [RuntimeError('incomplete path'), (t, RobotState())]
        def pretend_plan(obj, *args, **kwargs):
            obj.commands = [('arm', 'approach', copy.deepcopy(t))]
            obj.preview_start = RobotState()
            obj.joints.update(obj.forced_solution)
        with patch.object(pick.Task, 'motion', pretend_plan):
            selected = a.find_approach([.3, 0., .1])
        self.assertEqual(a.vertical.plan.call_count, 2)
        self.assertIsNone(a.forced_solution)
        self.assertEqual([a.joints[n] for n in ARM], [0.]*7)
        self.assertEqual(selected.xyz, [.3, 0., .1])

    def test_scene_failure_in_lookahead_is_not_retried_as_an_ik_candidate(self):
        a = self.picker()
        a.hardware = False
        a.c = dict(pick=[.3, 0., 0.], open_width=.04)
        a.commands = []
        a.ik_candidates = MagicMock(return_value=[dict.fromkeys(ARM, .1), dict.fromkeys(ARM, .2)])
        with patch.object(pick.Task, 'motion', side_effect=pick.PlanningInfrastructureError('restore failed')) as motion:
            with self.assertRaises(Intervention):
                a.find_approach([.3, 0., .1])
        self.assertEqual(motion.call_count, 1)

    def test_distinct_ik_uses_different_valid_seeds_and_discards_duplicate_solutions(self):
        a = self.picker()
        a.c = dict(pick=[.3, 0., 0.])
        a.failed_approaches = []
        a.policy = Policy(ik_attempts=3)
        a.vertical = MagicMock()
        a.vertical.bounds = dict.fromkeys(ARM, (-3., 3.))
        a.vertical.scoped_scene.side_effect = lambda *args: nullcontext(RobotState())
        a.vertical.state.return_value = RobotState()
        a.ik = object()
        requests = []
        def answer(client, req, **kwargs):
            requests.append(copy.deepcopy(req))
            state = RobotState()
            state.joint_state.name = ARM
            state.joint_state.position = [.1 if len(requests) < 3 else .3]*7
            return SimpleNamespace(solution=state, error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS))
        a.vertical.call.side_effect = answer
        candidates = a.ik_candidates([.3, 0., .1], a.joints)
        self.assertEqual(len(candidates), 2)
        used_seeds = [c.args[2] for c in a.vertical.state.call_args_list]
        self.assertNotEqual(used_seeds[0], used_seeds[1])
        for req in requests:
            q = req.ik_request.pose_stamped.pose.orientation
            self.assertEqual([q.x, q.y, q.z, q.w], [1., 0., 0., 0.])
            self.assertTrue(req.ik_request.avoid_collisions)

    def test_descent_monitor_stop_requires_confirmed_cancel_before_recovery(self):
        for cancellation_fails in (False, True):
            a = self.picker()
            a.preview = MagicMock()
            a.executor = MagicMock()
            a.audit = MagicMock()
            a.vertical = MagicMock()
            a.stop_requested = MagicMock()
            a.stop_requested.is_set.return_value = False
            a.pump = MagicMock()
            a._monitor = MagicMock(side_effect=PickFailure('corridor exceeded'))
            a.cancel_and_settle = MagicMock()
            if cancellation_fails:
                a.cancel_and_settle.side_effect = Intervention('stop not confirmed')
            result_future = MagicMock()
            result_future.done.return_value = False
            handle = MagicMock()
            handle.accepted = True
            handle.get_result_async.return_value = result_future
            a.wait = MagicMock(return_value=handle)
            t = RobotTrajectory()
            t.joint_trajectory.joint_names = ARM
            t.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.]*7)]
            plan = pick.Plan(t, RobotState(), [.3, 0., 0.], True)
            with patch.object(pick, 'confirm'):
                with self.assertRaises(Intervention if cancellation_fails else DescentStopped):
                    a.move_arm(plan, 'Descent', vertical=True)
            a.cancel_and_settle.assert_called_once_with(handle, result_future)

    def test_retreat_endpoint_failure_requires_intervention(self):
        a = self.picker()
        a.holding = False
        a.c = dict(endpoint_z_tolerance_m=.002)
        p = Pose()
        p.orientation.x, p.orientation.w = 1., 0.
        p.position.x, p.position.z = .3, .05
        a.state_pose = MagicMock(return_value=(RobotState(), p))
        a.vertical = MagicMock()
        a.vertical.plan.return_value = (RobotTrajectory(), RobotState())
        a.audit = MagicMock()
        a.move_arm = MagicMock()
        a.verify = MagicMock(return_value=dict(accepted=False))
        with self.assertRaisesRegex(Intervention, 'endpoint tolerances'):
            a.recover([.3, 0., .1])
        a.move_arm.assert_called_once()


if __name__ == '__main__':
    unittest.main()
