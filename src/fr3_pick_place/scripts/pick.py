#!/usr/bin/env python3
"""Single-brick picker: candidate lookahead, measured descent, bounded recovery.

Preview by default. --execute is real; --fake-execute is verified mock execution.
Both execution modes require approval for every arm/gripper move.
--serve keeps a Trigger-based development task interface alive between requests.
"""
import argparse
import copy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from franka_msgs.msg import FrankaRobotState
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from rcl_interfaces.srv import GetParameters
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.action import ActionClient
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from std_srvs.srv import Trigger

from pick_core import (ARM, DescentStopped, Intervention, Kinematics, PickFailure,
                       PickWorkflow, Policy, angle_deg, seeds)
from pick_place import Task
from pick_place_cartesian import (BASE, TCP, Limits, PlanningInfrastructureError,
                                  coordinates, is_support_touch, timed_samples)
from pick_place_stepwise import (MotionLog, OperatorStop, StepwiseTask, check_start,
                                 confirm, load_camera_target, load_config)


@dataclass
class Plan:
    trajectory: object
    state: object
    xyz: list
    vertical: bool = False
    carrying: bool = False
    placed: bool = False


class Picker(StepwiseTask):
    def __init__(self, node, config, execute, log, policy, robot_state_topic):
        self.fake_backend = None
        self.background = False
        self.stop_requested = threading.Event()
        self.robot = None
        self.robot_received = 0.
        self.holding = False
        self.blocked = False
        self.last_plan = None
        self.failed_approaches = []
        self.selected_approach = None
        self.policy = policy
        self.forced_solution = None
        super().__init__(node, config, execute, log)
        if policy.relative_jump_factor != 2.:
            self.vertical.limits = replace(self.vertical.limits, relative_jump_factor=policy.relative_jump_factor)
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = node.create_publisher(String, '~/status', retained)
        self.result_pub = node.create_publisher(String, '~/result', retained)
        self.robot_sub = node.create_subscription(FrankaRobotState, robot_state_topic,
                                                  self.on_robot, qos_profile_sensor_data)
        response = self.vertical.call(self.vertical.parameters, GetParameters.Request(names=['robot_description']))
        self.robot_description = response.values[0].string_value
        self.model = Kinematics(self.robot_description)
        self.ik = node.create_client(GetPositionIK, 'compute_ik')
        if not self.ik.wait_for_service(timeout_sec=10):
            raise RuntimeError('MoveIt compute_ik is unavailable')
        self.status('IDLE', 'Picker ready; start empty-handed')

    def enable_fake_execution(self):
        from fake_execution import FakeExecution
        backend = FakeExecution(self)
        self.executor = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')
        if not self.executor.wait_for_server(timeout_sec=10):
            raise Intervention('Fake MoveIt execute_trajectory action unavailable')
        self.vertical.controller_parameters = self.node.create_client(GetParameters, 'fr3_arm_controller/get_parameters')
        if not self.vertical.controller_parameters.wait_for_service(timeout_sec=5):
            raise Intervention('Fake controller parameters unavailable')
        self.vertical.check_interpolation()
        self.fake_backend, self.hardware = backend, True
        self.log_event('execution_backend', backend='fake', mock_identity_verified=True)
        self.status('FAKE_EXECUTION', 'Verified mock arm. Opaque robot/gripper will move; no physical grasp/contact simulation.')

    def command_hand(self, close=False):
        backend = getattr(self, 'fake_backend', None)
        if backend is not None:
            backend.hand(self.c['grasp_width'] if close else self.c['open_width'], self.c['gripper_speed'])
        else:
            Task.hand(self, close=close)

    def pump(self, seconds=.03):
        if self.background:
            time.sleep(seconds)
        else:
            rclpy.spin_once(self.node, timeout_sec=seconds)

    def wait(self, future, timeout):
        if not self.background:
            return super().wait(future, timeout)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            raise TimeoutError('ROS operation timed out')
        return future.result()

    def snapshot(self, timeout=5.):
        started = time.monotonic()
        while time.monotonic()-started < timeout:
            self.pump()
            now = self.node.get_clock().now().nanoseconds*1e-9
            if (self.received >= started and self.latest_stamp is not None
                    and -.05 <= now-self.latest_stamp <= .5):
                return dict(self.latest)
        raise Intervention('No fresh arm joint feedback; no automatic movement')

    def on_robot(self, msg):
        self.robot, self.robot_received = msg, time.monotonic()

    def robot_ok(self):
        if not self.hardware:
            return
        if getattr(self, 'fake_backend', None) is not None:
            self.fake_backend.check_health()
            return
        msg = self.robot
        if msg is None or time.monotonic()-self.robot_received > .5:
            raise Intervention('Franka robot-state feedback is stale/missing')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        now = self.node.get_clock().now().nanoseconds*1e-9
        if not -.05 <= now-stamp <= .5:
            raise Intervention('Franka robot-state timestamp is stale')
        if msg.robot_mode not in (msg.ROBOT_MODE_IDLE, msg.ROBOT_MODE_MOVE):
            raise Intervention(f'Robot mode {msg.robot_mode} does not permit automatic recovery')
        if any(getattr(msg.current_errors, field) for field in msg.current_errors.get_fields_and_field_types()):
            raise Intervention('Franka reports a robot error')
        flags = msg.collision_indicators
        vectors = (flags.is_cartesian_linear_collision, flags.is_cartesian_angular_collision,
                   flags.is_cartesian_linear_contact, flags.is_cartesian_angular_contact)
        if (any(flags.is_joint_collision) or any(flags.is_joint_contact)
                or any(any((v.x, v.y, v.z)) for v in vectors)):
            raise Intervention('Franka reports contact/collision; automatic retreat is inhibited')

    def status(self, state, message):
        self.node.get_logger().info(f'[{state}] {message}')
        data = dict(state=state, message=message)
        self.status_pub.publish(String(data=json.dumps(data)))
        self.log_event('pick_status', **data)

    def ensure_running(self):
        stop = getattr(self, 'stop_requested', None)
        if stop is not None and stop.is_set():
            raise OperatorStop('Shutdown requested; no further motion authorized')

    def grasp_target(self):
        return list(self.c['pick'])

    def pregrasp(self):
        xyz = self.grasp_target()
        xyz[2] += self.c['clearance']
        return xyz

    def seed_pose_goal(self, goal, carrying):
        if self.forced_solution is None:
            raise RuntimeError('New picker must use an explicitly evaluated IK candidate')
        goal.request.goal_constraints[0].joint_constraints = [JointConstraint(
            joint_name=n, position=self.forced_solution[n], tolerance_above=.001,
            tolerance_below=.001, weight=1.) for n in ARM]

    def state_pose(self, joints):
        carrying, placed = self.scene_mode()
        with self.vertical.scoped_scene(carrying, placed) as original:
            state = self.vertical.state(original, carrying, joints)
            pose = self.vertical.fk(state)
        return state, pose

    def scene_mode(self):
        return False, False

    def confirm_empty_context(self):
        """Extension hook after an explicit operator empty-hand attestation."""

    def audit(self, plan, collision=True):
        """Sample the timed spline, including approach detours and singularity margin."""
        limits = Limits(max_joint_step_rad=.35, sample_period_s=.04, sample_joint_step_rad=.02)
        samples = list(timed_samples(plan.trajectory, limits))
        worst_sigma, margin, travel = float('inf'), float('inf'), np.zeros(7)
        previous = None
        with self.vertical.scoped_scene(plan.carrying, plan.placed) as original:
            first_state = self.vertical.state(original, plan.carrying, samples[0])
            xyz, q = coordinates(self.vertical.fk(first_state))
            t, _, _ = self.model.evaluate(samples[0], self.policy.jacobian_length_m)
            # Cross-check parsed chain against MoveIt before trusting its metric.
            if np.linalg.norm(t[:3, 3]-xyz) > 1e-5:
                raise Intervention('URDF Jacobian FK disagrees with MoveIt TCP position')
            from scipy.spatial.transform import Rotation
            if angle_deg(Rotation.from_matrix(t[:3, :3]).as_quat(), q) > .01:
                raise Intervention('URDF Jacobian FK disagrees with MoveIt TCP orientation')
            for index, joints in enumerate(samples):
                if index and index % 100 == 0:
                    self.status('AUDITING_PATH', f'Checking timed path sample {index}/{len(samples)}')
                _, _, sigma = self.model.evaluate(joints, self.policy.jacobian_length_m)
                worst_sigma = min(worst_sigma, sigma)
                for n in ARM:
                    low, high = self.vertical.bounds[n]
                    margin = min(margin, joints[n]-low, high-joints[n])
                if margin < self.policy.joint_margin_rad or sigma < self.policy.min_sigma:
                    raise PickFailure(f'Path rejected: joint margin={margin:.4f} rad, normalized sigma={sigma:.4f}')
                values = np.array([joints[n] for n in ARM])
                if previous is not None:
                    travel += np.abs(values-previous)
                previous = values
                if max(travel) > self.policy.max_joint_travel_rad:
                    raise PickFailure('Path rejected: excessive accumulated joint travel')
                if collision:
                    state = self.vertical.state(original, plan.carrying, joints)
                    valid = self.vertical.call(self.vertical.validity,
                        GetStateValidity.Request(robot_state=state, group_name='fr3_arm'))
                    if not valid.valid and not is_support_touch(valid, plan.carrying):
                        contacts = [(c.contact_body_1, c.contact_body_2) for c in valid.contacts[:8]]
                        raise PickFailure(f'Path collision/invalid state: {contacts}')
        metrics = dict(min_sigma=worst_sigma, joint_margin_rad=margin,
                       travel_rad=travel.tolist(), samples=len(samples))
        self.log_event('path_audit', **metrics)
        return float(sum(travel) + 2*sum(travel[4:]) + .05/max(worst_sigma, 1e-9)), metrics

    def ik_candidates(self, xyz, start):
        candidates = []
        carrying, placed = self.scene_mode()
        with self.vertical.scoped_scene(carrying, placed) as original:
            for index, seed in enumerate(seeds(start, self.vertical.bounds, self.policy.ik_attempts)):
                self.ensure_running()
                self.status('FINDING_APPROACH', f'Finding approach IK seed {index+1}/{self.policy.ik_attempts}')
                req = GetPositionIK.Request()
                req.ik_request.group_name, req.ik_request.ik_link_name = 'fr3_arm', TCP
                req.ik_request.robot_state = self.vertical.state(original, carrying, seed)
                req.ik_request.pose_stamped.header.frame_id = BASE
                p = req.ik_request.pose_stamped.pose
                p.position.x, p.position.y, p.position.z = xyz
                p.orientation.x = 1.
                p.orientation.y = p.orientation.z = p.orientation.w = 0.
                req.ik_request.avoid_collisions = True
                req.ik_request.timeout.sec = 1
                result = self.vertical.call(self.ik, req, timeout=3)
                if result.error_code.val != MoveItErrorCodes.SUCCESS:
                    continue
                solved = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
                if any(n not in solved or not math.isfinite(solved[n]) for n in ARM):
                    continue
                if any(max(abs(solved[n]-other[n]) for n in ARM) < .05 for other in candidates):
                    continue
                candidates.append(solved)
        candidates.sort(key=lambda s: sum((s[n]-start[n])**2 for n in ARM))
        self.log_event('distinct_ik', candidates=candidates, reference_joints_rad=start)
        candidates = [s for s in candidates if not any(
            max(abs(s[n]-old[n]) for n in ARM) < .05 for old in self.failed_approaches)]
        return candidates[:self.policy.max_candidates]

    def reject_approach(self):
        if self.selected_approach is not None:
            self.failed_approaches.append(dict(self.selected_approach))

    def check_pick_candidate(self, approach, descent):
        """Extension point for full-cycle checks, e.g. post-grasp lift feasibility."""
        return 0.

    def find_approach(self, target):
        start = self.settled_snapshot() if self.hardware else dict(self.joints)
        start.update({f'fr3_finger_joint{i}': self.c['open_width']/2 for i in (1, 2)})
        candidates = self.ik_candidates(target, start)
        evaluated = []
        try:
            for index, solution in enumerate(candidates):
                self.ensure_running()
                self.status('CHECKING_CANDIDATE', f'Checking approach/descent candidate {index+1}/{len(candidates)}')
                self.joints, self.commands = dict(start), []
                self.forced_solution = solution
                self.execute = False
                try:
                    Task.motion(self, 'Candidate approach', target)
                    trajectory = self.commands[-1][2]
                    plan = Plan(copy.deepcopy(trajectory), copy.deepcopy(self.preview_start), list(target))
                    approach_score, _ = self.audit(plan)
                    # Task.motion advanced virtual joints to the planned endpoint.
                    descent, descent_state = self.vertical.plan(self.grasp_target())
                    descent_plan = Plan(descent, descent_state, self.grasp_target(), True)
                    descent_score, _ = self.audit(descent_plan, collision=False)
                    extra_score = self.check_pick_candidate(plan, descent_plan)
                    evaluated.append((approach_score+descent_score+extra_score, plan))
                except (PlanningInfrastructureError, TimeoutError) as exc:
                    raise Intervention(f'Planning infrastructure failed: {exc}') from exc
                except Intervention:
                    raise
                except RuntimeError as exc:
                    self.log_event('candidate_rejected', index=index, reason=str(exc))
                    self.node.get_logger().warning(f'Candidate rejected: {exc}')
        finally:
            self.forced_solution = None
            self.joints = start
            self.commands = []
        if not evaluated:
            raise PickFailure('NO_VALID_PICK_PLAN: no approach with a feasible descent')
        score, selected = min(evaluated, key=lambda item: item[0])
        jt = selected.trajectory.joint_trajectory
        self.selected_approach = dict(zip(jt.joint_names, jt.points[-1].positions))
        self.log_event('approach_selected', score=score)
        return selected

    def plan_descent(self):
        self.joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        _, pose = self.state_pose(self.joints)
        xyz, q = coordinates(pose)
        if (math.hypot(xyz[0]-self.c['pick'][0], xyz[1]-self.c['pick'][1])
                > self.c['endpoint_lateral_tolerance_m']
                or angle_deg(q, [1., 0., 0., 0.]) > self.c['endpoint_orientation_tolerance_deg']):
            raise PickFailure('NOT_ALIGNED: descent inhibited')
        try:
            trajectory, state = self.vertical.plan(self.grasp_target())
            plan = Plan(trajectory, state, self.grasp_target(), True)
            self.audit(plan, collision=False)
            return plan
        except (PlanningInfrastructureError, TimeoutError) as exc:
            raise Intervention(f'Planning infrastructure failed: {exc}') from exc
        except Intervention:
            raise
        except RuntimeError as exc:
            raise PickFailure(f'DESCENT_PLAN_FAILED: {exc}') from exc

    def open_hand(self):
        self.ensure_running()
        self.status('OPENING_HAND', 'Opening the empty hand')
        confirm('Open empty hand', self.hardware)
        self.ensure_running()
        if self.hardware:
            self.settled_snapshot()
            self.robot_ok()
        self.execute = self.hardware
        self.command_hand()
        self.execute = False

    def close_hand(self):
        self.ensure_running()
        confirm('Close gripper at verified grasp pose', self.hardware)
        self.ensure_running()
        if self.hardware:
            # Recheck after waiting at the approval prompt, before closing.
            if not self.verify(self.grasp_target())['accepted']:
                raise Intervention('Grasp pose changed during approval; hand was not closed')
            self.robot_ok()
        self.execute = self.hardware
        try:
            self.command_hand(close=True)
        except BaseException:
            self.holding = True  # uncertain grasp: do not open/restart automatically
            self.blocked = True
            raise Intervention('Gripper outcome uncertain; inspect the hand before another task')
        finally:
            self.execute = False
        if self.hardware:
            self.holding = True

    def verify(self, target):
        joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        state, pose = self.state_pose(joints)
        xyz, q = coordinates(pose)
        delta = [a-b for a, b in zip(xyz, target)]
        angle = angle_deg(q, [1., 0., 0., 0.])
        joint_error, tracking_ok = 0., True
        if self.last_plan is not None:
            jt = self.last_plan.trajectory.joint_trajectory
            wanted = dict(zip(jt.joint_names, jt.points[-1].positions))
            joint_error = max(abs(joints[n]-wanted[n]) for n in ARM)
            self.preview_start = self.last_plan.state
            try:
                self.check_endpoint(self.last_plan.trajectory, joints)
            except RuntimeError:
                tracking_ok = False
        accepted = (tracking_ok and joint_error <= .01
                    and math.hypot(*delta[:2]) <= self.c['endpoint_lateral_tolerance_m']
                    and abs(delta[2]) <= self.c['endpoint_z_tolerance_m']
                    and angle <= self.c['endpoint_orientation_tolerance_deg'])
        self.joints = joints
        error = dict(accepted=accepted, xyz_m=math.sqrt(sum(v*v for v in delta)),
                     angle_deg=angle, joint_rad=joint_error)
        self.log_event('requested_pose_check', requested_xyz_m=target, requested_quaternion_xyzw=[1., 0., 0., 0.],
                       measured_xyz_m=xyz, measured_quaternion_xyzw=q, **error)
        self.status('VERIFYING_POSE', f'Pose error: {error["xyz_m"]*1000:.3f} mm, {angle:.3f} deg; accepted={accepted}')
        return error

    def _monitor(self, plan):
        self.robot_ok()
        now = self.node.get_clock().now().nanoseconds*1e-9
        if self.latest_stamp is None or not -.05 <= now-self.latest_stamp <= .5:
            raise Intervention('Joint feedback became stale during execution')
        joints = dict(self.latest)
        t, _, sigma = self.model.evaluate(joints, self.policy.jacobian_length_m)
        if getattr(self, 'fake_backend', None) is not None:
            joint_error = self.fake_backend.tracking_error()
        else:
            desired = dict(zip(self.robot.desired_joint_state.name, self.robot.desired_joint_state.position))
            measured = dict(zip(self.robot.measured_joint_state.name, self.robot.measured_joint_state.position))
            if not all(n in desired and n in measured for n in ARM):
                raise Intervention('Franka joint tracking feedback is incomplete')
            if not all(math.isfinite(desired[n]) and math.isfinite(measured[n]) for n in ARM):
                raise Intervention('Franka joint tracking feedback is non-finite')
            joint_error = max(abs(measured[n]-desired[n]) for n in ARM)
        if joint_error > self.policy.monitor_joint_error_rad:
            raise PickFailure(f'Franka joint tracking error {joint_error:.4f} rad exceeds monitor limit')
        for name in ARM:
            low, high = self.vertical.bounds[name]
            if min(joints[name]-low, high-joints[name]) < self.policy.joint_margin_rad:
                raise PickFailure('Measured joints approach a joint limit')
        if sigma < self.policy.min_sigma:
            raise PickFailure('Measured posture approaches singularity threshold')
        if plan.vertical:
            start_joints = dict(zip(plan.state.joint_state.name, plan.state.joint_state.position))
            start_t, _, _ = self.model.evaluate(start_joints, self.policy.jacobian_length_m)
            start_xyz = start_t[:3, 3]
            from scipy.spatial.transform import Rotation
            start_q = Rotation.from_matrix(start_t[:3, :3]).as_quat()
            q = Rotation.from_matrix(t[:3, :3]).as_quat()
            lateral = math.hypot(t[0, 3]-start_xyz[0], t[1, 3]-start_xyz[1])
            angle = angle_deg(q, start_q)
            if (lateral > self.policy.monitor_xy_m or angle > self.policy.monitor_angle_deg
                    or t[2, 3] < min(start_xyz[2], plan.xyz[2])-.002
                    or t[2, 3] > max(start_xyz[2], plan.xyz[2])+.002):
                raise PickFailure('Measured motion left the checked vertical corridor')
        self.log_event('motion_sample', joints_rad=joints, xyz_m=t[:3, 3].tolist(), sigma=sigma,
                       max_joint_tracking_error_rad=joint_error)

    def cancel_and_settle(self, handle, result_future):
        self.status('STOPPING', 'Requesting cancellation and verifying the stopped arm')
        response = self.wait(handle.cancel_goal_async(), 5)
        result = self.wait(result_future, 10)
        if result.status not in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED):
            raise Intervention('Execution did not finish with confirmed cancellation/success')
        self.joints = self.settled_snapshot()
        self.robot_ok()
        return result

    def move_arm(self, plan, label, vertical=False):
        self.ensure_running()
        self.preview.publish(DisplayTrajectory(trajectory_start=plan.state, trajectory=[plan.trajectory]))
        confirm(f'{label}: review the planned path', self.hardware)
        self.ensure_running()
        self.label = self.active_motion_label = label
        if not self.hardware:
            jt = plan.trajectory.joint_trajectory
            self.joints.update(zip(jt.joint_names, jt.points[-1].positions))
            self.last_plan = plan
            return
        live = self.settled_snapshot()
        self.robot_ok()
        check_start(live, plan.trajectory, tolerance=.001)
        if plan.vertical:
            self.vertical.check_execution_start(live, plan.state)
        self.audit(plan, collision=True)  # refresh collision/kinematic checks after approval
        live = self.settled_snapshot()
        check_start(live, plan.trajectory, tolerance=.001)
        self.robot_ok()
        jt = plan.trajectory.joint_trajectory
        self.ensure_running()
        if getattr(self, 'fake_backend', None) is not None:
            self.fake_backend.verify_identity()
        self.log_event('execution_start', label=label, measured_start_joints_rad=live,
                       joint_names=list(jt.joint_names), trajectory=[dict(
                           time_s=p.time_from_start.sec+p.time_from_start.nanosec*1e-9,
                           positions_rad=list(p.positions), velocities_rad_s=list(p.velocities),
                           accelerations_rad_s2=list(p.accelerations)) for p in jt.points])
        send_future = self.executor.send_goal_async(ExecuteTrajectory.Goal(trajectory=plan.trajectory))
        try:
            handle = self.wait(send_future, 10)
        except BaseException as exc:
            # A missing acknowledgement does not prove the command was rejected.
            try:
                late_handle = self.wait(send_future, 5)
                if late_handle.accepted:
                    self.cancel_and_settle(late_handle, late_handle.get_result_async())
            except BaseException:
                pass
            raise Intervention('Execution acknowledgement uncertain; inspect stopped robot before restart') from exc
        if not handle.accepted:
            raise PickFailure('Trajectory execution was rejected; no recovery motion requested')
        result_future = handle.get_result_async()
        started = time.monotonic()
        try:
            while not result_future.done():
                self.pump(self.policy.monitor_period_s)
                if self.stop_requested.is_set():
                    raise Intervention('Shutdown requested')
                if time.monotonic()-started > self.policy.motion_timeout_s:
                    raise Intervention('Motion timeout')
                self._monitor(plan)
        except BaseException as exc:
            try:
                self.cancel_and_settle(handle, result_future)
            except BaseException as stop_error:
                raise Intervention(f'Cannot confirm a healthy stopped arm: {stop_error}') from exc
            if isinstance(exc, (OperatorStop, KeyboardInterrupt)):
                raise
            if isinstance(exc, Intervention) or not plan.vertical:
                raise Intervention(str(exc)) from exc
            raise DescentStopped(str(exc)) from exc
        result = result_future.result()
        self.log_event('execution_result', label=label, status=result.status,
                       moveit_error_code=result.result.error_code.val)
        if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise Intervention('Execution failed; automatic recovery inhibited until robot status is inspected')
        self.joints = self.settled_snapshot()
        self.robot_ok()
        self.last_plan = plan

    def recover(self, above):
        self.ensure_running()
        if self.holding:
            raise Intervention('Grasp state is uncertain; automatic recovery inhibited')
        self.joints = self.settled_snapshot() if self.hardware else self.joints
        self.robot_ok()
        _, pose = self.state_pose(self.joints)
        xyz, q = coordinates(pose)
        if (xyz[2] > above[2] + .001
                or math.hypot(xyz[0]-above[0], xyz[1]-above[1]) > self.policy.monitor_xy_m
                or angle_deg(q, [1., 0., 0., 0.]) > self.policy.monitor_angle_deg):
            raise Intervention('No verified vertical recovery corridor; intervention required')
        if above[2]-xyz[2] > .001:
            retreat_target = [xyz[0], xyz[1], above[2]]
            try:
                trajectory, state = self.vertical.plan(retreat_target)
                retreat = Plan(trajectory, state, retreat_target, True)
                self.audit(retreat, collision=False)
                self.move_arm(retreat, 'Recovery retreat', vertical=True)
                if not self.verify(retreat_target)['accepted']:
                    raise Intervention('Recovery retreat did not meet endpoint tolerances')
            except (OperatorStop, KeyboardInterrupt):
                raise
            except RuntimeError as exc:
                raise Intervention(f'Checked retreat failed: {exc}') from exc
        _, pose = self.state_pose(self.joints)
        actual, _ = coordinates(pose)
        if abs(actual[2]-above[2]) > self.c['endpoint_z_tolerance_m']:
            raise Intervention('Recovery did not reach pre-grasp height')


def run_cli(picker_class=Picker, workflow_class=PickWorkflow, node_name='pick',
            description=__doc__, configure_parser=None, prepare_request=None):
    """Shared process/service lifecycle; callers reuse picking without calling main."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--camera-target', required=True, type=Path)
    parser.add_argument('--policy', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Real-hardware execution with native Franka feedback')
    mode.add_argument('--fake-execute', action='store_true', help='Execute on verified GenericSystem mock hardware with animated fake gripper')
    parser.add_argument('--serve', action='store_true', help='Remain alive; ~/start queues a pick, ~/status and ~/result report outcomes')
    parser.add_argument('--log-dir', type=Path, default=Path('data/motion_logs'))
    parser.add_argument('--robot-state-topic', default='/franka_robot_state_broadcaster/robot_state')
    if configure_parser is not None:
        configure_parser(parser)
    args, ros_args = parser.parse_known_args()
    policy = Policy.from_config(yaml.safe_load(args.policy.read_text()) if args.policy else {})
    config = load_config(args.config, camera_mode=True)
    if max(config['velocity_scale'], config['acceleration_scale']) > .05:
        parser.error('Use velocity/acceleration scaling at most 0.05')
    # Keep ROS alive through Ctrl-C long enough to request/confirm cancellation.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(node_name)
    log = MotionLog(args.log_dir)
    print(f'Motion log: {log.path.resolve()}', flush=True)
    picker = None
    worker = None
    try:
        picker = picker_class(node, config, args.execute, log, policy, args.robot_state_topic)
        if args.fake_execute:
            picker.enable_fake_execution()
        gate = threading.Lock()

        def perform():
            try:
                if picker.holding or picker.blocked:
                    raise Intervention('Picker is holding/uncertain/blocked; inspect before another request')
                log.write('pick_request', execute=picker.hardware,
                          backend='fake' if args.fake_execute else 'real' if args.execute else 'preview', policy=policy.__dict__)
                picker.c = load_config(args.config, camera_mode=True)
                if max(picker.c['velocity_scale'], picker.c['acceleration_scale']) > .05:
                    raise PickFailure('Use velocity/acceleration scaling at most 0.05')
                picker.last_plan = None
                picker.failed_approaches = []
                picker.selected_approach = None
                if prepare_request is not None:
                    prepare_request(picker, args)
                picker.use_camera_target(load_camera_target(args.camera_target))
                if args.fake_execute:
                    picker.fake_backend.verify_identity()
                    picker.fake_backend.install_scene()
                if picker.hardware:
                    picker.settled_snapshot()
                    picker.robot_ok()
                outcome = workflow_class(picker, policy).run()
                result = dict(outcome=outcome if picker.hardware else 'PREVIEW_COMPLETE', success=True,
                              backend='fake' if args.fake_execute else 'real' if args.execute else 'preview',
                              holding=picker.holding, log_path=str(log.path.resolve()))
            except (OperatorStop, EOFError, KeyboardInterrupt):
                result = dict(outcome='OPERATOR_STOPPED', success=False, holding=picker.holding)
            except Exception as exc:
                log.write('pick_exception', exception_type=type(exc).__name__, traceback=traceback.format_exc())
                if isinstance(exc, (Intervention, PlanningInfrastructureError, TimeoutError)):
                    picker.blocked = True
                result = dict(outcome='INTERVENTION_REQUIRED' if picker.blocked else 'PICK_FAILED',
                              success=False, holding=picker.holding, error=str(exc))
            picker.status(result['outcome'], json.dumps(result))
            picker.result_pub.publish(String(data=json.dumps(result)))
            log.write('pick_result', **result)
            return result

        if not args.serve:
            return 0 if perform()['success'] else 1

        picker.background = True

        def start(request, response):
            nonlocal worker
            if picker.holding or picker.blocked:
                response.success, response.message = False, 'HOLDING/BLOCKED: inspect and explicitly confirm empty/healthy first'
            elif not gate.acquire(blocking=False):
                response.success, response.message = False, 'BUSY'
            else:
                def job():
                    try:
                        perform()
                    finally:
                        gate.release()
                worker = threading.Thread(target=job, daemon=True)
                worker.start()
                response.success, response.message = True, f'Accepted; observe /{node_name}/status and /{node_name}/result; approve motion in this terminal'
            return response

        def confirm_empty(request, response):
            nonlocal worker
            if not gate.acquire(blocking=False):
                response.success, response.message = False, 'BUSY'
                return response
            def clear_context():
                try:
                    picker.robot_ok()
                    # Explicit operator attestation, no gripper motion/fault reset.
                    # A worker keeps ROS spinning for any scene-update futures.
                    picker.confirm_empty_context()
                    picker.holding = picker.blocked = False
                    picker.log_event('operator_confirmed_empty')
                    picker.status('EMPTY_CONFIRMED', 'Operator empty-hand attestation processed; next task enabled')
                except Exception as exc:
                    picker.blocked = True
                    picker.log_event('empty_confirmation_failed', error=str(exc))
                    picker.status('INTERVENTION_REQUIRED', str(exc))
                finally:
                    gate.release()
            worker = threading.Thread(target=clear_context, daemon=True)
            worker.start()
            response.success, response.message = True, 'Empty-hand attestation accepted; wait for EMPTY_CONFIRMED status'
            return response

        services = [node.create_service(Trigger, '~/start', start),
                    node.create_service(Trigger, '~/confirm_empty', confirm_empty)]
        picker.status('IDLE', f'Service ready: /{node_name}/start. Update camera XYZ at the observation posture before each request.')
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(str(exc))
        log.write('startup_error', error=str(exc))
        return 1
    finally:
        if picker is not None:
            picker.stop_requested.set()
        if worker is not None and worker.is_alive():
            # Keep processing cancellation futures briefly during orderly exit.
            deadline = time.monotonic()+15
            while worker.is_alive() and time.monotonic() < deadline and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=.05)
        log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main():
    return run_cli()


if __name__ == '__main__':
    raise SystemExit(main())
