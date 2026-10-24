#!/usr/bin/env python3
"""Single-brick stopped refinement; explicit --execute and per-move approval."""
import argparse
import copy
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import rclpy
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose
from franka_msgs.action import Move, Grasp
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject, CollisionObject,
    Constraints, DisplayTrajectory, MoveItErrorCodes, OrientationConstraint, PlanningScene,
    PlanningSceneComponents, PositionConstraint)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetStateValidity
import yaml

from cartesian_jog import JogTask
from pick_place import box
from pick_place_cartesian import ARM, BASE, TCP, Limits, coordinates, timed_samples
from pick_place_stepwise import OperatorStop, check_start, confirm
from vision_core import Refinement, angle, matrix, pose_dict, validate_config

OBJECT = 'fr3_vision_brick'


def ros_pose(transform):
    p = Pose()
    d = pose_dict(transform)
    p.position.x, p.position.y, p.position.z = d['xyz_m']
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = d['quaternion_xyzw']
    return p


def pose_matrix(pose):
    xyz, q = coordinates(pose)
    return matrix({'xyz_m': xyz, 'quaternion_xyzw': q})


def object_box(transform, size, frame=BASE):
    obj = box(OBJECT, size, transform[:3, 3], frame)
    obj.primitive_poses = [ros_pose(transform)]
    return obj


def finger_contacts(acm):
    acm = copy.deepcopy(acm)
    names = (OBJECT, 'fr3_leftfinger', 'fr3_rightfinger')
    for name in names:
        if name not in acm.entry_names:
            for row in acm.entry_values:
                row.enabled.append(False)
            acm.entry_names.append(name)
            acm.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(acm.entry_names)))
    i = acm.entry_names.index(OBJECT)
    for name in names[1:]:
        j = acm.entry_names.index(name)
        acm.entry_values[i].enabled[j] = acm.entry_values[j].enabled[i] = True
    return acm


class VisionTask(JogTask):
    def __init__(self, node, c, execute, auto_confirm):
        super().__init__(node, SimpleNamespace(execute=execute, scale=c['velocity_scale'],
            camera_object_id='handeye_camera', table_object_id='fr3_vision_table'))
        self.config = c
        self.c['acceleration_scale'] = c['acceleration_scale']
        self.auto_confirm = auto_confirm
        if auto_confirm and (not execute or self.hardware):
            raise RuntimeError('--auto-confirm is only permitted with verified mock hardware execution.')
        if self.hardware and c['backend'] != 'sam3':
            raise RuntimeError('Mock perception is forbidden on a real arm.')
        self.arm = ActionClient(node, MoveGroup, 'move_action')
        if not self.arm.wait_for_server(timeout_sec=10):
            raise RuntimeError('MoveIt move_action unavailable.')
        self.observer = node.create_client(Trigger, 'vision/observe')
        if not self.observer.wait_for_service(timeout_sec=30):
            raise RuntimeError('vision/observe unavailable; start vision.launch.py first.')
        self.move = self.grasp = None
        if self.hardware:
            self.move = ActionClient(node, Move, 'franka_gripper/move')
            self.grasp = ActionClient(node, Grasp, 'franka_gripper/grasp')
            if not all(client.wait_for_server(timeout_sec=10) for client in (self.move, self.grasp)):
                raise RuntimeError('Real Franka gripper actions unavailable.')
        self.object_in_tcp = None
        self.install_geometry()

    def approve(self, label):
        if self.auto_confirm:
            self.node.get_logger().info('MOCK AUTO-CONFIRM: '+label)
        else:
            confirm(label, self.execute)

    def scene(self):
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.ROBOT_STATE
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        return self.vertical.call(self.scene_client, req).scene

    def apply_scene(self, diff):
        diff.is_diff = diff.robot_state.is_diff = True
        if not self.vertical.call(self.vertical.apply, ApplyPlanningScene.Request(scene=diff)).success:
            raise RuntimeError('Planning scene update failed.')

    def install_geometry(self):
        scene = self.scene()
        if any(o.object.id == OBJECT for o in scene.robot_state.attached_collision_objects):
            raise RuntimeError('A previous vision brick remains attached. Inspect scene/robot before restarting.')
        diff = PlanningScene()
        t = self.config['table']
        diff.world.collision_objects = [box(self.table_object_id, t['size_m'], t['center_m'])]
        if self.hardware:
            c = self.config['camera_collision']
            attached = AttachedCollisionObject(link_name=TCP, touch_links=['fr3_hand'])
            attached.object = box(self.camera_object_id, c['size_m'], c['center_m'], TCP)
            diff.robot_state.attached_collision_objects = [attached]
        self.apply_scene(diff)

    def update_object(self, obj):
        diff = PlanningScene()
        diff.allowed_collision_matrix = finger_contacts(self.scene().allowed_collision_matrix)
        diff.world.collision_objects = [object_box(obj, self.config['brick_size_m'])]
        self.apply_scene(diff)

    def observe(self):
        before = self.settled_snapshot()
        requested = self.node.get_clock().now().nanoseconds
        response = self.wait(self.observer.call_async(Trigger.Request()), self.config['perception']['inference_timeout_s'])
        if response is None or not response.success:
            raise RuntimeError('Observation failed: '+(response.message if response else 'no response'))
        result = json.loads(response.message)
        now = self.node.get_clock().now().nanoseconds
        if (result.get('schema') != 1 or result.get('frame_id') != BASE
                or result.get('backend') != self.config['backend']
                or not requested <= result['capture_stamp_ns'] <= now
                or (now-result['capture_stamp_ns'])*1e-9 > self.config['refinement']['max_result_age_s']
                or len(result.get('objects', [])) != 1):
            raise RuntimeError('Invalid, stale or mismatched observation.')
        after = self.settled_snapshot()
        if any(abs(before[n]-after[n]) > .002 for n in ARM):
            raise RuntimeError('Robot moved during observation; discard detection.')
        self.joints = after
        return result['objects'][0]

    def validate_path(self, trajectory, start_state, fixed_height=None):
        # Generic transfers may have coarser joint waypoints than vertical paths.
        limits = Limits(max_joint_step_rad=math.pi if fixed_height is None else .1)
        scene = self.scene()
        for joints in timed_samples(trajectory, limits):
            if any(not self.vertical.bounds[n][0] <= v <= self.vertical.bounds[n][1] for n, v in joints.items()):
                raise RuntimeError('Trajectory exceeds joint limits.')
            candidate = self.vertical.state(scene.robot_state, False, joints)
            response = self.vertical.call(self.vertical.validity,
                GetStateValidity.Request(robot_state=candidate, group_name='fr3_arm'))
            if not response.valid:
                raise RuntimeError('Trajectory invalid in current collision scene.')
            if fixed_height is not None:
                fk = pose_matrix(self.vertical.fk(candidate))
                if abs(fk[2, 3]-fixed_height) > .001 or fk[2, 2] > -math.cos(.02):
                    raise RuntimeError('Refinement leaves its horizontal tool-down corridor.')

    def execute_plan(self, label, trajectory, start_state, target, vertical=False, fixed_height=None):
        self.preview.publish(DisplayTrajectory(trajectory_start=start_state, trajectory=[trajectory]))
        self.approve(label)
        if not self.execute:
            return
        if vertical:
            self.vertical.revalidate(trajectory, start_state, target[2, 3])
        else:
            self.validate_path(trajectory, start_state, fixed_height)
        live = self.settled_snapshot()
        check_start(live, trajectory, tolerance=.001)
        self.vertical.check_execution_start(live, start_state)
        response = self.action(self.executor, ExecuteTrajectory.Goal(trajectory=trajectory))
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'Execution failed: {response.error_code.val}.')
        actual = pose_matrix(self.measured_pose())
        if np.linalg.norm(actual[:3, 3]-target[:3, 3]) > .002 or angle(actual, target) > 1.:
            raise RuntimeError('Measured TCP endpoint differs from commanded pose.')

    def motion(self, label, target, fixed_height=None):
        if fixed_height is not None:
            return self.correction(label, target, fixed_height)
        self.joints = self.settled_snapshot()
        start_state = self.vertical.state(self.scene().robot_state, False, self.joints)
        actual = pose_matrix(self.vertical.fk(start_state))
        if np.linalg.norm(actual[:3, 3]-target[:3, 3]) < .0005 and angle(actual, target) < .1:
            return
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name, req.allowed_planning_time, req.num_planning_attempts = 'fr3_arm', 10., 5
        req.start_state = start_state
        req.max_velocity_scaling_factor = self.c['velocity_scale']
        req.max_acceleration_scaling_factor = self.c['acceleration_scale']
        position = PositionConstraint()
        position.header.frame_id, position.link_name = BASE, TCP
        region = box('goal_region', [.001]*3, target[:3, 3])
        position.constraint_region.primitives = region.primitives
        position.constraint_region.primitive_poses = region.primitive_poses
        position.weight = 1.
        orientation = OrientationConstraint()
        orientation.header.frame_id, orientation.link_name = BASE, TCP
        orientation.orientation = ros_pose(target).orientation
        orientation.absolute_x_axis_tolerance = orientation.absolute_y_axis_tolerance = .005
        orientation.absolute_z_axis_tolerance = .005
        orientation.weight = 1.
        req.goal_constraints = [Constraints(position_constraints=[position], orientation_constraints=[orientation])]
        goal.planning_options.plan_only = True
        response = self.action(self.arm, goal)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'{label}: planning failed ({response.error_code.val}).')
        trajectory = response.planned_trajectory
        check_start(self.joints, trajectory, tolerance=.001)
        self.validate_path(trajectory, start_state, fixed_height)
        self.execute_plan(label, trajectory, start_state, target, fixed_height=fixed_height)

    def correction(self, label, target, height):
        self.joints = self.settled_snapshot()
        state = self.vertical.state(self.scene().robot_state, False, self.joints)
        start = pose_matrix(self.vertical.fk(state))
        distance = np.linalg.norm(target[:3, 3]-start[:3, 3])
        n = max(1, math.ceil(distance/.002), math.ceil(angle(start, target)/2.))
        rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])))
        req = GetCartesianPath.Request()
        req.header.frame_id, req.group_name, req.link_name = BASE, 'fr3_arm', TCP
        req.start_state = state
        req.max_step, req.jump_threshold, req.revolute_jump_threshold = .002, 2., .1
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = self.c['velocity_scale']
        req.max_acceleration_scaling_factor = self.c['acceleration_scale']
        for fraction in np.linspace(1./n, 1., n):
            waypoint = start.copy()
            waypoint[:3, 3] += fraction*(target[:3, 3]-start[:3, 3])
            waypoint[:3, :3] = rotations(fraction).as_matrix()
            req.waypoints.append(ros_pose(waypoint))
        response = self.vertical.call(self.vertical.cartesian, req, timeout=60)
        if (response.error_code.val != MoveItErrorCodes.SUCCESS
                or not math.isfinite(response.fraction) or abs(response.fraction-1.) > 1e-6):
            raise RuntimeError(f'Incomplete Cartesian correction: {response.fraction}.')
        check_start(self.joints, response.solution, tolerance=.001)
        self.validate_path(response.solution, state, height)
        self.execute_plan(label, response.solution, state, target, fixed_height=height)

    def vertical_motion(self, label, target):
        self.joints = self.settled_snapshot()
        current = pose_matrix(self.measured_pose())
        if abs(current[2, 3]-target[2, 3]) < .001:
            return
        if angle(current, target) > .5:
            raise RuntimeError('Vertical segment requires matching starting and target orientations.')
        trajectory, start = self.vertical.plan(target[:3, 3])
        self.execute_plan(label, trajectory, start, target, vertical=True)

    def hand(self, close=False):
        self.approve('GRASP selected brick' if close else 'OPEN hand')
        if not self.hardware:
            self.node.get_logger().info('Mock gripper event only; no real gripper command.')
            return
        g = self.config['gripper']
        goal = Grasp.Goal() if close else Move.Goal()
        goal.width = g['grasp_width_m'] if close else g['open_width_m']
        goal.speed = g['speed_m_s']
        if close:
            goal.force = g['force_n']
            goal.epsilon.inner = goal.epsilon.outer = .003
        response = self.action(self.grasp if close else self.move, goal)
        if not response.success:
            raise RuntimeError(f'Gripper failed: {response.error}')

    def attach(self, obj):
        self.object_in_tcp = np.linalg.inv(pose_matrix(self.measured_pose())) @ obj
        attached = AttachedCollisionObject(link_name=TCP, touch_links=['fr3_leftfinger', 'fr3_rightfinger'])
        attached.object = object_box(self.object_in_tcp, self.config['brick_size_m'], TCP)
        diff = PlanningScene()
        # MoveIt removes the matching world object while attaching it. An
        # additional REMOVE in this same diff would target an already absent ID.
        diff.robot_state.attached_collision_objects = [attached]
        self.apply_scene(diff)

    def support_contact(self, allow, obj=None):
        # A brick resting on the table starts in contact with its support.
        # Permit only this pair during the strictly upward lift, then revoke it.
        if allow:
            half_extent_z = np.abs(obj[2, :3]) @ (np.asarray(self.config['brick_size_m'])/2)
            bottom = obj[2, 3]-half_extent_z
            table = self.config['table']
            top = table['center_m'][2]+table['size_m'][2]/2
            if bottom < top-.001:
                raise RuntimeError('Estimated brick penetrates measured table; inspect perception/calibration.')
        acm = self.scene().allowed_collision_matrix
        for name in (OBJECT, self.table_object_id):
            if name not in acm.entry_names:
                for row in acm.entry_values:
                    row.enabled.append(False)
                acm.entry_names.append(name)
                acm.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(acm.entry_names)))
        i, k = acm.entry_names.index(OBJECT), acm.entry_names.index(self.table_object_id)
        acm.entry_values[i].enabled[k] = acm.entry_values[k].enabled[i] = allow
        diff = PlanningScene(allowed_collision_matrix=acm)
        self.apply_scene(diff)

    def detach(self):
        obj = pose_matrix(self.measured_pose()) @ self.object_in_tcp
        attached = AttachedCollisionObject(link_name=TCP)
        attached.object.id, attached.object.operation = OBJECT, CollisionObject.REMOVE
        diff = PlanningScene()
        diff.robot_state.attached_collision_objects = [attached]
        diff.world.collision_objects = [object_box(obj, self.config['brick_size_m'])]
        self.apply_scene(diff)
        self.object_in_tcp = None

    def run(self):
        self.node.get_logger().info('REAL ARM' if self.hardware else 'MOCK ARM / PREVIEW')
        self.motion('Move to taught observation pose', matrix(self.config['observation_tcp']))
        if not self.execute:
            self.node.get_logger().info('Preview only: stopping after observation-pose preview. Use --execute on fake hardware for full cycle.')
            return
        self.hand()
        refine = Refinement(self.config)
        obj = refine.observe(self.observe())
        self.update_object(obj)
        # Obtain the desired tool-down grasp using the same adapter as refinement.
        raw_grasp = refine.grasp_pose(obj)
        approach = raw_grasp.copy()
        approach[2, 3] += self.config['refinement']['clearance_m']
        self.motion('Approach above selected brick', approach)
        while True:
            obj = refine.observe(self.observe())
            self.update_object(obj)
            current = pose_matrix(self.measured_pose())
            decision, target = refine.next_pose(current, obj)
            self.node.get_logger().info(f'Refinement {refine.iterations}: {decision}; stable={refine.stable}')
            if decision == 'ready':
                self.vertical_motion('FINAL DESCENT to grasp', target)
                break
            if decision == 'correct':
                self.motion('Bounded XY/yaw correction at fixed height', target, fixed_height=current[2, 3])
        self.hand(close=True)
        self.attach(obj)
        self.support_contact(True, obj)
        lift = pose_matrix(self.measured_pose())
        lift[2, 3] += self.config['refinement']['clearance_m']
        try:
            self.vertical_motion('Lift vertically', lift)
        finally:
            self.support_contact(False)
        place = matrix(self.config['place_tcp'])
        above_place = place.copy()
        above_place[2, 3] += self.config['refinement']['clearance_m']
        self.motion('Transfer above taught destination', above_place)
        self.vertical_motion('Lower to taught release TCP pose', place)
        self.hand()
        self.detach()
        self.vertical_motion('Retreat vertically', above_place)
        self.node.get_logger().info('One separated-brick cycle complete. No automatic second pick.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--auto-confirm', action='store_true', help='Verified mock hardware only')
    args, ros_args = parser.parse_known_args()
    if not args.auto_confirm and not sys.stdin.isatty():
        parser.error('Use an interactive terminal. --auto-confirm is only for mock execution.')
    try:
        with open(args.config, encoding='utf-8') as stream:
            config = validate_config(yaml.safe_load(stream))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Configuration incomplete/invalid: {exc}')
    rclpy.init(args=ros_args)
    node = rclpy.create_node('fr3_vision_pick_place')
    try:
        VisionTask(node, config, args.execute, args.auto_confirm).run()
    except (OperatorStop, EOFError, KeyboardInterrupt):
        node.get_logger().warning('Stopped. No automatic release, retreat or recovery.')
    except Exception as exc:
        node.get_logger().error(f'Aborted: {exc}. No automatic release or recovery.')
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
