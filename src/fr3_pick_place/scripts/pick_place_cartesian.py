"""Collision-checked vertical planning through ROS 2 Humble MoveIt services.

The service's timed trajectory is checked using FK and collision queries at
its waypoints and dense samples of the controller's default spline interpolation.
This is a sampled planned-path check, not a continuous physical safety guarantee.
"""
import copy
import math
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass

from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType
from moveit_msgs.msg import (
    AllowedCollisionEntry, AttachedCollisionObject, CollisionObject, Constraints,
    MoveItErrorCodes, OrientationConstraint, PlanningScene, PlanningSceneComponents,
)
from moveit_msgs.srv import (
    ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionFK, GetPositionIK, GetStateValidity,
)

from pick_place import box, block_object

ARM = [f'fr3_joint{i}' for i in range(1, 8)]
BASE = 'fr3_link0'
TCP = 'fr3_hand_tcp'
BLOCK = 'fr3_demo_block'
TABLE = 'fr3_demo_table'


class PlanningInfrastructureError(RuntimeError):
    """Planning services/scene restoration failed; do not try another motion."""


def is_support_touch(validity, carrying):
    """Only numerical zero-depth contact with the support, never robot contact."""
    return (carrying and bool(validity.contacts)
            and all({contact.contact_body_1, contact.contact_body_2} == {TABLE, BLOCK}
                    and math.isfinite(contact.depth) and abs(contact.depth) <= 1e-9
                    for contact in validity.contacts))


@dataclass(frozen=True)
class Limits:
    step_m: float = 0.002
    line_tolerance_m: float = 0.001
    orientation_tolerance_deg: float = 0.5
    target_xy_tolerance_m: float = 0.002
    max_joint_step_rad: float = 0.1
    sample_period_s: float = 0.02
    sample_joint_step_rad: float = 0.01
    max_validation_samples: int = 5000
    relative_jump_factor: float = 2.0


def coordinates(pose):
    p = pose.position
    q = pose.orientation
    xyz = [p.x, p.y, p.z]
    quat = [q.x, q.y, q.z, q.w]
    if not all(math.isfinite(v) for v in xyz + quat):
        raise RuntimeError('FK returned non-finite pose values.')
    norm = math.sqrt(sum(v * v for v in quat))
    if abs(norm - 1.) > 1e-3:
        raise RuntimeError('FK returned an invalid orientation.')
    return xyz, [v / norm for v in quat]


def seconds(duration):
    return duration.sec + duration.nanosec * 1e-9


def timed_samples(trajectory, limits):
    """Evaluate default JTC linear/cubic/quintic interpolation without retiming."""
    jt = trajectory.joint_trajectory
    if len(jt.joint_names) != 7 or set(jt.joint_names) != set(ARM) or len(jt.points) < 2:
        raise RuntimeError('Cartesian trajectory must contain all seven arm joints and two points.')
    if trajectory.multi_dof_joint_trajectory.points:
        raise RuntimeError('Unexpected multi-DOF Cartesian trajectory.')
    size = len(jt.joint_names)
    times = [seconds(p.time_from_start) for p in jt.points]
    if times[0] != 0. or any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError('Cartesian trajectory timing is absent or not strictly increasing from zero.')
    for p in jt.points:
        if len(p.positions) != size or not all(math.isfinite(x) for x in p.positions):
            raise RuntimeError('Invalid Cartesian joint positions.')
        for values in (p.velocities, p.accelerations):
            if len(values) not in (0, size) or not all(math.isfinite(x) for x in values):
                raise RuntimeError('Invalid Cartesian trajectory derivatives.')
        if p.accelerations and not p.velocities:
            raise RuntimeError('Accelerations supplied without velocities.')
    if any(abs(v) > 1e-6 for p in (jt.points[0], jt.points[-1]) for v in p.velocities):
        raise RuntimeError('Vertical segment must start and end at rest.')
    count = 1
    yield dict(zip(jt.joint_names, jt.points[0].positions))
    for i, (a, b) in enumerate(zip(jt.points, jt.points[1:])):
        dt = times[i + 1] - times[i]
        jump = max(abs(y - x) for x, y in zip(a.positions, b.positions))
        if jump > limits.max_joint_step_rad:
            raise RuntimeError(f'Cartesian joint discontinuity ({jump:.3f} rad).')
        n = max(1, math.ceil(dt / limits.sample_period_s),
                math.ceil(jump / limits.sample_joint_step_rad))
        count += n
        if count > limits.max_validation_samples:
            raise RuntimeError('Cartesian trajectory exceeds the validation sample budget.')
        if bool(a.velocities) != bool(b.velocities) or bool(a.accelerations) != bool(b.accelerations):
            raise RuntimeError('Inconsistent trajectory interpolation derivatives.')
        for j in range(1, n + 1):
            u = j / n
            values = []
            for k, (q0, q1) in enumerate(zip(a.positions, b.positions)):
                if a.accelerations:
                    c0, c1, c2 = q0, a.velocities[k] * dt, .5 * a.accelerations[k] * dt * dt
                    d = q1 - c0 - c1 - c2
                    e = b.velocities[k] * dt - c1 - 2 * c2
                    f = b.accelerations[k] * dt * dt - 2 * c2
                    c3, c4, c5 = 10*d - 4*e + .5*f, -15*d + 7*e - f, 6*d - 3*e + .5*f
                    q = c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * c5))))
                elif a.velocities:
                    q = ((2*u**3 - 3*u*u + 1)*q0 + (u**3 - 2*u*u + u)*dt*a.velocities[k]
                         + (-2*u**3 + 3*u*u)*q1 + (u**3 - u*u)*dt*b.velocities[k])
                else:
                    q = q0 + u * (q1 - q0)
                if not math.isfinite(q):
                    raise RuntimeError('Non-finite spline interpolation.')
                values.append(q)
            yield dict(zip(jt.joint_names, values))


def validate_vertical(trajectory, start_pose, target_z, inspect_state, limits):
    """inspect_state checks collisions and returns FK for each sampled joint state."""
    start, quat = coordinates(start_pose)
    direction = 1. if target_z >= start[2] else -1.
    previous = start[2]
    max_xy = max_angle = 0.
    count = 0
    for joints in timed_samples(trajectory, limits):
        xyz, q = coordinates(inspect_state(joints))
        xy = math.hypot(xyz[0] - start[0], xyz[1] - start[1])
        angle = math.degrees(2 * math.acos(min(1., abs(sum(a*b for a, b in zip(quat, q))))))
        if xy > limits.line_tolerance_m or angle > limits.orientation_tolerance_deg:
            raise RuntimeError(f'Timed path leaves vertical corridor: {xy*1000:.2f} mm, {angle:.3f} deg.')
        if (xyz[2] < min(start[2], target_z) - limits.line_tolerance_m
                or xyz[2] > max(start[2], target_z) + limits.line_tolerance_m
                or direction * (xyz[2] - previous) < -0.0001):
            raise RuntimeError('Timed vertical path overshoots or reverses direction.')
        previous = xyz[2]
        max_xy, max_angle = max(max_xy, xy), max(max_angle, angle)
        count += 1
    if abs(previous - target_z) > limits.line_tolerance_m:
        raise RuntimeError('Cartesian path did not reach the requested height.')
    return count, max_xy, max_angle


def allow_fingers(acm):
    acm = copy.deepcopy(acm)
    for name in (BLOCK, 'fr3_leftfinger', 'fr3_rightfinger'):
        if name not in acm.entry_names:
            for row in acm.entry_values:
                row.enabled.append(False)
            acm.entry_names.append(name)
            acm.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(acm.entry_names)))
    i = acm.entry_names.index(BLOCK)
    for name in ('fr3_leftfinger', 'fr3_rightfinger'):
        j = acm.entry_names.index(name)
        acm.entry_values[i].enabled[j] = acm.entry_values[j].enabled[i] = True
    return acm


def joint_bounds(urdf):
    joints = {j.get('name'): j for j in ET.fromstring(urdf).findall('joint')}
    bounds = {}
    for name in ARM:
        joint = joints.get(name)
        limit = joint.find('limit') if joint is not None else None
        if limit is None:
            raise RuntimeError(f'Robot model is missing joint limits for {name}.')
        lower, upper = float(limit.get('lower')), float(limit.get('upper'))
        safety = joint.find('safety_controller')
        if safety is not None:
            lower = max(lower, float(safety.get('soft_lower_limit', lower)))
            upper = min(upper, float(safety.get('soft_upper_limit', upper)))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise RuntimeError(f'Invalid robot joint limits for {name}.')
        bounds[name] = (lower, upper)
    return bounds


class CartesianPlanner:
    def __init__(self, task):
        self.task = task
        self.limits = Limits()
        self.cartesian = task.node.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.fk_client = task.node.create_client(GetPositionFK, 'compute_fk')
        self.validity = task.node.create_client(GetStateValidity, 'check_state_validity')
        self.apply = task.node.create_client(ApplyPlanningScene, 'apply_planning_scene')
        self.parameters = task.node.create_client(GetParameters, 'move_group/get_parameters')
        for client in (self.cartesian, self.fk_client, self.validity, self.apply, self.parameters):
            if not client.wait_for_service(timeout_sec=10):
                raise RuntimeError('MoveIt Cartesian/FK/state-validity/planning-scene services unavailable.')
        response = self.call(self.parameters, GetParameters.Request(names=['robot_description']))
        if len(response.values) != 1 or response.values[0].type != ParameterType.PARAMETER_STRING:
            raise RuntimeError('MoveIt robot_description parameter unavailable.')
        self.bounds = joint_bounds(response.values[0].string_value)
        self.controller_parameters = None
        if task.hardware:
            self.controller_parameters = task.node.create_client(
                GetParameters, 'fr3_arm_controller/get_parameters')
            if not self.controller_parameters.wait_for_service(timeout_sec=10):
                raise RuntimeError('Cannot verify arm-controller interpolation settings.')
            self.check_interpolation()

    def call(self, client, req, timeout=30):
        try:
            result = self.task.wait(client.call_async(req), timeout)
        except TimeoutError as exc:
            raise PlanningInfrastructureError(str(exc)) from exc
        if result is None:
            raise PlanningInfrastructureError('MoveIt service returned no response.')
        return result

    @contextmanager
    def scoped_scene(self, carrying, placed):
        # GetCartesianPath has no planning_scene_diff request field. Apply only
        # our table/block/ACM temporarily, then restore them before any prompt.
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                                     | PlanningSceneComponents.ROBOT_STATE
                                     | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
                                     | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        original = self.call(self.task.scene_client, req).scene
        old = {obj.id: obj for obj in original.world.collision_objects}
        diff = PlanningScene(is_diff=True)
        diff.robot_state.is_diff = True
        diff.allowed_collision_matrix = allow_fingers(original.allowed_collision_matrix)
        diff.world.collision_objects = [box(TABLE, self.task.c['table_size'], self.task.c['table_center'])]
        if carrying:
            # MoveIt reports failure when asked to remove an absent world object.
            if BLOCK in old:
                diff.world.collision_objects.append(CollisionObject(id=BLOCK, operation=CollisionObject.REMOVE))
        else:
            diff.world.collision_objects.append(block_object(self.task.c, placed=placed))
        restore = PlanningScene(is_diff=True)
        restore.robot_state.is_diff = True
        restore.allowed_collision_matrix = copy.deepcopy(original.allowed_collision_matrix)
        restore.world.collision_objects = [
            copy.deepcopy(old[obj.id]) if obj.id in old else
            CollisionObject(id=obj.id, operation=CollisionObject.REMOVE)
            for obj in diff.world.collision_objects]
        failure = None
        try:
            if not self.call(self.apply, ApplyPlanningScene.Request(scene=diff)).success:
                raise PlanningInfrastructureError('Could not install Cartesian collision scene.')
            yield original.robot_state
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if not self.call(self.apply, ApplyPlanningScene.Request(scene=restore)).success:
                detail = f' Original failure: {failure}' if failure is not None else ''
                raise PlanningInfrastructureError('Could not restore the planning scene; abort and inspect it.' + detail) from failure

    def state(self, original, carrying, joints):
        state = copy.deepcopy(original)
        state.is_diff = False
        values = dict(zip(state.joint_state.name, state.joint_state.position))
        values.update(joints)
        state.joint_state.name = list(values)
        state.joint_state.position = list(values.values())
        state.joint_state.velocity = []
        state.joint_state.effort = []
        state.joint_state.header.stamp.sec = state.joint_state.header.stamp.nanosec = 0
        # Preserve camera and other attachments, replace only our carried brick.
        state.attached_collision_objects = [
            obj for obj in state.attached_collision_objects if obj.object.id != BLOCK]
        if carrying:
            attached = AttachedCollisionObject(link_name=TCP)
            attached.touch_links = ['fr3_leftfinger', 'fr3_rightfinger']
            attached.object = block_object(self.task.c, carrying=True)
            state.attached_collision_objects.append(attached)
        return state

    def fk(self, state):
        req = GetPositionFK.Request(robot_state=state, fk_link_names=[TCP])
        req.header.frame_id = BASE
        response = self.call(self.fk_client, req)
        if (response.error_code.val != MoveItErrorCodes.SUCCESS or len(response.pose_stamped) != 1
                or response.fk_link_names != [TCP] or response.pose_stamped[0].header.frame_id != BASE):
            raise RuntimeError('TCP forward kinematics failed or returned an unexpected frame.')
        return response.pose_stamped[0].pose

    def check_execution_start(self, live_joints, approved_state):
        self.check_interpolation()
        actual = copy.deepcopy(approved_state)
        values = dict(zip(actual.joint_state.name, actual.joint_state.position))
        values.update(live_joints)
        actual.joint_state.name = list(values)
        actual.joint_state.position = list(values.values())
        planned_xyz, planned_q = coordinates(self.fk(approved_state))
        actual_xyz, actual_q = coordinates(self.fk(actual))
        distance = math.sqrt(sum((a-b)**2 for a, b in zip(actual_xyz, planned_xyz)))
        angle = math.degrees(2*math.acos(min(1., abs(sum(a*b for a, b in zip(actual_q, planned_q))))))
        if distance > self.limits.line_tolerance_m or angle > self.limits.orientation_tolerance_deg:
            raise RuntimeError('Measured TCP no longer matches the approved vertical-path start; replan.')

    def check_interpolation(self):
        if self.controller_parameters is not None:
            r = self.call(self.controller_parameters, GetParameters.Request(names=['interpolation_method']))
            if (len(r.values) != 1 or r.values[0].type != ParameterType.PARAMETER_STRING
                    or r.values[0].string_value != 'splines'):
                raise RuntimeError('Arm controller must use the validated splines interpolation mode.')

    def partial_path_diagnostic(self, response, original, carrying, start, target_z):
        """Read-only IK/validity probe; its solution is never returned for execution."""
        details = [f'MoveIt code={response.error_code.val}']
        try:
            joints = dict(self.task.joints)
            jt = response.solution.joint_trajectory
            if jt.points:
                joints.update(zip(jt.joint_names, jt.points[-1].positions))
            seed = self.state(original, carrying, joints)
            last, _ = coordinates(self.fk(seed))
            details.append(f'last returned TCP Z={last[2]:.6f} m; target Z={target_z:.6f} m')
            if 'block_size' in self.task.c:
                details.append(f'block XYZ size={self.task.c["block_size"]} m')
            client = self.task.node.create_client(GetPositionIK, 'compute_ik')
            if not client.wait_for_service(timeout_sec=2):
                return '; '.join(details + ['IK diagnostic service unavailable'])
            req = GetPositionIK.Request()
            req.ik_request.group_name, req.ik_request.ik_link_name = 'fr3_arm', TCP
            req.ik_request.robot_state = seed
            req.ik_request.pose_stamped.header.frame_id = BASE
            probe = copy.deepcopy(start)
            direction = 1. if target_z >= start.position.z else -1.
            probe.position.z = (min(target_z, last[2] + self.limits.step_m) if direction > 0
                                else max(target_z, last[2] - self.limits.step_m))
            req.ik_request.pose_stamped.pose = probe
            req.ik_request.timeout.sec = 1
            # Deliberately inspect an unchecked IK candidate, never a motion goal.
            req.ik_request.avoid_collisions = False
            result = self.call(client, req, timeout=5)
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                return '; '.join(details + [f'next-pose IK failed (code {result.error_code.val})'])
            candidate_joints = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
            if any(n not in candidate_joints for n in ARM):
                raise RuntimeError('IK diagnostic omitted arm joints')
            joints.update(candidate_joints)
            candidate = self.state(original, carrying, joints)
            valid = self.call(self.validity, GetStateValidity.Request(
                robot_state=candidate, group_name='fr3_arm'))
            contacts = [(c.contact_body_1, c.contact_body_2) for c in valid.contacts[:8]]
            details.append(f'diagnostic probe Z={probe.position.z:.6f} m; '
                           f'state valid={valid.valid}; collision contacts={contacts}')
            details.append('probe is diagnostic only; other limits/constraints may also stop the path')
        except Exception as exc:
            details.append(f'diagnostic unavailable: {exc}')
        return '; '.join(details)

    def plan(self, target, carrying=False, placed=False):
        with self.scoped_scene(carrying, placed) as original:
            state = self.state(original, carrying, self.task.joints)
            start = self.fk(state)
            xyz, _ = coordinates(start)
            if math.hypot(xyz[0]-target[0], xyz[1]-target[1]) > self.limits.target_xy_tolerance_m:
                raise RuntimeError('TCP is not aligned above/below target within 2 mm; replan approach.')
            distance = abs(target[2] - xyz[2])
            if distance < self.limits.line_tolerance_m:
                raise RuntimeError('Vertical segment is too short to validate; inspect the requested heights.')
            req = GetCartesianPath.Request()
            req.header.frame_id = BASE
            req.start_state = state
            req.group_name, req.link_name = 'fr3_arm', TCP
            req.max_step = self.limits.step_m
            req.jump_threshold = self.limits.relative_jump_factor
            req.revolute_jump_threshold = self.limits.max_joint_step_rad
            req.avoid_collisions = True
            if not hasattr(req, 'max_velocity_scaling_factor'):
                raise RuntimeError('This GetCartesianPath build lacks speed scaling; refusing to use full speed.')
            req.max_velocity_scaling_factor = float(self.task.c['velocity_scale'])
            req.max_acceleration_scaling_factor = float(self.task.c['acceleration_scale'])
            orientation = OrientationConstraint()
            orientation.header.frame_id, orientation.link_name = BASE, TCP
            orientation.orientation = copy.deepcopy(start.orientation)
            tol = math.radians(self.limits.orientation_tolerance_deg)
            orientation.absolute_x_axis_tolerance = tol
            orientation.absolute_y_axis_tolerance = tol
            orientation.absolute_z_axis_tolerance = tol
            orientation.weight = 1.
            req.path_constraints = Constraints(orientation_constraints=[orientation])
            n = math.ceil(distance / self.limits.step_m)
            for i in range(1, n + 1):
                p = copy.deepcopy(start)
                p.position.z = xyz[2] + (target[2] - xyz[2]) * i / n
                req.waypoints.append(p)
            response = self.call(self.cartesian, req, timeout=60)
            if (response.error_code.val != MoveItErrorCodes.SUCCESS or not math.isfinite(response.fraction)
                    or abs(response.fraction - 1.) > 1e-6):
                diagnostic = self.partial_path_diagnostic(response, original, carrying, start, target[2])
                raise RuntimeError(f'Incomplete Cartesian path ({response.fraction!r}); '
                                   f'no fallback or execution. {diagnostic}')
            trajectory = response.solution
            jt = trajectory.joint_trajectory
            if not jt.points or any(name not in self.task.joints
                    or abs(self.task.joints[name] - q) > 0.001
                    for name, q in zip(jt.joint_names, jt.points[0].positions)):
                raise RuntimeError('Cartesian service changed the requested starting joints.')
            def inspect(joints):
                if any(not self.bounds[name][0] <= value <= self.bounds[name][1]
                       for name, value in joints.items()):
                    raise RuntimeError('Timed Cartesian path exceeds robot joint limits.')
                merged = dict(self.task.joints)
                merged.update(joints)
                candidate = self.state(original, carrying, merged)
                valid = self.call(self.validity, GetStateValidity.Request(
                    robot_state=candidate, group_name='fr3_arm'))
                if not valid.valid and not is_support_touch(valid, carrying):
                    contacts = [(c.contact_body_1, c.contact_body_2, f'depth={c.depth*1000:.6f} mm')
                                for c in valid.contacts[:8]]
                    raise RuntimeError(f'Collision/invalid state along the timed Cartesian path: {contacts}.')
                return self.fk(candidate)
            count, xy, angle = validate_vertical(trajectory, start, target[2], inspect, self.limits)
            self.task.node.get_logger().info(
                f'Vertical path verified at {count} timed samples: '
                f'max lateral deviation {xy*1000:.3f} mm; orientation {angle:.3f} deg.')
            return copy.deepcopy(trajectory), copy.deepcopy(state)
