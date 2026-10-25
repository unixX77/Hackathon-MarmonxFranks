#!/usr/bin/env python3
"""Nine operator-gated stages; plan only unless --execute is supplied.

Install beside pick_place.py. Home is the measured starting arm joint pose.
Each stage is planned before its prompt; Enter executes those exact trajectories.
Lower/lift/retreat use verified Cartesian paths; transfers/home use normal planning.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject, CollisionObject, Constraints, DisplayTrajectory, JointConstraint,
    MoveItErrorCodes, PlanningSceneComponents,
)
from moveit_msgs.srv import GetPlanningScene, GetPositionFK, GetPositionIK

from pick_place import Task, box, block_center, block_object, rotate_vector, validate
from pick_place_cartesian import BASE, TCP, CartesianPlanner, coordinates

ARM_JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]
BLOCK = 'fr3_demo_block'


class MotionLog:
    """Append-only JSONL; flush each event so an aborted run retains evidence."""
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'stepwise_{time.time_ns()}.jsonl'
        self.stream = self.path.open('x', encoding='utf-8')

    def write(self, event, **data):
        self.stream.write(json.dumps(dict(event=event, unix_time_s=time.time(), **data),
                                     allow_nan=False) + '\n')
        self.stream.flush()

    def close(self):
        self.stream.close()


def finite_vector(value, length, label):
    if (not isinstance(value, list) or len(value) != length
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f'{label} must contain {length} finite numbers.')
    return list(value)


def load_camera_target(path):
    """Read manual optical-frame XYZ and the matching flange extrinsics."""
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        target = yaml.safe_load(stream)
    frame = target['camera_frame']
    if frame not in ('camera_color_optical_frame', 'camera_depth_optical_frame'):
        raise ValueError('camera_frame must select the color or depth optical frame.')
    xyz = finite_vector(target['xyz_m'], 3, 'xyz_m')
    if xyz[2] <= 0:
        raise ValueError('Optical-frame Z must be positive (metres in front of camera).')
    offset = finite_vector(target['point_to_block_center_world_m'], 3,
                           'point_to_block_center_world_m')
    extrinsics = Path(target['handeye_file'])
    if not extrinsics.is_absolute():
        extrinsics = path.parent / extrinsics
    with extrinsics.open(encoding='utf-8') as stream:
        calibration = yaml.safe_load(stream)
    if calibration['parent_frame'] != 'fr3_link8':
        raise ValueError('This manual input adapter requires fr3_link8 extrinsics.')
    transform = calibration[f'T_link8_{frame}']
    translation = finite_vector(transform['translation'], 3, 'handeye translation')
    q = finite_vector(transform['quaternion_xyzw'], 4, 'handeye quaternion')
    norm = math.sqrt(sum(v*v for v in q))
    if abs(norm - 1.) > 1e-3:
        raise ValueError('Handeye quaternion must be normalized.')
    return dict(camera_frame=frame, xyz_m=xyz, point_to_block_center_world_m=offset,
                translation=translation, quaternion_xyzw=[v/norm for v in q])


def camera_pick_config(config, target, flange_xyz, flange_q):
    """p_base = T_base_link8 T_link8_optical p_optical; use configured grip offset."""
    local = rotate_vector(target['quaternion_xyzw'], target['xyz_m'])
    local = [a+b for a, b in zip(target['translation'], local)]
    rotated = rotate_vector(flange_q, local)
    center = [a+b+c for a, b, c in zip(flange_xyz, rotated,
                                      target['point_to_block_center_world_m'])]
    result = copy.deepcopy(config)
    offset = config.get('block_center_offset_world', [0., 0., 0.])
    result['pick'] = [a-b for a, b in zip(center, offset)]
    # Destination stays at its configured/taught base-frame position.
    bottom = center[2] - config['block_size'][2] / 2
    table_top = config['table_center'][2] + config['table_size'][2] / 2
    if bottom < table_top - .001:
        raise ValueError(
            f'pick: block would intersect the table. '
            f'Converted block center ({BASE})={center}; TCP={result["pick"]}; '
            f'block bottom Z={bottom:.6f} m; table top Z={table_top:.6f} m; '
            f'penetration={(table_top-bottom)*1000:.1f} mm. '
            'Reacquire camera XYZ at the current stationary arm pose and check '
            'the mount extrinsics, optical frame, and measured table height.')
    return validate(result)


def load_config(path, poses_path=None, camera_mode=False):
    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    validate(config)
    if camera_mode:
        # Camera picks use task geometry and live arm joints, never old pick data.
        config['tcp_orientation_xyzw'] = [1., 0., 0., 0.]
        config.pop('approach_seed_joints', None)
        config['camera_ik'] = True
    elif poses_path is not None:
        taught = json.loads(Path(poses_path).read_text(encoding='utf-8'))
        if taught.get('frame_id') != BASE or taught.get('tcp_link') != TCP:
            raise ValueError('Taught poses must use fr3_link0 and fr3_hand_tcp.')
        pick, place = taught['poses']['pick'], taught['poses']['place']
        q, other = pick['quaternion_xyzw'], place['quaternion_xyzw']
        for quat in (q, other):
            validate(dict(config, tcp_orientation_xyzw=quat))
        q = [v / math.sqrt(sum(x*x for x in q)) for v in q]
        other = [v / math.sqrt(sum(x*x for x in other)) for v in other]
        if abs(sum(a*b for a, b in zip(q, other))) < 1 - 1e-8:
            raise ValueError('This cycle requires the same pick/place orientation.')
        config.update(pick=pick['xyz_m'], place=place['xyz_m'], tcp_orientation_xyzw=q)
        seed = pick.get('arm_joints_rad')
        if seed is not None:
            config['approach_seed_joints'] = seed
        # Upright block resting on the configured table, centred beneath the TCP.
        top = config['table_center'][2] + config['table_size'][2] / 2
        config['block_center_offset_world'] = [0., 0., top + config['block_size'][2]/2 - config['pick'][2]]
    for key, default, maximum in (('endpoint_z_tolerance_m', .002, .002),
                                  ('endpoint_lateral_tolerance_m', .002, .002),
                                  ('endpoint_orientation_tolerance_deg', .5, .5)):
        value = config.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= maximum):
            raise ValueError(f'{key} must be in (0, {maximum}].')
        config[key] = value
    clearance = config.get('place_release_clearance_m', .002)
    if (isinstance(clearance, bool) or not isinstance(clearance, (int, float))
            or not math.isfinite(clearance) or not 0 <= clearance <= .005):
        raise ValueError('place_release_clearance_m must be in [0, 0.005] metres.')
    config['place_release_clearance_m'] = clearance
    if 'approach_seed_joints' in config:
        seed = config['approach_seed_joints']
        if (not isinstance(seed, dict) or any(n not in seed or isinstance(seed[n], bool)
                or not isinstance(seed[n], (int, float)) or not math.isfinite(seed[n]) for n in ARM_JOINTS)):
            raise ValueError('approach_seed_joints must contain seven finite arm joint angles.')
    return validate(config)


class OperatorStop(Exception):
    pass


def confirm(label, execute):
    # Discard keys pressed during planning/motion, so one old Enter cannot
    # accidentally authorize the next stage. Non-terminal input is refused.
    if not sys.stdin.isatty():
        raise OperatorStop('An interactive terminal is required.')
    import termios
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    verb = 'EXECUTE' if execute else 'accept this planning-only stage'
    while True:
        answer = input(f'\n{label}\nEnter to {verb}; q to stop: ').strip().lower()
        if answer == 'q':
            raise OperatorStop('Stopped by operator.')
        if answer == '':
            return
        print('Press Enter or type q.')


def check_start(positions, trajectory, tolerance=0.01):
    jt = trajectory.joint_trajectory
    if not jt.points or not jt.joint_names:
        raise RuntimeError('Planner returned an empty trajectory.')
    first = jt.points[0].positions
    if len(first) != len(jt.joint_names):
        raise RuntimeError('Invalid trajectory start point.')
    if any(name not in positions or not math.isfinite(positions[name])
           or abs(positions[name] - q) > tolerance
           for name, q in zip(jt.joint_names, first)):
        raise RuntimeError('Arm no longer matches the approved trajectory start. Stop and replan.')


class StepwiseTask(Task):
    def __init__(self, node, config, execute, motion_log=None):
        self.motion_log = motion_log
        super().__init__(node, config, execute)
        self.hardware = execute
        self.latest = {}
        self.latest_stamp = None
        self.received = 0.0
        self.commands = []
        self.vertical_starts = {}
        self.label = ''
        self.preview = node.create_publisher(DisplayTrajectory, 'display_planned_path', 10)
        self.subscription = node.create_subscription(
            JointState, 'joint_states', self.on_joints, qos_profile_sensor_data)
        self.vertical = CartesianPlanner(self)
        self.executor = None
        if execute:
            self.executor = ActionClient(node, ExecuteTrajectory, 'execute_trajectory')
            if not self.executor.wait_for_server(timeout_sec=10):
                raise RuntimeError('MoveIt execute_trajectory action unavailable.')
        self.joints = self.settled_snapshot()
        self.home = {name: self.joints[name] for name in ARM_JOINTS}
        self.log_event('initial_state', joints_rad=self.joints, config=self.c)

    def log_event(self, event, **data):
        log = getattr(self, 'motion_log', None)
        if log is not None:
            log.write(event, **data)

    def use_camera_target(self, target):
        # Resolve once BEFORE stage 1, from the stationary observation posture.
        # Subsequent arm motion must not move the stored world-frame target.
        joints = self.settled_snapshot()
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ROBOT_STATE
        original = self.vertical.call(self.scene_client, req).scene.robot_state
        state = self.vertical.state(original, False, joints)
        fk = GetPositionFK.Request(robot_state=state, fk_link_names=['fr3_link8'])
        fk.header.frame_id = BASE
        response = self.vertical.call(self.vertical.fk_client, fk)
        if (response.error_code.val != MoveItErrorCodes.SUCCESS
                or response.fk_link_names != ['fr3_link8'] or len(response.pose_stamped) != 1
                or response.pose_stamped[0].header.frame_id != BASE):
            raise RuntimeError('Flange FK failed or returned an unexpected frame.')
        xyz, q = coordinates(response.pose_stamped[0].pose)
        self.log_event('camera_observation', target=target, joints_rad=joints,
                       flange_xyz_m=xyz, flange_quaternion_xyzw=q)
        print(f"Camera point ({target['camera_frame']}, metres): {target['xyz_m']}", flush=True)
        print(f'Observation flange ({BASE}): xyz_m={xyz}, quaternion_xyzw={q}', flush=True)
        self.c = camera_pick_config(self.c, target, xyz, q)
        self.log_event('resolved_camera_config', config=self.c)
        self.joints = joints
        self.home = {name: joints[name] for name in ARM_JOINTS}
        print(f'Block center ({BASE}, metres): {block_center(self.c)}')
        print(f"Grasp TCP ({BASE}, metres): {self.c['pick']}")
        print(f"Place TCP unchanged ({BASE}, metres): {self.c['place']}")
        print('Camera extrinsics are CAD estimates; verify mount, optical frame and point meaning.')

    def on_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(name in values and math.isfinite(values[name]) for name in ARM_JOINTS):
            self.latest = {name: q for name, q in values.items() if math.isfinite(q)}
            self.latest_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.received = time.monotonic()

    def snapshot(self, timeout=5.0):
        # Collect after waiting at the prompt; never use an old cached pose.
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            now = self.node.get_clock().now().nanoseconds * 1e-9
            if (self.received >= started and self.latest_stamp is not None
                    and -0.05 <= now - self.latest_stamp <= 0.5):
                return dict(self.latest)
        raise RuntimeError('No fresh timestamped arm joint states; inspect the robot connection.')

    def settled_snapshot(self):
        started = time.monotonic()
        stable_since = None
        anchor = None
        while time.monotonic() - started < 8.0:
            q = self.snapshot()
            if anchor is None or any(abs(q[n] - anchor[n]) > 0.002 for n in ARM_JOINTS):
                anchor, stable_since = q, time.monotonic()
            if time.monotonic() - stable_since >= 1.0:
                return q
        raise RuntimeError('Arm did not settle; inspect the robot before continuing.')

    def action(self, client, goal):
        if client is not self.arm:
            return super().action(client, goal)
        # All arm requests are plan-only, including in hardware mode.
        goal.planning_options.plan_only = True
        scene = goal.planning_options.planning_scene_diff
        carrying = any(o.object.id == BLOCK and o.object.operation == CollisionObject.ADD
                       for o in scene.robot_state.attached_collision_objects)
        if carrying:
            # The explicit attached-object ADD transfers any same-ID world object.
            # A subsequent world REMOVE would fail because it is already absent.
            scene.world.collision_objects = [o for o in scene.world.collision_objects if o.id != BLOCK]
        else:
            removed = AttachedCollisionObject(link_name='fr3_hand_tcp')
            removed.object.id = BLOCK
            removed.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects.append(copy.deepcopy(removed))
            goal.request.start_state.attached_collision_objects.append(removed)
        if ((self.c.get('camera_ik') or self.c.get('approach_seed_joints'))
                and goal.request.goal_constraints[0].position_constraints):
            self.seed_pose_goal(goal, carrying)
        result = super().action(client, goal)
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'{self.label}: planning failed ({result.error_code.val}).')
        trajectory = result.planned_trajectory
        jt = trajectory.joint_trajectory
        if not jt.points or not set(ARM_JOINTS).issubset(jt.joint_names):
            raise RuntimeError(f'{self.label}: incomplete/empty arm trajectory.')
        if not any(kind == 'arm' for kind, _, _ in self.commands):
            self.preview_start = copy.deepcopy(result.trajectory_start)
        self.commands.append(('arm', self.label, copy.deepcopy(trajectory)))
        return result

    def seed_pose_goal(self, goal, carrying):
        """Choose pose-goal IK near a known posture instead of unrestricted sampling."""
        constraints = goal.request.goal_constraints[0]
        target = constraints.position_constraints[0].constraint_region.primitive_poses[0]
        pose = copy.deepcopy(target)
        pose.orientation = copy.deepcopy(constraints.orientation_constraints[0].orientation)
        camera_mode = self.c.get('camera_ik', False)
        reference = self.joints if camera_mode or carrying else self.c['approach_seed_joints']
        seed_joints = dict(self.joints)
        seed_joints.update({n: reference[n] for n in ARM_JOINTS})
        client = self.node.create_client(GetPositionIK, 'compute_ik')
        if not client.wait_for_service(timeout_sec=10):
            raise RuntimeError('MoveIt compute_ik unavailable for posture-preserving pose goals.')
        with self.vertical.scoped_scene(carrying, False) as original:
            req = GetPositionIK.Request()
            req.ik_request.group_name, req.ik_request.ik_link_name = 'fr3_arm', TCP
            req.ik_request.robot_state = self.vertical.state(original, carrying, seed_joints)
            req.ik_request.pose_stamped.header.frame_id = BASE
            req.ik_request.pose_stamped.pose = pose
            req.ik_request.avoid_collisions = True
            req.ik_request.timeout.sec = 2
            if camera_mode:
                # Current joints guide redundant-arm IK without restricting the
                # target to a neighborhood of any saved or starting posture.
                candidates = []
                for _ in range(5):
                    result = self.vertical.call(client, req, timeout=5)
                    if result.error_code.val != MoveItErrorCodes.SUCCESS:
                        continue
                    solution = dict(zip(result.solution.joint_state.name,
                                        result.solution.joint_state.position))
                    if any(n not in solution or not math.isfinite(solution[n]) for n in ARM_JOINTS):
                        raise RuntimeError('IK returned incomplete/non-finite arm joints.')
                    distance = max(abs(solution[n] - reference[n]) for n in ARM_JOINTS)
                    candidates.append((distance, solution))
                if not candidates:
                    raise RuntimeError('Camera target: IK found no collision-free solution after 5 attempts. '
                                       'No old taught-posture restriction was applied; '
                                       'check reachability, orientation and scene geometry.')
                change, solved = min(candidates, key=lambda candidate: candidate[0])
                self.log_event('camera_ik', reference_joints_rad=reference,
                               candidates=[dict(max_joint_change_rad=d, joints_rad=j) for d, j in candidates],
                               selected_joints_rad=solved)
                self.node.get_logger().info(
                    f'Camera IK seeded from current joints; max endpoint joint change {change:.3f} rad. '
                    'No taught-posture bound applied; review the planned path.')
                constraints.joint_constraints = [JointConstraint(
                    joint_name=n, position=solved[n], tolerance_above=.001,
                    tolerance_below=.001, weight=1.) for n in ARM_JOINTS]
                return
            # Ask IK to respect the same posture bound that we independently
            # verify below. A redundant arm can return different IK branches;
            # one unsuitable solution does not prove no nearby solution exists.
            req.ik_request.constraints.joint_constraints = [JointConstraint(
                joint_name=n, position=reference[n], tolerance_above=.75,
                tolerance_below=.75, weight=1.) for n in ARM_JOINTS]
            nearest = None
            for _ in range(5):
                result = self.vertical.call(client, req, timeout=5)
                if result.error_code.val != MoveItErrorCodes.SUCCESS:
                    continue
                solved = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
                if any(n not in solved or not math.isfinite(solved[n]) for n in ARM_JOINTS):
                    raise RuntimeError('IK returned incomplete/non-finite arm joints.')
                worst = max(ARM_JOINTS, key=lambda n: abs(solved[n] - reference[n]))
                change = abs(solved[worst] - reference[worst])
                if nearest is None or change < nearest[0]:
                    nearest = (change, worst)
                if change <= .75:
                    break
            else:
                detail = (f' Closest returned solution changes {nearest[1]} by {nearest[0]:.3f} rad.'
                          if nearest is not None else '')
                raise RuntimeError('No collision-free IK within 0.75 rad of the reference posture '
                                   'after 5 attempts.' + detail +
                                   ' Re-teach a suitable approach posture for this target; '
                                   'this is not a singularity test.')
        constraints.joint_constraints = [JointConstraint(
            joint_name=n, position=solved[n], tolerance_above=.001,
            tolerance_below=.001, weight=1.) for n in ARM_JOINTS]
        self.node.get_logger().info(f'Pose-goal IK anchored near reference posture; max joint change {change:.3f} rad.')

    def motion(self, label, xyz, carrying=False, placed=False):
        self.label = label
        self.log_event('pose_goal', label=label, frame_id=BASE, tcp_link=TCP,
                       xyz_m=list(xyz), quaternion_xyzw=self.c.get('tcp_orientation_xyzw'),
                       carrying=carrying)
        super().motion(label, xyz, carrying=carrying, placed=placed)

    def hand(self, close=False):
        # During stage preparation, model finger width and queue the action.
        super().hand(close=close)
        self.commands.append(('hand', 'Grasp' if close else 'Open hand', close))

    def cartesian_motion(self, label, xyz, carrying=False, placed=False):
        self.log_event('cartesian_goal', label=label, frame_id=BASE, tcp_link=TCP,
                       xyz_m=list(xyz), quaternion_xyzw=self.c.get('tcp_orientation_xyzw'),
                       carrying=carrying)
        self.node.get_logger().info(f'{label}: vertical Cartesian planning to {xyz}')
        trajectory, start_state = self.vertical.plan(xyz, carrying=carrying, placed=placed)
        if not any(kind == 'arm' for kind, _, _ in self.commands):
            self.preview_start = copy.deepcopy(start_state)
        self.commands.append(('arm', label, trajectory))
        self.vertical_starts[label] = copy.deepcopy(start_state)
        jt = trajectory.joint_trajectory
        self.joints.update(zip(jt.joint_names, jt.points[-1].positions))

    def plan_home(self):
        self.label = 'Return to starting arm joint pose (home)'
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = 'fr3_arm'
        req.allowed_planning_time = 10.0
        req.num_planning_attempts = 5
        req.max_velocity_scaling_factor = float(self.c['velocity_scale'])
        req.max_acceleration_scaling_factor = float(self.c['acceleration_scale'])
        req.start_state.is_diff = True
        req.start_state.joint_state.name = list(self.joints)
        req.start_state.joint_state.position = list(self.joints.values())
        req.goal_constraints = [Constraints(joint_constraints=[JointConstraint(
            joint_name=name, position=q, tolerance_above=0.005,
            tolerance_below=0.005, weight=1.0) for name, q in self.home.items()])]
        scene = goal.planning_options.planning_scene_diff
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = copy.deepcopy(self.acm)
        scene.world.collision_objects = [
            box('fr3_demo_table', self.c['table_size'], self.c['table_center']),
            block_object(self.c, placed=True),
        ]
        result = self.action(self.arm, goal)
        jt = result.planned_trajectory.joint_trajectory
        self.joints.update(zip(jt.joint_names, jt.points[-1].positions))

    def placement_target(self):
        """Place the held block above its destination using the captured grip."""
        place = list(self.c['place'])
        if 'attached_block_pose' in self.c:
            q = self.c['carrying_orientation_xyzw']
            offset = rotate_vector(q, self.c['attached_block_pose']['xyz_m'])
            place = [a-b for a, b in zip(block_center(self.c, placed=True), offset)]
        place[2] += self.c.get('place_release_clearance_m', 0.)
        return place

    def prepare_stage(self, stage):
        pick, place = self.c['pick'], self.placement_target()
        above_pick, above_place = list(pick), list(place)
        above_pick[2] += self.c['clearance']
        above_place[2] += self.c['clearance']
        if stage == 1:
            self.hand()
            self.motion('Approach above brick', above_pick)
        elif stage == 2:
            self.cartesian_motion('Lower vertically to grasp position', pick)
        elif stage == 3:
            self.hand(close=True)
            if not self.hardware:
                self.capture_grasp_transform(dict(self.joints))
        elif stage == 4:
            self.cartesian_motion('Lift brick vertically', above_pick, carrying=True)
        elif stage == 5:
            self.motion('Transfer above destination', above_place, carrying=True)
        elif stage == 6:
            self.cartesian_motion('Lower vertically to placement position', place, carrying=True)
        elif stage == 7:
            self.hand()
        elif stage == 8:
            self.cartesian_motion('Retreat vertically above placed brick', above_place, placed=True)
        elif stage == 9:
            self.plan_home()
        else:
            raise ValueError('Unknown stage')

    def check_endpoint(self, trajectory, joints):
        """Report tracking error with the same teaching acceptance as the jog."""
        expected_state = copy.deepcopy(self.preview_start)
        actual_state = copy.deepcopy(self.preview_start)
        jt = trajectory.joint_trajectory
        for state, values in ((expected_state, dict(zip(jt.joint_names, jt.points[-1].positions))),
                              (actual_state, joints)):
            merged = dict(zip(state.joint_state.name, state.joint_state.position))
            merged.update(values)
            state.joint_state.name = list(merged)
            state.joint_state.position = list(merged.values())
            state.joint_state.velocity = []
            state.joint_state.effort = []
            state.joint_state.header.stamp.sec = state.joint_state.header.stamp.nanosec = 0
            state.is_diff = False
        expected, eq = coordinates(self.vertical.fk(expected_state))
        actual, aq = coordinates(self.vertical.fk(actual_state))
        lateral = math.hypot(actual[0]-expected[0], actual[1]-expected[1])
        dz = actual[2]-expected[2]
        angle = math.degrees(2*math.acos(min(1., abs(sum(a*b for a, b in zip(eq, aq))))))
        report = f'Endpoint error: lateral={lateral*1000:.3f} mm, Z={dz*1000:+.3f} mm, orientation={angle:.3f} deg.'
        planned_joints = dict(zip(jt.joint_names, jt.points[-1].positions))
        joint_error = {n: joints[n]-planned_joints[n] for n in ARM_JOINTS}
        limits = dict(lateral_m=self.c.get('endpoint_lateral_tolerance_m', .002),
                      z_m=self.c.get('endpoint_z_tolerance_m', .002),
                      orientation_deg=self.c.get('endpoint_orientation_tolerance_deg', .5),
                      max_joint_rad=.01)
        accepted = (lateral <= limits['lateral_m'] and abs(dz) <= limits['z_m']
                    and angle <= limits['orientation_deg']
                    and max(abs(v) for v in joint_error.values()) <= limits['max_joint_rad'])
        self.log_event('endpoint', label=getattr(self, 'active_motion_label', self.label),
                       frame_id=BASE, tcp_link=TCP,
                       planned=dict(joints_rad=planned_joints, xyz_m=expected, quaternion_xyzw=eq),
                       measured=dict(joints_rad=joints, xyz_m=actual, quaternion_xyzw=aq,
                                     joint_state_stamp_s=getattr(self, 'latest_stamp', None)),
                       error=dict(joints_rad=joint_error, xyz_m=[a-b for a, b in zip(actual, expected)],
                                  lateral_m=lateral, z_m=dz, orientation_deg=angle),
                       tolerances=limits, accepted=accepted)
        self.node.get_logger().info(report)
        if (lateral > self.c.get('endpoint_lateral_tolerance_m', .002)
                or abs(dz) > self.c.get('endpoint_z_tolerance_m', .002)
                or angle > self.c.get('endpoint_orientation_tolerance_deg', .5)):
            raise RuntimeError(report + ' Outside endpoint tolerance; no automatic correction.')
        if abs(dz) > .001 or lateral > .001:
            self.node.get_logger().warning('Accepted teaching endpoint offset above 1 mm. ' + report)

    def capture_grasp_transform(self, joints=None):
        # The actual grip may differ slightly from the taught target. Attach the
        # fixed world block at the measured TCP, rather than shifting it into the table.
        # Preview uses the virtual descent endpoint; hardware uses measured joints
        # only after the grasp action succeeds.
        if joints is None:
            joints = self.settled_snapshot()
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.ROBOT_STATE
                                     | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        original = self.vertical.call(self.scene_client, req).scene.robot_state
        state = self.vertical.state(original, False, joints)
        xyz, q = coordinates(self.vertical.fk(state))
        inverse = [-q[0], -q[1], -q[2], q[3]]
        offset = [a-b for a, b in zip(block_center(self.c), xyz)]
        self.c['attached_block_pose'] = {
            'xyz_m': rotate_vector(inverse, offset), 'quaternion_xyzw': inverse}
        self.c['carrying_orientation_xyzw'] = q

    def execute_commands(self):
        for kind, label, payload in self.commands:
            self.node.get_logger().info(label)
            if kind == 'hand':
                # Bypass the queue-building override; this is the real action.
                Task.hand(self, close=payload)
                self.settled_snapshot()
                if payload:
                    self.capture_grasp_transform()
            else:
                live = self.settled_snapshot()
                if label in self.vertical_starts:
                    check_start(live, payload, tolerance=0.001)
                    self.vertical.check_execution_start(live, self.vertical_starts[label])
                else:
                    check_start(live, payload)
                # Humble ExecuteTrajectory has no controllers request field;
                # MoveIt selects the configured controller for these joints.
                goal = ExecuteTrajectory.Goal(trajectory=payload)
                self.active_motion_label = label
                jt = payload.joint_trajectory
                self.log_event('execution_start', label=label, measured_start_joints_rad=live,
                               joint_names=list(jt.joint_names),
                               trajectory=[dict(time_s=p.time_from_start.sec+p.time_from_start.nanosec*1e-9,
                                                positions_rad=list(p.positions),
                                                velocities_rad_s=list(p.velocities),
                                                accelerations_rad_s2=list(p.accelerations)) for p in jt.points])
                result = self.action(self.executor, goal)
                self.log_event('execution_result', label=label, moveit_error_code=result.error_code.val)
                if result.error_code.val != MoveItErrorCodes.SUCCESS:
                    raise RuntimeError(f'{label}: execution failed ({result.error_code.val}).')
                q = self.settled_snapshot()
                jt = payload.joint_trajectory
                # Log endpoint evidence before either joint or TCP checks abort.
                self.check_endpoint(payload, q)
                if any(abs(q[n] - value) > 0.01
                       for n, value in zip(jt.joint_names, jt.points[-1].positions)):
                    raise RuntimeError(f'{label}: measured endpoint differs from planned endpoint.')

    def run(self, stop_after=9):
        labels = [
            '1/9 Open empty hand and approach above brick',
            '2/9 Lower vertically to grasp position',
            '3/9 Grasp brick',
            '4/9 Lift brick vertically',
            '5/9 Transfer above destination',
            '6/9 Lower vertically to placement position',
            '7/9 Release brick',
            '8/9 Retreat vertically',
            '9/9 Return to starting arm joint pose (home)',
        ]
        print('HARDWARE EXECUTION' if self.hardware else 'PLANNING ONLY: no arm or gripper execution')
        print('Start with an empty hand. Home is the starting joint pose:', self.home)
        print('No camera collision model is required; existing scene attachments remain checked.')
        print('Review each stage in RViz before Enter. Lower/lift/retreat are checked vertical paths; '
              'transfer/home paths can curve.')
        for stage, label in enumerate(labels, 1):
            if self.hardware:
                self.joints = self.settled_snapshot()
                expected_start = dict(self.joints)
            self.commands = []
            self.vertical_starts = {}
            # Only preview mode advances virtually; hardware plans from measured joints.
            self.execute = False
            self.prepare_stage(stage)
            paths = [payload for kind, _, payload in self.commands if kind == 'arm']
            if paths:
                # Publish the segment for this approval step.
                self.preview.publish(DisplayTrajectory(
                    trajectory_start=self.preview_start, trajectory=paths))
            confirm(label, self.hardware)
            if self.hardware:
                actual = self.settled_snapshot()
                if any(abs(actual[n] - expected_start[n]) > 0.01 for n in ARM_JOINTS):
                    raise RuntimeError('Arm moved while waiting for approval; stop and replan.')
                self.execute = True
                self.execute_commands()
            print(f'Stage {stage}/9 complete.')
            if stage == stop_after:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execute', action='store_true', help='Enable physical arm AND gripper actions')
    parser.add_argument('--poses', type=Path,
                        help='Optional taught TCP JSON; overrides pick/place XYZ and orientation from config')
    parser.add_argument('--camera-target', type=Path,
                        help='Manual optical-frame XYZ; fixed [1,0,0,0] orientation, current-joint IK; ignores --poses')
    parser.add_argument('--stop-after', type=int, choices=range(1, 10), default=9,
                        help='Stop after this stage; use 3 for approach, lower and grasp only')
    parser.add_argument('--resolve-only', action='store_true',
                        help='With --camera-target, print converted targets and exit without planning/motion')
    parser.add_argument('--log-dir', type=Path, default=Path('data/motion_logs'),
                        help='JSONL motion logs (default: data/motion_logs relative to working directory)')
    args, ros_args = parser.parse_known_args()
    if args.resolve_only and (args.camera_target is None or args.execute):
        parser.error('--resolve-only requires --camera-target and cannot use --execute.')
    config = load_config(args.config, args.poses, camera_mode=args.camera_target is not None)
    camera_target = load_camera_target(args.camera_target) if args.camera_target else None
    if max(config['velocity_scale'], config['acceleration_scale']) > 0.05:
        parser.error('For this staged script use velocity/acceleration scaling at most 0.05.')
    rclpy.init(args=ros_args)
    node = rclpy.create_node('fr3_pick_place_stepwise')
    motion_log = None
    try:
        motion_log = MotionLog(args.log_dir)
        print(f'Motion log: {motion_log.path.resolve()}', flush=True)
        motion_log.write('run_start', execute=args.execute, stop_after=args.stop_after,
                         config_path=str(args.config), poses_path=str(args.poses) if args.poses else None,
                         camera_target_path=str(args.camera_target) if args.camera_target else None,
                         resolve_only=args.resolve_only, config=config)
        task = StepwiseTask(node, config, args.execute, motion_log=motion_log)
        if camera_target is not None:
            task.use_camera_target(camera_target)
        if not args.resolve_only:
            task.run(stop_after=args.stop_after)
            node.get_logger().info(f'Completed through stage {args.stop_after}.')
        motion_log.write('run_end', status='completed')
    except (OperatorStop, EOFError, KeyboardInterrupt):
        if motion_log is not None:
            motion_log.write('run_end', status='operator_stopped')
        node.get_logger().warning('Stopped. No automatic gripper release or return home. Inspect the robot.')
    except Exception as exc:
        if motion_log is not None:
            motion_log.write('run_end', status='aborted', error=str(exc))
        node.get_logger().error(f'Aborted: {exc}. No automatic release or recovery.')
        return 1
    finally:
        if motion_log is not None:
            motion_log.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
