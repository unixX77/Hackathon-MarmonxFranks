#!/usr/bin/python3 -s
"""Eye-in-hand calibration: board, collect, solve, and optional taught-pose replay.

T_A_B maps coordinates expressed in B into A. No motion occurs without
`replay --execute`. Collection and solving never instantiate motion clients.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import shlex
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]


def matrix(rotation, translation):
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = np.asarray(translation).reshape(3)
    return t


def rigid(t):
    t = np.asarray(t, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    return t


def make_board(c):
    b = c['board']
    if b.get('type') == 'charuco':
        for key in ('square_length', 'marker_length'):
            value = b.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'board.{key}: enter a positive measured dimension in metres')
        if b['marker_length'] >= b['square_length']:
            raise ValueError('board.marker_length must be smaller than board.square_length')
        if b['rows'] < 3 or b['columns'] < 3:
            raise ValueError('ChArUco board needs at least three rows and columns')
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b['dictionary']))
        if hasattr(cv2.aruco, 'CharucoBoard_create'):
            board = cv2.aruco.CharucoBoard_create(
                b['columns'], b['rows'], b['square_length'], b['marker_length'], dictionary)
        else:
            board = cv2.aruco.CharucoBoard(
                (b['columns'], b['rows']), b['square_length'], b['marker_length'], dictionary)
        return dictionary, board
    for key in ('marker_length', 'marker_separation'):
        value = b.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'board.{key}: enter a positive measured dimension in metres')
    if b['rows'] < 2 or b['columns'] < 2:
        raise ValueError('Board must have at least two rows and columns')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b['dictionary']))
    if hasattr(cv2.aruco, 'GridBoard_create'):
        board = cv2.aruco.GridBoard_create(
            b['columns'], b['rows'], b['marker_length'], b['marker_separation'], dictionary)
    else:
        board = cv2.aruco.GridBoard(
            (b['columns'], b['rows']), b['marker_length'], b['marker_separation'], dictionary)
    return dictionary, board


def detect_target(image, info, c):
    """Return T_camera_target, rejecting poor/ambiguous planar solutions."""
    if info['distortion_model'] not in ('plumb_bob', 'rational_polynomial'):
        raise ValueError('Use raw color image/CameraInfo with a supported distortion model')
    k = np.asarray(info['k'], dtype=float).reshape(3, 3)
    d = np.asarray(info['d'], dtype=float)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isfinite(k).all():
        raise ValueError('Camera intrinsics are invalid')
    dictionary, board = make_board(c)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
    if ids is None:
        raise ValueError('No board markers visible')
    if c['board'].get('type') == 'charuco':
        if hasattr(cv2.aruco, 'interpolateCornersCharuco'):
            count, found, found_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, board, cameraMatrix=k, distCoeffs=d)
        else:
            found, found_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
            count = 0 if found is None else len(found)
        need = c.get('min_charuco_corners', 10)
        if found is None or count < need:
            raise ValueError(f'Need at least {need} ChArUco corners (found {count})')
        corner_points = board.chessboardCorners if hasattr(board, 'chessboardCorners') \
            else board.getChessboardCorners()
        obj = np.asarray(corner_points)[found_ids.flatten()].reshape(-1, 3).astype(np.float64)
        # Optional per-axis correction for a print that is not exactly square.
        obj[:, 0] *= c['board'].get('scale_x', 1.0)
        obj[:, 1] *= c['board'].get('scale_y', 1.0)
        pixels = found.reshape(-1, 2).astype(np.float64)
        matches = [None] * int(count)
    else:
        board_ids = board.getIds() if hasattr(board, 'getIds') else board.ids
        board_points = board.getObjPoints() if hasattr(board, 'getObjPoints') else board.objPoints
        mapping = {int(i): p for i, p in zip(board_ids.flatten(), board_points)}
        matches = [(mapping[int(i)], p.reshape(4, 2)) for i, p in zip(ids.flatten(), corners)
                   if int(i) in mapping]
        if len(matches) < c['min_markers']:
            raise ValueError(f'Need at least {c["min_markers"]} board markers')
        obj = np.concatenate([p[0] for p in matches]).astype(np.float64)
        pixels = np.concatenate([p[1] for p in matches]).astype(np.float64)
    result = cv2.solvePnPGeneric(obj, pixels, k, d, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for r, t in zip(result[1], result[2]):
        rot = cv2.Rodrigues(r)[0]
        if np.min((rot @ obj.T + t.reshape(3, 1))[2]) <= 0:
            continue
        predicted = cv2.projectPoints(obj, r, t, k, d)[0].reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((predicted - pixels) ** 2, axis=1))))
        candidates.append((error, matrix(rot, t)))
    candidates.sort(key=lambda x: x[0])
    if not candidates or candidates[0][0] > c['max_reprojection_px']:
        raise ValueError('Board reprojection error too high or board behind camera')
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < c['min_ambiguity_gap_px']:
        raise ValueError('Planar board pose ambiguous; tilt camera slightly and capture again')
    return candidates[0][1], candidates[0][0], len(matches)


def fit(samples):
    g = [rigid(s['T_base_ee']) for s in samples]
    ct = [rigid(s['T_camera_target']) for s in samples]
    r, t = cv2.calibrateHandEye(
        [a[:3, :3] for a in g], [a[:3, 3] for a in g],
        [a[:3, :3] for a in ct], [a[:3, 3] for a in ct],
        method=cv2.CALIB_HAND_EYE_PARK)
    return rigid(matrix(r, t))


def excitation(samples, c):
    rotations = [rigid(s['T_base_ee'])[:3, :3] for s in samples]
    vectors = np.array([Rotation.from_matrix(rotations[0].T @ r).as_rotvec()
                        for r in rotations[1:]])
    singular = np.linalg.svd(vectors, compute_uv=False)
    span = np.rad2deg(np.max(np.linalg.norm(vectors, axis=1)))
    if (span < c['min_rotation_span_deg'] or singular[1] / math.sqrt(len(vectors)) < 0.05
            or singular[1] < 0.15 * singular[0]):
        raise ValueError('Insufficient rotation about different axes; translations alone cannot calibrate')
    return {'rotation_span_deg': float(span), 'rotation_singular_values': singular.tolist()}


def consistency(samples, x, reference=None):
    targets = [rigid(s['T_base_ee']) @ x @ rigid(s['T_camera_target']) for s in samples]
    if reference is None:
        reference = matrix(Rotation.from_matrix(np.array([t[:3, :3] for t in targets])).mean().as_matrix(),
                           np.mean([t[:3, 3] for t in targets], axis=0))
    distances = [np.linalg.norm(t[:3, 3] - reference[:3, 3]) for t in targets]
    angles = [np.rad2deg(Rotation.from_matrix(reference[:3, :3].T @ t[:3, :3]).magnitude())
              for t in targets]
    return {'translation_rms_m': float(np.sqrt(np.mean(np.square(distances)))),
            'translation_max_m': float(max(distances)),
            'rotation_rms_deg': float(np.sqrt(np.mean(np.square(angles)))),
            'rotation_max_deg': float(max(angles))}, reference


def solve(data, c):
    samples = data['samples']
    if len(samples) < c['min_samples'] or len(samples) < 8:
        raise ValueError(f'Collect at least {max(8, c["min_samples"])} accepted poses')
    coverage = excitation(samples, c)
    # Hold out every fifth pose. Do not use these in the validation fit.
    train = [s for i, s in enumerate(samples) if i % 5 != 0]
    test = [s for i, s in enumerate(samples) if i % 5 == 0]
    excitation(train, c)
    candidate = fit(train)
    _, reference = consistency(train, candidate)
    heldout, _ = consistency(test, candidate, reference)
    x = fit(samples)
    all_stats, board_reference = consistency(samples, x)
    passed = all(stats['translation_max_m'] <= c['max_consistency_translation_m']
                 and stats['rotation_max_deg'] <= c['max_consistency_rotation_deg']
                 for stats in (heldout, all_stats))
    result = {'status': 'passed_quality_gates' if passed else 'REJECTED',
              'parent_frame': c['ee_frame'], 'child_frame': data['camera_frame'],
              'convention': 'p_ee = T_ee_camera @ p_camera; metres; quaternion xyzw',
              'T_ee_camera': x.tolist(), 'translation_m': x[:3, 3].tolist(),
              'quaternion_xyzw': Rotation.from_matrix(x[:3, :3]).as_quat().tolist(),
              'samples': len(samples), 'coverage': coverage,
              'base_frame': c['base_frame'], 'board': c['board'],
              'T_base_target': board_reference.tolist(),
              'held_out': heldout, 'all_samples': all_stats,
              'note': 'Consistency is not an absolute accuracy or robot safety guarantee.'}
    if data.get('T_mount_camera') is not None:
        mount = x @ np.linalg.inv(rigid(data['T_mount_camera']))
        result['mount_frame'] = c['camera_mount_frame']
        result['T_ee_mount'] = mount.tolist()
        if passed:
            q = Rotation.from_matrix(mount[:3, :3]).as_quat()
            parts = ['ros2', 'run', 'tf2_ros', 'static_transform_publisher']
            for key, value in zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), [*mount[:3, 3], *q]):
                parts += [f'--{key}', str(float(value))]
            parts += ['--frame-id', c['ee_frame'], '--child-frame-id', c['camera_mount_frame']]
            result['mount_tf_command_after_validation'] = shlex.join(parts)
    return result


def validate_result(data, result, c):
    if result['status'] != 'passed_quality_gates':
        raise ValueError('Cannot validate a rejected calibration')
    if (result['parent_frame'] != c['ee_frame'] or result['child_frame'] != data['camera_frame']
            or result['base_frame'] != c['base_frame'] or result['board'] != c['board']):
        raise ValueError('Calibration and validation dataset frames/board differ')
    if len(data['samples']) < 5:
        raise ValueError('Collect at least five NEW poses for independent validation')
    excitation(data['samples'], c)
    stats, _ = consistency(data['samples'], rigid(result['T_ee_camera']), rigid(result['T_base_target']))
    passed = (stats['translation_max_m'] <= c['max_consistency_translation_m'] and
              stats['rotation_max_deg'] <= c['max_consistency_rotation_deg'])
    return {'status': 'passed_quality_gates' if passed else 'REJECTED',
            'independent_validation': stats,
            'note': 'Uses the original transform and board reference WITHOUT refitting.'}


def save(path, data):
    """Atomic update of our own dataset; new commands refuse existing outputs."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
    temporary.replace(target)


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Collector:
    def __init__(self, c):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.duration import Duration
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from cv_bridge import CvBridge
        from tf2_ros import Buffer, TransformListener
        self.c = c
        self.node = rclpy.create_node('fr3_handeye')
        self.bridge = CvBridge()
        self.buffer = Buffer(cache_time=Duration(seconds=30))
        self.listener = TransformListener(self.buffer, self.node)
        self.image = self.info = None
        self.joints = deque(maxlen=5000)
        self.subscriptions = [
            self.node.create_subscription(Image, c['image_topic'], self.on_image, qos_profile_sensor_data),
            self.node.create_subscription(CameraInfo, c['camera_info_topic'], self.on_info, qos_profile_sensor_data),
            self.node.create_subscription(JointState, c['joint_topic'], self.on_joints, qos_profile_sensor_data)]

    def on_image(self, msg):
        self.image = msg

    def on_info(self, msg):
        self.info = msg

    def on_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(j in values for j in JOINTS):
            self.joints.append((stamp_seconds(msg.header.stamp), [values[j] for j in JOINTS]))

    def spin(self, seconds):
        import rclpy
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0., end - time.monotonic())))

    def transform(self, parent, child, stamp):
        from rclpy.time import Time
        msg = self.buffer.lookup_transform(parent, child, Time.from_msg(stamp)).transform
        q, t = msg.rotation, msg.translation
        return matrix(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [t.x, t.y, t.z])

    def capture(self):
        self.spin(0.1)
        self.joints.clear()
        self.spin(self.c['settle_seconds'] + 0.15)
        image, info = self.image, self.info
        if image is None or info is None or len(self.joints) < 3:
            raise ValueError('Need image, CameraInfo and timestamped arm joint states')
        now = self.node.get_clock().now().nanoseconds * 1e-9
        image_time = stamp_seconds(image.header.stamp)
        if not 0 <= now - image_time <= self.c['max_sample_age_seconds']:
            raise ValueError('Image stale/future-dated; check ROS clock and camera timestamps')
        if image.header.frame_id != info.header.frame_id or not image.header.frame_id:
            raise ValueError('Image and CameraInfo optical frame IDs must match')
        if (image.width, image.height) != (info.width, info.height):
            raise ValueError('Image and CameraInfo dimensions do not match')
        if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
            raise ValueError('Use unbinned full-frame images')
        stamps, positions = zip(*self.joints)
        positions = np.array(positions)
        if stamps[-1] - stamps[0] < self.c['settle_seconds'] * 0.8:
            raise ValueError('Insufficient fresh joint-state coverage')
        if np.max(np.ptp(positions, axis=0)) > self.c['max_joint_drift_rad']:
            raise ValueError('Arm moving; hold still and capture again')
        nearest = int(np.argmin(np.abs(np.asarray(stamps) - image_time)))
        if abs(stamps[nearest] - image_time) > 0.05 or image_time < stamps[0]:
            raise ValueError('Image/joint timestamps are not aligned within 50 ms')
        camera_info = {'k': list(info.k), 'd': list(info.d), 'width': info.width,
                       'height': info.height, 'distortion_model': info.distortion_model}
        pixels = self.bridge.imgmsg_to_cv2(image, desired_encoding='mono8')
        target, error, markers = detect_target(pixels, camera_info, self.c)
        view = float(np.degrees(np.arccos(min(1.0, abs(target[2, 2])))))
        distance = float(np.linalg.norm(target[:3, 3]))
        for key, bad, text in (
                ('min_view_angle_deg', lambda lim: view < lim, 'board too head-on: tilt the hand more'),
                ('max_view_angle_deg', lambda lim: view > lim, 'board too oblique: tilt the hand less'),
                ('min_distance_m', lambda lim: distance < lim, 'too close to the board: move back'),
                ('max_distance_m', lambda lim: distance > lim, 'too far from the board: move closer')):
            if key in self.c and bad(self.c[key]):
                raise ValueError(f'{text} (view {view:.0f} deg, distance {distance:.2f} m; '
                                 f'{key} = {self.c[key]})')
        robot = self.transform(self.c['base_frame'], self.c['ee_frame'], image.header.stamp)
        mount = self.transform(self.c['camera_mount_frame'], image.header.frame_id, image.header.stamp)
        sample = {'stamp_seconds': image_time, 'T_base_ee': robot.tolist(),
                  'T_camera_target': target.tolist(), 'reprojection_rms_px': error,
                  'markers': markers, 'joint_positions': positions[nearest].tolist()}
        return sample, image.header.frame_id, camera_info, mount.tolist(), pixels


def add_sample(data, captured, path):
    sample, frame, info, mount, pixels = captured
    if data['samples']:
        if (frame != data['camera_frame'] or info != data['camera_info']
                or not np.allclose(mount, data['T_mount_camera'], atol=1e-7)):
            raise ValueError('Camera frame/intrinsics/internal extrinsics changed; start a new dataset')
        previous = rigid(data['samples'][-1]['T_base_ee'])
        current = rigid(sample['T_base_ee'])
        angle = Rotation.from_matrix(previous[:3, :3].T @ current[:3, :3]).magnitude()
        if np.linalg.norm(previous[:3, 3] - current[:3, 3]) < 0.01 and angle < np.deg2rad(5):
            raise ValueError('Pose too similar to the previous capture')
    data.update(camera_frame=frame, camera_info=info, T_mount_camera=mount)
    image_dir = Path(str(path) + '.images')
    image_dir.mkdir(exist_ok=True)
    image_path = image_dir / f'{len(data["samples"]):03d}.png'
    if not cv2.imwrite(str(image_path), pixels):
        raise IOError('Could not save captured image')
    sample['image_file'] = str(image_path)
    data['samples'].append(sample)
    save(path, data)
    print(f'Saved pose {len(data["samples"])}; reprojection {sample["reprojection_rms_px"]:.3f} px', flush=True)


def wait_future(collector, future, timeout=15):
    import rclpy
    rclpy.spin_until_future_complete(collector.node, future, timeout_sec=timeout)
    if not future.done():
        raise TimeoutError('ROS operation timed out; inspect robot before retrying')
    return future.result()


def collision_scene(collector, apply=False):
    """Install measured envelopes, or require them before planning any replay."""
    from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
    from moveit_msgs.msg import PlanningScene, PlanningSceneComponents, AttachedCollisionObject
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    c = collector.c
    if apply:
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        for name in ('camera', 'table', 'board'):
            center, size = c[f'{name}_collision_center'], c[f'{name}_collision_size']
            for v in (center, size):
                if (not isinstance(v, list) or len(v) != 3 or
                        not np.isfinite(np.asarray(v, dtype=float)).all()):
                    raise ValueError(f'Measure {name} collision center and size first')
            if min(size) <= 0:
                raise ValueError('Collision box dimensions must be positive')
            obj = CollisionObject(id=f'handeye_{name}', operation=CollisionObject.ADD)
            obj.header.frame_id = c['ee_frame'] if name == 'camera' else c['base_frame']
            pose = Pose()
            pose.orientation.w = 1.
            pose.position.x, pose.position.y, pose.position.z = map(float, center)
            obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(map(float, size)))]
            obj.primitive_poses = [pose]
            if name == 'camera':
                # Only permit intentional mounting contact, never all robot links.
                scene.robot_state.attached_collision_objects = [AttachedCollisionObject(
                    link_name=c['ee_frame'], object=obj, touch_links=c['camera_touch_links'])]
            else:
                scene.world.collision_objects.append(obj)
        client = collector.node.create_client(ApplyPlanningScene, 'apply_planning_scene')
        if not client.wait_for_service(timeout_sec=10):
            raise RuntimeError('MoveIt planning-scene service unavailable')
        if not wait_future(collector, client.call_async(ApplyPlanningScene.Request(scene=scene))).success:
            raise RuntimeError('Could not apply collision geometry')
        print('Measured collision envelopes installed; inspect geometry in RViz before replay.')
        return
    client = collector.node.create_client(GetPlanningScene, 'get_planning_scene')
    if not client.wait_for_service(timeout_sec=10):
        raise RuntimeError('MoveIt planning-scene service unavailable')
    req = GetPlanningScene.Request()
    req.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
                                 PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
    scene = wait_future(collector, client.call_async(req)).scene
    world = {o.id: o for o in scene.world.collision_objects}
    attached = {o.object.id: o for o in scene.robot_state.attached_collision_objects}
    if not all(name in world and world[name].primitives for name in ('handeye_table', 'handeye_board')):
        raise RuntimeError('Measured table/board collision boxes missing; run scene and inspect RViz')
    camera = attached.get('handeye_camera')
    if camera is None or not camera.object.primitives or camera.link_name != c['ee_frame']:
        raise RuntimeError('Camera collision envelope missing on end effector; run scene first')


def action(collector, client, goal):
    from action_msgs.msg import GoalStatus
    handle = wait_future(collector, client.send_goal_async(goal))
    if not handle.accepted:
        raise RuntimeError('MoveIt rejected goal')
    try:
        result = wait_future(collector, handle.get_result_async(), 120)
    except (KeyboardInterrupt, TimeoutError):
        wait_future(collector, handle.cancel_goal_async(), 5)
        raise
    if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code.val != 1:
        raise RuntimeError(f'MoveIt failed: status={result.status}, code={result.result.error_code.val}')
    return result.result


def replay(collector, source, output, execute):
    """Plan the WHOLE taught sequence first; execute those exact plans only on request."""
    from rclpy.action import ActionClient
    from moveit_msgs.action import MoveGroup, ExecuteTrajectory
    from moveit_msgs.msg import Constraints, JointConstraint
    c = collector.c
    collision_scene(collector)
    arm = ActionClient(collector.node, MoveGroup, 'move_action')
    if not arm.wait_for_server(timeout_sec=15):
        raise RuntimeError('MoveIt unavailable')
    plans = []
    seed = None
    for i, sample in enumerate(source['samples']):
        q = np.asarray(sample['joint_positions'], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Invalid taught joint pose')
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = 'fr3_arm'
        req.allowed_planning_time = 10.
        req.num_planning_attempts = 5
        req.max_velocity_scaling_factor = float(c['velocity_scale'])
        req.max_acceleration_scaling_factor = float(c['acceleration_scale'])
        req.start_state.is_diff = True
        if seed is not None:
            req.start_state.joint_state.name = list(seed[0])
            req.start_state.joint_state.position = list(seed[1])
        req.goal_constraints = [Constraints(joint_constraints=[JointConstraint(
            joint_name=j, position=float(v), tolerance_above=0.001,
            tolerance_below=0.001, weight=1.) for j, v in zip(JOINTS, q)])]
        goal.planning_options.plan_only = True
        trajectory = action(collector, arm, goal).planned_trajectory
        if not trajectory.joint_trajectory.points:
            raise RuntimeError(f'Pose {i + 1}: empty plan')
        seed = (trajectory.joint_trajectory.joint_names,
                trajectory.joint_trajectory.points[-1].positions)
        plans.append(trajectory)
        print(f'Planned taught pose {i + 1}/{len(source["samples"])}', flush=True)
    if not execute:
        print('Planning only: no execution commands sent. Review ALL paths in RViz.')
        return
    executor = ActionClient(collector.node, ExecuteTrajectory, 'execute_trajectory')
    if not executor.wait_for_server(timeout_sec=10):
        raise RuntimeError('Trajectory execution unavailable')
    data = {'config': c, 'samples': []}
    # Board must be visible BEFORE first motion. Later captures abort the sequence on loss.
    collector.capture()
    for trajectory in plans:
        collector.spin(0.2)
        if not collector.joints:
            raise RuntimeError('No current robot joints before execution')
        stamp, current = collector.joints[-1]
        now = collector.node.get_clock().now().nanoseconds * 1e-9
        if not 0 <= now - stamp <= c['max_sample_age_seconds']:
            raise RuntimeError('Joint state stale before execution')
        start = dict(zip(trajectory.joint_trajectory.joint_names,
                         trajectory.joint_trajectory.points[0].positions))
        if any(abs(start[j] - v) > 0.01 for j, v in zip(JOINTS, current)):
            raise RuntimeError('Robot no longer at planned start; refusing execution')
        action(collector, executor, ExecuteTrajectory.Goal(trajectory=trajectory))
        add_sample(data, collector.capture(), output)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('scene').add_argument('--output', default='unused_scene_output')
    sub.add_parser('board').add_argument('--output', required=True)
    sub.add_parser('collect').add_argument('--output', required=True)
    s = sub.add_parser('solve')
    s.add_argument('--input', required=True)
    s.add_argument('--output', required=True)
    s = sub.add_parser('validate')
    s.add_argument('--input', required=True, help='Independent dataset; keep board fixed')
    s.add_argument('--calibration', required=True)
    s.add_argument('--output', required=True)
    s = sub.add_parser('replay')
    s.add_argument('--input', required=True, help='Previously collected dataset containing taught poses')
    s.add_argument('--output', required=True, help='New dataset; only written when executing')
    s.add_argument('--execute', action='store_true')
    args = p.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        c = yaml.safe_load(stream)
    make_board(c)
    for key in ('velocity_scale', 'acceleration_scale'):
        if not 0 < c[key] <= 0.05:
            raise ValueError(f'{key} must be in (0, 0.05]')
    if c['settle_seconds'] < 1.0:
        raise ValueError('settle_seconds must be at least 1 second')
    if Path(args.output).exists() or Path(str(args.output) + '.images').exists():
        raise FileExistsError('Choose a new output path; existing data will not be overwritten')
    if args.command == 'board':
        _, board = make_board(c)
        b = c['board']
        if b.get('type') == 'charuco':
            width, height = b['columns'] * b['square_length'], b['rows'] * b['square_length']
        else:
            width = b['columns'] * b['marker_length'] + (b['columns'] - 1) * b['marker_separation']
            height = b['rows'] * b['marker_length'] + (b['rows'] - 1) * b['marker_separation']
        # Equal pixels/metre in both axes; white external border excluded from board dimensions.
        size = (round(width * 10000), round(height * 10000))
        image = board.draw(size) if hasattr(board, 'draw') else board.generateImage(size)
        image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
        if not cv2.imwrite(args.output, image):
            raise IOError('Could not write board')
        print(f'Black grid extent: {width * 1000:.1f} x {height * 1000:.1f} mm; '
              f'print at 10000 pixels/metre (254 dpi), measure marker edges and gaps after printing.')
        return
    if args.command in ('solve', 'replay', 'validate'):
        with open(args.input, encoding='utf-8') as stream:
            source = json.load(stream)
        if any(source['config'][k] != c[k] for k in ('base_frame', 'ee_frame', 'board', 'camera_mount_frame')):
            raise ValueError('Dataset frames/board differ from config')
        if args.command in ('solve', 'validate'):
            if args.command == 'solve':
                result = solve(source, c)
            else:
                with open(args.calibration, encoding='utf-8') as stream:
                    calibration = json.load(stream)
                result = validate_result(source, calibration, c)
            save(args.output, result)
            print(json.dumps(result, indent=2))
            if result['status'] == 'REJECTED':
                raise SystemExit(2)
            return
        if not source['samples']:
            raise ValueError('No taught poses')
    import rclpy
    rclpy.init()
    collector = Collector(c)
    try:
        if args.command == 'replay':
            replay(collector, source, args.output, args.execute)
        elif args.command == 'scene':
            collision_scene(collector, apply=True)
        else:
            from tf2_ros import TransformException
            data = {'config': c, 'samples': []}
            print('No motion commands. Hold a pose; Enter captures. Type q to finish.')
            while input('Capture [Enter/q]: ').strip().lower() != 'q':
                try:
                    add_sample(data, collector.capture(), args.output)
                except (ValueError, RuntimeError, TransformException) as exc:
                    print(f'Capture rejected: {exc}')
            print('Dataset complete. Run solve next.')
    finally:
        collector.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
