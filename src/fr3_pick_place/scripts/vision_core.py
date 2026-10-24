"""Frame math, measurement gates and bounded stopped refinement (no ROS imports)."""
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation

BASE = 'fr3_link0'
TCP = 'fr3_hand_tcp'


def vector(value, n, name):
    if not isinstance(value, list) or len(value) != n or any(isinstance(x, bool) for x in value):
        raise ValueError(f'{name}: supply {n} measured numeric values.')
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: invalid numeric values.') from exc
    if not np.isfinite(result).all():
        raise ValueError(f'{name}: values must be finite.')
    return result


def matrix(pose):
    xyz = vector(pose.get('xyz_m'), 3, 'xyz_m')
    q = vector(pose.get('quaternion_xyzw'), 4, 'quaternion_xyzw')
    if abs(np.linalg.norm(q) - 1) > .001:
        raise ValueError('Quaternion must have unit norm.')
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = Rotation.from_quat(q).as_matrix(), xyz
    return t


def pose_dict(t):
    return {'xyz_m': np.asarray(t[:3, 3]).tolist(),
            'quaternion_xyzw': Rotation.from_matrix(t[:3, :3]).as_quat().tolist()}


def angle(a, b):
    return float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


def validate_config(config):
    c = copy.deepcopy(config)
    if c.get('base_frame') != BASE or c.get('tcp_frame') != TCP:
        raise ValueError(f'Version 1 requires base_frame={BASE}, tcp_frame={TCP}.')
    if c.get('backend') not in ('mock', 'sam3'):
        raise ValueError('backend must be mock or sam3.')
    for name in ('observation_tcp', 'place_tcp', 'object_tcp_grasp'):
        matrix(c[name])
    ext = c['camera_extrinsic']
    matrix(ext)
    if ext.get('parent_frame') not in (TCP, 'fr3_hand', 'fr3_link8') or not ext.get('optical_frame'):
        raise ValueError('Camera extrinsic needs a wrist parent and an exact optical_frame.')
    size = vector(c['brick_size_m'], 3, 'brick_size_m')
    if np.any(size <= 0) or size[1] >= c['gripper']['open_width_m']:
        raise ValueError('Brick dimensions must be positive and fit inside the open hand.')
    for name in ('table', 'camera_collision'):
        vector(c[name].get('center_m'), 3, name + '.center_m')
        if np.any(vector(c[name].get('size_m'), 3, name + '.size_m') <= 0):
            raise ValueError(f'{name} dimensions must be positive.')
    for key in ('velocity_scale', 'acceleration_scale'):
        if not 0 < c[key] <= .05:
            raise ValueError(f'{key} must be in (0, 0.05].')
    g = c['gripper']
    if not (0 < g['grasp_width_m'] < g['open_width_m'] <= .08
            and 0 < g['speed_m_s'] <= .05 and 0 < g['force_n'] <= 20):
        raise ValueError('Invalid gripper widths, speed or force.')
    r = c['refinement']
    for key in ('clearance_m', 'max_xy_step_m', 'max_yaw_step_deg', 'xy_tolerance_m',
                'yaw_tolerance_deg', 'position_stability_m', 'angle_stability_deg',
                'association_distance_m', 'max_object_shift_m', 'min_camera_depth_m',
                'max_camera_depth_m', 'final_descent_max_m', 'max_result_age_s'):
        if not isinstance(r[key], (int, float)) or not math.isfinite(r[key]) or r[key] <= 0:
            raise ValueError(f'refinement.{key} must be positive and finite.')
    for key in ('max_iterations', 'stable_observations'):
        if isinstance(r[key], bool) or not isinstance(r[key], int) or r[key] < 2:
            raise ValueError(f'refinement.{key} must be an integer >= 2.')
    if r['final_descent_max_m'] < r['clearance_m'] or r['max_camera_depth_m'] <= r['min_camera_depth_m']:
        raise ValueError('Invalid descent/camera depth range.')
    if c['backend'] == 'mock':
        matrix(c['mock']['object_base'])
    if c['backend'] == 'sam3' and not c.get('calibration_validated', False):
        raise ValueError('Validate the camera transform independently, then set calibration_validated: true.')
    return c


def normalize_bricks(bricks, report, camera_to_base, config):
    """Strict single isolated-brick gate; quality is colour class, not confidence."""
    p, r = config['perception'], config['refinement']
    if not all(math.isfinite(report[k]) for k in ('table_median_residual_m', 'table_inlier_fraction')):
        raise RuntimeError('Non-finite plane metrics.')
    if report['table_median_residual_m'] > p['max_plane_residual_m'] or report['table_inlier_fraction'] < p['min_plane_inlier_fraction']:
        raise RuntimeError('Table plane quality rejected.')
    normal_base = camera_to_base[:3, :3] @ np.asarray(report['table_normal'])
    if not np.isfinite(normal_base).all() or normal_base[2] < math.cos(math.radians(p['max_table_tilt_deg'])):
        raise RuntimeError('Version 1 requires a horizontal table and downward grasp.')
    valid = [b for b in bricks if b.get('id') is not None]
    if len(valid) != 1 or valid[0].get('status') != 'brick':
        raise RuntimeError('Version 1 requires exactly one separated, unsplit regular brick.')
    b = valid[0]
    for key, threshold, direction in (
            ('score', p['min_score'], 1), ('valid_fraction', p['min_valid_fraction'], 1),
            ('height_spread_m', p['max_height_spread_m'], -1)):
        if key not in b or not math.isfinite(b[key]) or direction*b[key] < direction*threshold:
            raise RuntimeError(f'Brick rejected by {key} gate.')
    if b.get('touches_border', True):
        raise RuntimeError('Brick touches the image edge.')
    measured = np.asarray(b['measured_mm']) / 1000
    if np.any(np.abs(measured - np.asarray(config['brick_size_m'][:2])) > p['size_relative_tolerance'] * np.asarray(config['brick_size_m'][:2])):
        raise RuntimeError('Measured footprint differs from configured dimensions.')
    object_camera = np.eye(4)
    rotation = np.asarray(b['box_rotation_cam'], dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > .002 or np.linalg.det(rotation) < .99:
        raise RuntimeError('Invalid object rotation.')
    object_camera[:3, :3] = Rotation.from_matrix(rotation).as_matrix()
    object_camera[:3, 3] = vector(b['box_centre_cam_m'], 3, 'box centre')
    top_z = b['centre_cam_m'][2]
    if not r['min_camera_depth_m'] <= top_z <= r['max_camera_depth_m']:
        raise RuntimeError('Object outside configured reliable camera depth range.')
    result = pose_dict(camera_to_base @ object_camera)
    result.update(colour=b.get('colour'), size_m=b['box_size_m'], source_id=b['id'])
    result['metrics'] = {'score': b['score'], 'depth_valid_fraction': b['valid_fraction'],
        'height_spread_m': b['height_spread_m'], 'top_camera_depth_m': top_z,
        'plane_median_residual_m': report['table_median_residual_m'],
        'plane_inlier_fraction': report['table_inlier_fraction']}
    return result


class Refinement:
    def __init__(self, config):
        self.c = config
        self.first = self.previous = None
        self.colour = None
        self.stable = self.iterations = 0

    def observe(self, detection):
        obj = matrix(detection)
        r = self.c['refinement']
        self.iterations += 1
        if self.iterations > r['max_iterations']:
            raise RuntimeError('Refinement iteration budget exhausted.')
        if self.previous is not None:
            if detection.get('colour') != self.colour or np.linalg.norm(obj[:3, 3]-self.previous[:3, 3]) > r['association_distance_m']:
                raise RuntimeError('Selected brick lost or association ambiguous.')
            # A rectangular footprint has a 180-degree yaw ambiguity. Keep the
            # equivalent orientation nearest to the selected object's last pose.
            alternative = obj.copy()
            alternative[:3, :3] = obj[:3, :3] @ np.diag([-1., -1., 1.])
            obj = min((obj, alternative), key=lambda t: angle(self.previous, t))
            if np.linalg.norm(obj[:3, 3]-self.first[:3, 3]) > r['max_object_shift_m']:
                raise RuntimeError('Object moved too far from initial selection.')
            if (np.linalg.norm(obj[:3, 3]-self.previous[:3, 3]) <= r['position_stability_m']
                    and angle(self.previous, obj) <= r['angle_stability_deg']):
                self.stable += 1
            else:
                self.stable = 1
        else:
            self.first, self.colour, self.stable = obj.copy(), detection.get('colour'), 1
        self.previous = obj.copy()
        return obj

    def grasp_pose(self, obj):
        grasp = obj @ matrix(self.c['object_tcp_grasp'])
        # Preserve base-vertical tool Z; only yaw may be refined for version 1.
        if grasp[2, 2] > -math.cos(math.radians(self.c['perception']['max_table_tilt_deg'])):
            raise RuntimeError('Configured grasp does not point tool Z downward.')
        yaw = math.atan2(grasp[1, 0], grasp[0, 0])
        grasp[:3, :3] = Rotation.from_euler('z', yaw).as_matrix() @ np.diag([1., -1., -1.])
        return grasp

    def next_pose(self, current, obj):
        r = self.c['refinement']
        grasp = self.grasp_pose(obj)
        yaw = math.atan2(grasp[1, 0], grasp[0, 0])
        desired = grasp.copy()
        desired[2, 3] += r['clearance_m']
        error = desired[:2, 3]-current[:2, 3]
        yaw_current = math.atan2(current[1, 0], current[0, 0])
        yaw_error = math.atan2(math.sin(yaw-yaw_current), math.cos(yaw-yaw_current))
        aligned = np.linalg.norm(error) <= r['xy_tolerance_m'] and abs(math.degrees(yaw_error)) <= r['yaw_tolerance_deg']
        if aligned and self.stable >= r['stable_observations']:
            if not .001 <= current[2, 3]-grasp[2, 3] <= r['final_descent_max_m']:
                raise RuntimeError('Final descent height outside configured limits.')
            return 'ready', grasp
        target = current.copy()
        target[:2, 3] += error * min(1., r['max_xy_step_m']/max(np.linalg.norm(error), 1e-12))
        step = np.clip(yaw_error, -math.radians(r['max_yaw_step_deg']), math.radians(r['max_yaw_step_deg']))
        target[:3, :3] = Rotation.from_euler('z', yaw_current+step).as_matrix() @ np.diag([1., -1., -1.])
        # Refinement stays at the established pregrasp height; no blind descent.
        return ('observe' if aligned else 'correct'), target
