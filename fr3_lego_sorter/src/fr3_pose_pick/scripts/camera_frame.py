"""Convert a pose given in the camera's colour optical frame into the robot base frame.

No detection happens here: someone else (the perception team, or a person reading a picture) says
"the brick is at this point / orientation in camera_color_optical_frame", and the hand-eye matrix
turns that into fr3_link0 coordinates:

    p_base = T_base_hand @ T_hand_camera @ p_camera

T_base_hand is the arm's forward kinematics AT THE MOMENT THE PICTURE WAS TAKEN (the arm's current
pose from TF, or --ee-pose typed in); T_hand_camera is the hand-eye result (calibration_file in the
config, parent frame fr3_hand_tcp, child frame camera_color_optical_frame, T_ee_camera in the file).
Camera optical frame: x right, y down, z forward (out of the lens).

The pure functions need no ROS and are unit tested.
"""
import json
import math

import numpy as np


def wrap_line_angle(angle):
    """A brick's long axis repeats every pi: report it in (-pi/2, pi/2]."""
    wrapped = (angle + math.pi / 2) % math.pi - math.pi / 2
    return math.pi / 2 if wrapped <= -math.pi / 2 + 1e-12 else wrapped


def pose_matrix(xyz, quat_xyzw):
    """4x4 homogeneous matrix from a position and an xyzw quaternion."""
    from scipy.spatial.transform import Rotation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    m[:3, 3] = xyz
    return m


def load_calibration(path, c, camera_frame='camera_color_optical_frame'):
    """(calibration dict, T_hand_camera 4x4). Refuses a file for other frames, and a calibration that
    failed its quality gates unless allow_rejected_calibration is set in the config."""
    calibration = json.load(open(path, encoding='utf-8'))
    if calibration['child_frame'] != camera_frame or calibration['parent_frame'] != c['ee_frame']:
        raise ValueError(f'Calibration frames {calibration["parent_frame"]} -> {calibration["child_frame"]} '
                         f'do not match {c["ee_frame"]} -> {camera_frame}')
    if calibration['status'] != 'passed_quality_gates' and not c.get('allow_rejected_calibration', False):
        raise ValueError('Calibration did not pass its quality gates; see allow_rejected_calibration')
    return calibration, np.array(calibration['T_ee_camera'], dtype=float)


def image_angle_to_quat(angle_deg):
    """The orientation (xyzw, camera frame) of a brick lying flat in front of the camera whose long side makes
    `angle_deg` in the IMAGE, measured from the image x axis (right) toward the image y axis (down), i.e. clockwise
    on the screen. The brick's x axis is then its long side, in the plane of the image (the camera looks almost
    straight down at the table, so that plane is the table plane). Use it with the long axis 'x'."""
    half = math.radians(angle_deg) / 2
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def camera_point_to_base(t_base_hand, t_hand_cam, p_cam):
    """A point in the camera frame -> the base frame."""
    return (np.asarray(t_base_hand) @ np.asarray(t_hand_cam) @ np.append(np.asarray(p_cam, dtype=float), 1.0))[:3]


def camera_yaw_to_base(t_base_hand, t_hand_cam, quat_cam, long_axis='x'):
    """Yaw (radians, in (-pi/2, pi/2]) in the base frame of a brick whose orientation is given as an
    xyzw quaternion in the camera frame. `long_axis` says which axis of the brick's own frame runs along
    its long side ('x' or 'y'); only that axis, projected on the table plane, is used. So the direction
    the brick frame's z points (up or down) does not matter. Returns (yaw, tilt_deg) where tilt is how
    far the brick's long axis leans out of the table plane (a brick lying flat has about 0)."""
    from scipy.spatial.transform import Rotation
    if long_axis not in ('x', 'y'):
        raise ValueError("long_axis must be 'x' or 'y'")
    x, y, z, w = quat_cam
    if math.sqrt(x * x + y * y + z * z + w * w) < 1e-9:
        raise ValueError('Quaternion is all zeros')
    r_base_brick = (np.asarray(t_base_hand) @ np.asarray(t_hand_cam))[:3, :3] @ Rotation.from_quat(quat_cam).as_matrix()
    axis = r_base_brick[:, 0 if long_axis == 'x' else 1]
    horizontal = math.hypot(axis[0], axis[1])
    tilt = math.degrees(math.atan2(abs(axis[2]), horizontal))
    if horizontal < 0.2:
        raise ValueError('The brick long axis points almost straight up or down in the base frame: '
                         'the camera orientation is probably not a brick lying flat (or long_axis is wrong)')
    return wrap_line_angle(math.atan2(axis[1], axis[0])), tilt


def table_check(c, top_z):
    """Signed difference (m) between a top-face z and where a brick lying on the table has it."""
    return top_z - (c['table_top_z'] + c['block_size'][2])
