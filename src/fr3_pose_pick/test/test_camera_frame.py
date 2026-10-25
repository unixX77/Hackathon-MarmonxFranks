"""Camera-frame input: the hand-eye chain, yaw from a camera-frame quaternion, and the table check."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import sys                                                    # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts'))
cf = load('camera_frame')
pp = load('pose_pick')
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))
REPO = ROOT.parents[1]
CALIBRATION = REPO / 'data' / 'calibration' / 'charuco_result.json'

# a hand pointing straight down, 40 cm out and 40 cm up, turned 25 degrees
HAND = cf.pose_matrix([0.40, 0.05, 0.40],
                      (Rotation.from_euler('z', 25, degrees=True) * Rotation.from_euler('x', 180, degrees=True)).as_quat())
# a plausible hand-eye matrix (camera a few cm off the hand, looking along the fingers)
HAND_CAM = cf.pose_matrix([0.05, -0.04, -0.045], Rotation.from_euler('z', 90, degrees=True).as_quat())


def camera_point_of(base_point):
    """Where a base-frame point appears in the camera frame (the inverse of the conversion)."""
    return (np.linalg.inv(HAND @ HAND_CAM) @ np.append(base_point, 1.0))[:3]


def test_a_known_base_point_is_recovered_from_its_camera_coordinates():
    base = np.array([0.46, 0.12, -0.0104])
    assert cf.camera_point_to_base(HAND, HAND_CAM, camera_point_of(base)) == pytest.approx(base, abs=1e-9)


def test_the_hand_pose_matters():
    p_cam = camera_point_of(np.array([0.46, 0.12, -0.0104]))
    moved = HAND.copy()
    moved[0, 3] += 0.05
    assert np.linalg.norm(cf.camera_point_to_base(moved, HAND_CAM, p_cam) -
                          cf.camera_point_to_base(HAND, HAND_CAM, p_cam)) == pytest.approx(0.05)


@pytest.mark.skipif(not CALIBRATION.exists(), reason='charuco_result.json not found')
def test_real_calibration_file_loads_and_is_used_as_is():
    calibration, t_hand_cam = cf.load_calibration(CALIBRATION, CONFIG)
    assert calibration['parent_frame'] == CONFIG['ee_frame']
    assert t_hand_cam == pytest.approx(np.array(json.load(open(CALIBRATION))['T_ee_camera']))
    p_cam = [0.01, -0.02, 0.43]
    expected = (HAND @ t_hand_cam @ np.array(p_cam + [1.0]))[:3]
    assert cf.camera_point_to_base(HAND, t_hand_cam, p_cam) == pytest.approx(expected)
    assert np.linalg.norm(cf.camera_point_to_base(HAND, np.eye(4), p_cam) - expected) > 0.03   # calibration matters


def test_calibration_for_other_frames_or_a_rejected_one_is_refused(tmp_path):
    good = {'parent_frame': 'fr3_hand_tcp', 'child_frame': 'camera_color_optical_frame',
            'status': 'REJECTED', 'T_ee_camera': np.eye(4).tolist()}
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps(good))
    with pytest.raises(ValueError, match='quality gates'):
        cf.load_calibration(path, dict(CONFIG, allow_rejected_calibration=False))
    cf.load_calibration(path, dict(CONFIG, allow_rejected_calibration=True))      # a recorded decision
    path.write_text(json.dumps(dict(good, child_frame='camera_link')))
    with pytest.raises(ValueError, match='do not match'):
        cf.load_calibration(path, CONFIG)


@pytest.mark.parametrize('yaw_deg', [-80, -45, 0, 20, 63, 89])
@pytest.mark.parametrize('flip', [False, True])        # brick frame z up, or z down (both conventions occur)
def test_brick_yaw_is_recovered_from_a_camera_frame_quaternion(yaw_deg, flip):
    r_base_brick = Rotation.from_euler('z', yaw_deg, degrees=True)
    if flip:
        r_base_brick = r_base_brick * Rotation.from_euler('x', 180, degrees=True)
    r_cam_brick = Rotation.from_matrix((HAND @ HAND_CAM)[:3, :3].T @ r_base_brick.as_matrix())
    yaw, tilt = cf.camera_yaw_to_base(HAND, HAND_CAM, r_cam_brick.as_quat())
    assert math.degrees(yaw) == pytest.approx(yaw_deg, abs=1e-6)
    assert tilt == pytest.approx(0, abs=1e-6)


def test_long_axis_y_and_a_leaning_or_upright_brick():
    r_base_brick = Rotation.from_euler('z', 30 - 90, degrees=True)      # its y axis points at 30 degrees
    r_cam_brick = Rotation.from_matrix((HAND @ HAND_CAM)[:3, :3].T @ r_base_brick.as_matrix()).as_quat()
    assert math.degrees(cf.camera_yaw_to_base(HAND, HAND_CAM, r_cam_brick, long_axis='y')[0]) == pytest.approx(30)
    upright = Rotation.from_matrix((HAND @ HAND_CAM)[:3, :3].T @ Rotation.from_euler('y', 90, degrees=True).as_matrix())
    with pytest.raises(ValueError, match='straight up or down'):
        cf.camera_yaw_to_base(HAND, HAND_CAM, upright.as_quat())
    with pytest.raises(ValueError, match='all zeros'):
        cf.camera_yaw_to_base(HAND, HAND_CAM, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="'x' or 'y'"):
        cf.camera_yaw_to_base(HAND, HAND_CAM, [0, 0, 0, 1], long_axis='z')


def test_table_check_and_resolving_a_camera_target():
    bias = CONFIG['camera_height_bias']
    top_z = CONFIG['table_top_z'] + CONFIG['block_size'][2]
    assert cf.table_check(CONFIG, top_z) == pytest.approx(0)
    on_table = camera_point_of(np.array([0.46, 0.12, top_z + bias]))              # the camera sees it `bias` too high
    x, y, centre_z, yaw = pp.resolve_camera_target(CONFIG, HAND, HAND_CAM, on_table, None, 'x', False, 'Pick')
    assert (x, y) == pytest.approx((0.46, 0.12), abs=1e-9)
    assert centre_z is None and yaw is None              # brick on the table: height from the table model
    # a top face 19.2 mm above a brick on the table (two bricks higher) plus 4 mm of noise: more than 3 mm from any
    # whole layer, so nothing is refused and the corrected measurement is used as it is
    floating = camera_point_of(np.array([0.46, 0.12, top_z + bias + 0.0192 + 0.004]))
    x, y, centre_z, _ = pp.resolve_camera_target(CONFIG, HAND, HAND_CAM, floating, None, 'x', False, 'Place')
    assert centre_z == pytest.approx(top_z + 0.0192 + 0.004 - CONFIG['block_size'][2] / 2)
    below = camera_point_of(np.array([0.46, 0.12, top_z - 0.02]))                  # 2 cm below a brick on the table
    with pytest.raises(ValueError, match='BELOW'):
        pp.resolve_camera_target(CONFIG, HAND, HAND_CAM, below, None, 'x', False, 'Pick')


def test_image_angle_makes_a_unit_quaternion_about_the_camera_z_axis():
    for angle in (-90, -30, 0, 45, 90):
        q = cf.image_angle_to_quat(angle)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        d = Rotation.from_quat(q).apply([1.0, 0.0, 0.0])                       # the brick's long axis in the camera frame
        assert d == pytest.approx([math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0], abs=1e-12)


@pytest.mark.skipif(not CALIBRATION.exists(), reason='charuco_result.json not found')
@pytest.mark.parametrize('image_angle', [90, 80, 100, 45, 0, -60])
def test_image_angle_to_base_yaw_with_the_real_calibration(image_angle):
    """Hand pointing down at yaw 0: image right is the robot's right (base -y) and image down is toward the robot
    (base -x), so a long side that is vertical in the image (90 deg) lies along base x (yaw 0), and an image angle of
    80 deg (turned 10 deg clockwise on the screen from vertical) is a brick turned +10 deg (counter-clockwise from above)."""
    _, t_hand_cam = cf.load_calibration(CALIBRATION, CONFIG)
    hand = cf.pose_matrix([0.40, 0.0, 0.30], [1.0, 0.0, 0.0, 0.0])           # tool pointing down, yaw 0
    yaw, tilt = cf.camera_yaw_to_base(hand, t_hand_cam, cf.image_angle_to_quat(image_angle), 'x')
    theta = math.radians(image_angle)
    expected = cf.wrap_line_angle(math.atan2(-math.cos(theta), -math.sin(theta)))
    difference = abs(math.degrees(cf.wrap_line_angle(yaw - expected)))
    assert difference < 4.0                                                    # the camera is mounted within ~2 deg of square
    assert tilt < 6.0


def test_the_hand_yaw_at_capture_is_taken_into_account():
    """The same image angle with the hand turned 25 degrees is a brick turned 25 degrees differently."""
    straight = cf.pose_matrix([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
    turned = cf.pose_matrix([0.4, 0.0, 0.3], (Rotation.from_euler('z', 25, degrees=True) * Rotation.from_euler('x', 180, degrees=True)).as_quat())
    a = cf.camera_yaw_to_base(straight, HAND_CAM, cf.image_angle_to_quat(90), 'x')[0]
    b = cf.camera_yaw_to_base(turned, HAND_CAM, cf.image_angle_to_quat(90), 'x')[0]
    assert abs(math.degrees(cf.wrap_line_angle(b - a))) == pytest.approx(25.0, abs=1e-6)
