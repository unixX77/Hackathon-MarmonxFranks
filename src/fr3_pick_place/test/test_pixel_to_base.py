"""Checks for pixel_to_base: the pixel/depth -> base chain, including a tilted camera."""
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pb', ROOT / 'scripts' / 'pixel_to_base.py')
pb = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / 'scripts'))
spec.loader.exec_module(pb)
CONFIG_PATH = ROOT / 'config' / 'pose_pick.yaml'
CONFIG = yaml.safe_load(open(CONFIG_PATH, encoding='utf-8'))
K = (910.0, 905.0, 645.0, 358.0)

# a real-looking hand-eye result (shape of charuco_result.json)
T_EE_CAM = np.array([[0.03938, -0.99825, 0.04420, 0.05156],
                     [0.99921, 0.03911, -0.00687, -0.03667],
                     [0.00513, 0.04444, 0.99900, -0.04470],
                     [0, 0, 0, 1.0]])


def tilted_hand(roll_deg, pitch_deg, xyz):
    """Hand pose looking down but tilted by the given angles (a camera that is NOT perpendicular)."""
    from scipy.spatial.transform import Rotation
    down = Rotation.from_euler('x', math.pi)
    tilt = Rotation.from_euler('xy', [roll_deg, pitch_deg], degrees=True)
    return pb.pose_matrix(xyz, (tilt * down).as_quat())


def test_deproject_project_roundtrip():
    p = pb.deproject(700, 300, 0.42, *K)
    u, v, z = pb.project(p, *K)
    assert (u, v, z) == pytest.approx((700, 300, 0.42))
    assert pb.deproject(K[2], K[3], 0.5, *K) == pytest.approx([0, 0, 0.5])      # principal point is straight ahead


@pytest.mark.parametrize('roll, pitch', [(0, 0), (12, -8), (-25, 20), (35, 0)])
def test_known_point_is_recovered_even_when_the_camera_is_tilted(roll, pitch):
    t_base_cam = tilted_hand(roll, pitch, [0.40, 0.05, 0.45]) @ T_EE_CAM
    truth = np.array([0.47, -0.02, CONFIG['table_top_z'] + CONFIG['block_size'][2]])    # top face on the table
    p_cam = pb.transform_point(np.linalg.inv(t_base_cam), truth)
    assert p_cam[2] > 0.05                                           # the point is in front of the camera
    u, v, depth = pb.project(p_cam, *K)
    found = pb.brick_pose_from_pixels(t_base_cam, K, (u, v), depth, CONFIG['block_size'][2])
    assert found['top'] == pytest.approx(truth, abs=1e-9)
    assert found['centre'][2] == pytest.approx(truth[2] - CONFIG['block_size'][2] / 2)
    assert abs(pb.table_check(CONFIG, found['top'][2])) < 1e-9


def test_straight_down_shortcut_would_be_wrong_when_tilted():
    """Why the full chain matters: assuming 'camera looks straight down' errs by centimetres."""
    t_base_cam = tilted_hand(25, 0, [0.40, 0.0, 0.45]) @ T_EE_CAM
    truth = np.array([0.47, -0.08, -0.0104])
    u, v, depth = pb.project(pb.transform_point(np.linalg.inv(t_base_cam), truth), *K)
    p = pb.deproject(u, v, depth, *K)
    shortcut = t_base_cam[:3, 3] + np.array([p[0], p[1], -p[2]])                  # ignore the rotation
    assert np.linalg.norm(shortcut - truth) > 0.05


def test_hand_eye_error_shows_up_in_the_table_check():
    """A 2 degree camera rotation error moves the top face by well over the 8 mm tolerance at 0.45 m."""
    from scipy.spatial.transform import Rotation
    good = tilted_hand(0, 0, [0.40, 0.0, 0.45]) @ T_EE_CAM
    bad = good.copy()
    bad[:3, :3] = Rotation.from_euler('y', 2, degrees=True).as_matrix() @ good[:3, :3]
    truth = np.array([0.47, 0.0, CONFIG['table_top_z'] + CONFIG['block_size'][2]])
    u, v, depth = pb.project(pb.transform_point(np.linalg.inv(good), truth), *K)
    found = pb.brick_pose_from_pixels(bad, K, (u, v), depth, CONFIG['block_size'][2])
    assert np.linalg.norm(found['top'] - truth) > 0.008


def test_yaw_from_second_click():
    t_base_cam = tilted_hand(0, 0, [0.40, 0.0, 0.45]) @ T_EE_CAM
    centre = np.array([0.45, 0.0, -0.0104])
    end = centre + 0.016 * np.array([math.cos(math.radians(35)), math.sin(math.radians(35)), 0])
    px = [pb.project(pb.transform_point(np.linalg.inv(t_base_cam), q), *K) for q in (centre, end)]
    found = pb.brick_pose_from_pixels(t_base_cam, K, px[0][:2], px[0][2], 0.0096, px[1][:2], px[1][2])
    assert math.degrees(found['yaw']) == pytest.approx(35, abs=1e-6)
    with pytest.raises(ValueError, match='less than 8 mm'):
        pb.brick_pose_from_pixels(t_base_cam, K, px[0][:2], px[0][2], 0.0096, px[0][:2], px[0][2])


def test_robust_depth_ignores_holes_and_rejects_missing_data():
    depth = np.full((20, 20), 0.5)
    depth[10, 10] = 0.0                                               # a hole at the clicked pixel
    depth[9, 9] = 2.0                                                 # one flying pixel
    value, spread = pb.robust_depth(depth, 10, 10)
    assert value == pytest.approx(0.5) and spread == pytest.approx(0.0)
    with pytest.raises(ValueError, match='depth is missing'):
        pb.robust_depth(np.zeros((20, 20)), 10, 10)
    with pytest.raises(ValueError, match='outside'):
        pb.robust_depth(depth, 99, 5)


def test_calibration_frame_mismatch_is_refused(tmp_path):
    bad = tmp_path / 'cal.json'
    bad.write_text(json.dumps({'parent_frame': 'fr3_link8', 'child_frame': 'x', 'status': 'passed_quality_gates',
                               'T_ee_camera': np.eye(4).tolist()}))
    with pytest.raises(ValueError, match='do not match'):
        pb.load_calibration(bad, CONFIG)


def test_offline_command_line_end_to_end(tmp_path):
    cal = tmp_path / 'cal.json'
    cal.write_text(json.dumps({'parent_frame': 'fr3_hand_tcp', 'child_frame': 'camera_color_optical_frame',
                               'status': 'REJECTED', 'T_ee_camera': T_EE_CAM.tolist(),
                               'all_samples': {'translation_rms_m': 0.003, 'rotation_rms_deg': 0.9}}))
    t_base_cam = pb.pose_matrix([0.4, 0.0, 0.45], [1, 0, 0, 0]) @ T_EE_CAM
    truth = np.array([0.47, 0.02, CONFIG['table_top_z'] + CONFIG['block_size'][2]])
    u, v, depth = pb.project(pb.transform_point(np.linalg.inv(t_base_cam), truth), *K)
    out = subprocess.run(
        [sys.executable, '-I', str(ROOT / 'scripts' / 'pixel_to_base.py'), '--config', str(CONFIG_PATH),
         '--calibration', str(cal), '--pixel', f'{u:.6f}', f'{v:.6f}', '--depth', f'{depth:.6f}',
         '--intrinsics', *map(str, K), '--ee-pose', '0.4', '0', '0.45', '1', '0', '0', '0', '--yaw', '20'],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert 'x=0.4700 y=0.0200' in out.stdout
    assert '-> OK' in out.stdout
    assert 'pose_pick.py' in out.stdout and '--yaw 20.0' in out.stdout
