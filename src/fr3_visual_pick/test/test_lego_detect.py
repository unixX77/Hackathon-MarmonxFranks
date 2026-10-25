"""Offline check: render a brick at a known pose and recover it with detect_bricks."""
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lego', ROOT / 'scripts' / 'lego_pick_place.py')
lego = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lego)

CONFIG = yaml.safe_load(open(ROOT / 'config' / 'lego.yaml'))
K = np.array([[928.2, 0, 640.9], [0, 928.1, 368.3], [0, 0, 1.0]])
# Ready pose (tool pointing down) and a plausible camera on the hand.
T_BASE_EE = np.eye(4)
T_BASE_EE[:3, :3] = Rotation.from_quat([1.0, 0.003, -0.003, -0.002]).as_matrix()
T_BASE_EE[:3, 3] = [0.307, 0.0, 0.485]
T_EE_CAM = np.eye(4)
T_EE_CAM[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
T_EE_CAM[:3, 3] = [0.052, -0.037, -0.045]
T_BASE_CAM = T_BASE_EE @ T_EE_CAM


def project(point):
    p = np.linalg.inv(T_BASE_CAM) @ np.append(point, 1.0)
    uvw = K @ p[:3]
    return uvw[:2] / uvw[2]


def render(bricks):
    image = np.full((720, 1280, 3), 200, np.uint8)
    top = CONFIG['table_top_z'] + CONFIG['block_size'][2]
    for cx, cy, yaw, length, width in bricks:
        rot = Rotation.from_euler('z', yaw).as_matrix()[:2, :2]
        corners = [np.array([cx, cy]) + rot @ (np.array(s) * [length / 2, width / 2])
                   for s in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        poly = np.array([project(np.array([x, y, top])) for x, y in corners], np.int32)
        cv2.fillPoly(image, [poly], (30, 30, 200))  # BGR red
    return image


def test_recovers_brick_position_and_yaw():
    for cx, cy, yaw_deg in ((0.40, 0.05, 30), (0.45, -0.10, -50), (0.35, 0.12, 5)):
        image = render([(cx, cy, math.radians(yaw_deg), 0.032, 0.016)])
        accepted, rejected = lego.detect_bricks(image, K, T_BASE_CAM, CONFIG)
        assert len(accepted) == 1, (accepted, rejected)
        found = accepted[0]
        assert np.linalg.norm(found['centre'][:2] - [cx, cy]) < 0.002
        diff = (found['yaw'] - math.radians(yaw_deg) + math.pi / 2) % math.pi - math.pi / 2
        assert abs(math.degrees(diff)) < 2.0


def test_rejects_wrong_size_and_out_of_workspace():
    big = render([(0.40, 0.0, 0.0, 0.08, 0.05)])           # e.g. a big red object
    assert not lego.detect_bricks(big, K, T_BASE_CAM, CONFIG)[0]
    outside = render([(0.40, 0.45, 0.0, 0.032, 0.016)])    # y beyond the workspace
    accepted, rejected = lego.detect_bricks(outside, K, T_BASE_CAM, CONFIG)
    assert not accepted


def test_circular_mean_treats_axis_as_line():
    angle = lego.circular_mean_line([math.radians(89), math.radians(-89)])
    assert abs(abs(math.degrees(angle)) - 90) < 1.0
