"""Pure-logic checks for smooth_motion: wrist yaw choice and the speed profile."""
import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sm', ROOT / 'scripts' / 'smooth_motion.py')
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def test_quaternion_matches_rz_rx_and_yaw_roundtrip():
    for yaw in (-1.2, 0.0, 0.7, 1.5):
        x, y, z, w = sm.tool_down_quaternion(yaw)
        expected = Rotation.from_euler('z', yaw) * Rotation.from_euler('x', math.pi)
        assert np.allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), expected.as_matrix(), atol=1e-9)
        assert abs(sm.hand_yaw(expected.as_matrix()) - yaw) < 1e-9


def test_nearest_yaw_never_turns_more_than_a_quarter_turn():
    rng = np.random.default_rng(1)
    for _ in range(200):
        current = rng.uniform(-math.pi, math.pi)
        axis = rng.uniform(-math.pi / 2, math.pi / 2)
        chosen = sm.nearest_yaw(axis, current)
        assert abs(chosen - current) <= math.pi / 2 + 1e-9
        assert abs(((chosen - axis) / math.pi) - round((chosen - axis) / math.pi)) < 1e-9  # same grasp


def test_old_behaviour_would_have_spun_the_wrist():
    # hand at yaw +80 deg, brick axis -80 deg: the old code turned 160 deg, the new one 20 deg
    chosen = sm.nearest_yaw(math.radians(-80), math.radians(80))
    assert abs(math.degrees(chosen - math.radians(80))) < 25


def test_retime_is_smooth_and_respects_speed_limit():
    q = np.linspace(np.zeros(7), np.array([0.3, -0.2, 0.1, 0.0, 0.25, 0.1, 0.4]), 80)
    t, v = sm.retime(q, 0.1)
    assert np.all(np.diff(t) > 0)
    assert np.allclose(v[0], 0) and np.allclose(v[-1], 0)             # starts and stops at rest
    assert np.all(np.abs(v) <= sm.VMAX * 0.1 * 1.05)                   # never above the cap
    mid = len(t) // 2
    assert np.abs(v[mid]).max() > 0.5 * (sm.VMAX * 0.1).max() * 0.3    # actually moves


def test_retime_short_move_still_valid():
    q = np.linspace(np.zeros(7), np.full(7, 0.002), 5)
    t, v = sm.retime(q, 0.05)
    assert np.all(np.diff(t) > 0) and np.isfinite(v).all()
