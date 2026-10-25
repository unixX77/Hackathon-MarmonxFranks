"""Pure-logic checks for smooth_motion: wrist yaw choice and the speed profile."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
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


def test_wrist_has_its_own_swing_limit():
    flat = [0.1] * 7
    assert sm.swing_violation(flat, 1.5, 2.2) is None
    assert sm.swing_violation([0.1] * 6 + [1.6], 1.5, 2.2) is None          # ~90 degree hand yaw is fine
    assert sm.swing_violation([0.1] * 6 + [2.3], 1.5, 2.2)[0] == 7
    joint, value, limit = sm.swing_violation([0.1, 0.1, 1.6, 0.1, 0.1, 0.1, 0.1], 1.5, 2.2)
    assert (joint, limit) == (3, 1.5) and value == 1.6                      # an arm joint is held to the tight limit


def test_wrist_excursion_is_the_largest_joint7_magnitude():
    path = np.zeros((4, 7))
    path[:, 6] = [0.5, 1.9, -2.6, 2.2]
    assert sm.wrist_excursion(path) == 2.6


def test_soft_wrist_limit_leaves_room_before_the_hard_stop_and_unwind_covers_a_half_turn():
    assert sm.WRIST_SOFT_LIMIT < 3.05 - 0.5          # FR3 joint 7 stops at +-3.0508 rad
    assert sm.UNWIND_WRIST_SWING > math.pi           # the equivalent yaw is 180 degrees away


def test_releasing_the_block_also_removes_it_from_the_world(monkeypatch):
    """Detaching puts the block back into the scene as an obstacle; with the hand open to 17 mm that obstacle
    touches a finger and blocks the back-off. It must be removed right after the detach."""
    pytest.importorskip('moveit_msgs')
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    import spawn_scene
    calls = []
    monkeypatch.setattr(spawn_scene, 'apply_scene', lambda node, **kw: calls.append(('world', kw)))
    mover = sm.Mover.__new__(sm.Mover)                        # no ROS node needed: only the scene calls are exercised
    mover.node, mover.c = None, {'ee_frame': 'fr3_hand_tcp'}
    mover.apply_scene = lambda scene: calls.append(('detach', scene))
    mover.detach_block()
    assert [name for name, _ in calls] == ['detach', 'world']                 # detach first, then clear the world copy
    assert calls[1][1]['remove'] == ('fr3_demo_block',)
    assert calls[1][1].get('strict') is False
