"""Smooth, constant-speed playback of the planner's straight-line joint path."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('ts', ROOT / 'scripts' / 'trajectory_smoothing.py')
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

READY = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])


# ------------------------------------------------------------------ a planner like MoveIt's, with its IK noise
def line_poses(n, move, turn_deg=0.0):
    """n flange poses on a straight line from the READY pose: translate by `move` (m), turn about the base z axis."""
    start = ts.fk_flange(READY)[0]
    poses = np.tile(start, (n, 1, 1))
    for i, f in enumerate(np.linspace(0.0, 1.0, n)):
        poses[i, :3, 3] = start[:3, 3] + f * np.asarray(move)
        poses[i, :3, :3] = Rotation.from_euler('z', f * math.radians(turn_deg)).as_matrix() @ start[:3, :3]
    return poses


def planner_like_path(n, move, turn_deg=0.0, wobble=0.01, seed=1):
    """Joint points that put the hand exactly on the line (IK) but let the elbow wobble along the spare degree of
    freedom by up to `wobble` rad from point to point, as the numerical solver does."""
    rng = np.random.default_rng(seed)
    poses = line_poses(n, move, turn_deg)
    q = READY.copy()[None, :]
    points = []
    for pose in poses:
        q, _, _ = ts.project_to_poses(q, pose[None], iterations=20)
        jac = ts.jacobian(q)[0]
        null = np.linalg.svd(jac)[2][-1]                       # the direction in which the joints move and the hand does not
        q = q + rng.uniform(-wobble, wobble) * null[None, :]
        q, err, _ = ts.project_to_poses(q, pose[None], iterations=20)
        assert err < 1e-8
        points.append(q[0])
    return np.array(points), poses


def hand_positions(q):
    return ts.fk_flange(q)[:, :3, 3]


# ------------------------------------------------------------------------------- kinematics
def test_forward_kinematics_matches_moveit():
    """Values from MoveIt's compute_fk (link fr3_link8) on the real robot model, for exact round joint angles."""
    for q, xyz, quat in (([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4], [0.306891, 0.0, 0.590282], None),
                         ([0.5, -0.5, 0.5, -2.0, 0.5, 2.0, 0.5], [0.170393, 0.426516, 0.67564], [0.948056, 0.189533, 0.077382, -0.243472]),
                         ([-1.0, 0.7, -0.4, -1.5, 1.2, 3.0, -1.0], [0.127726, -0.728431, 0.359837], [0.708766, -0.548567, 0.344631, 0.279205])):
        pose = ts.fk_flange(q)[0]
        assert pose[:3, 3] == pytest.approx(xyz, abs=2e-6)
        if quat is not None:
            assert Rotation.from_matrix(pose[:3, :3]).as_quat() == pytest.approx(quat, abs=2e-6)
    assert ts.fk_flange(READY)[0][:3, :3] @ [0, 0, 1] == pytest.approx([0, 0, -1], abs=1e-9)       # the flange points down


def test_projection_returns_the_hand_to_the_planned_poses():
    q, poses = planner_like_path(8, (0.10, 0.04, -0.15), 20.0, wobble=0.02)
    off = q + 0.01                                                         # push every joint a little away
    back, err_pos, err_rot = ts.project_to_poses(off, ts.fk_flange(q))
    assert err_pos < 1e-8 and err_rot < 1e-8
    assert np.abs(back - q).max() < 0.05                                   # and it moved the joints as little as it could


# --------------------------------------------------------------------------------- the time law
def test_speed_profile_matches_the_old_durations():
    v, ramp, total = ts.speed_profile(6.0)                                  # a long move: cruise
    assert (ramp, total) == (1.5, 7.5) and v == pytest.approx(1 / 6.0)
    v, ramp, total = ts.speed_profile(0.3)                                  # a short move: never reaches full speed
    assert total == pytest.approx(2 * math.sqrt(0.3 * 1.5)) and v < 1 / 0.3 and v * ramp == pytest.approx(1.0)


@pytest.mark.parametrize('full_speed_time', [0.2, 1.0, 1.5, 4.0, 12.0])
def test_the_move_covers_exactly_the_path_with_zero_speed_and_acceleration_at_both_ends(full_speed_time):
    v, ramp, total = ts.speed_profile(full_speed_time)
    t = np.linspace(0.0, total, 4001)
    x, xd, xdd = ts.profile_state(t, v, ramp, total)
    assert x[0] == pytest.approx(0.0, abs=1e-12) and x[-1] == pytest.approx(1.0, abs=1e-9)
    assert xd[0] == pytest.approx(0.0, abs=1e-12) and xd[-1] == pytest.approx(0.0, abs=1e-9)
    assert xdd[0] == pytest.approx(0.0, abs=1e-12) and xdd[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(x) >= -1e-12)                                     # never backwards
    assert xd.max() == pytest.approx(v, rel=1e-6)                           # never faster than allowed
    assert np.abs(np.diff(xd) / np.diff(t)).max() < 2 * v / ramp + 1e-9     # no acceleration spike
    assert np.abs(np.diff(xdd) / np.diff(t)).max() < 40 * v / ramp ** 2     # finite jerk (a trapezoid has an infinite one)


# --------------------------------------------------------------------------- the smoothing itself
@pytest.mark.parametrize('move, turn', [((0.0, 0.0, -0.40), 0.0), ((0.10, 0.04, -0.15), 25.0), ((0.0, 0.07, 0.0), 0.0),
                                        ((0.0, 0.0, 0.10), 0.0), ((-0.05, -0.20, 0.0), -30.0)])
def test_smooth_result_for_planner_like_input(move, turn):
    n = max(int(np.linalg.norm(move) / 0.015), 6)
    q, poses = planner_like_path(n, move, turn)
    out = ts.smooth_trajectory(q, 0.05)
    assert out['info']['mode'] == 'smooth', out['info']
    t, qs, vs, acc = out['t'], out['q'], out['v'], out['a']
    # start and end are exactly the planner's, at rest
    assert qs[0] == pytest.approx(q[0], abs=1e-12) and qs[-1] == pytest.approx(q[-1], abs=1e-12)
    assert np.abs(vs[0]).max() == 0 and np.abs(vs[-1]).max() == 0 and np.abs(acc[0]).max() == 0 and np.abs(acc[-1]).max() == 0
    assert np.all(np.diff(t) > 0) and np.diff(t).max() <= 0.0501
    # the hand stays on the planned line (checked at 600 instants, not just at the planned points)
    dense_t = np.linspace(0, t[-1], 600)
    dq, _, _ = out['eval'](dense_t)
    positions = hand_positions(dq)
    start, end = poses[0, :3, 3], poses[-1, :3, 3]
    chord = end - start
    length = np.linalg.norm(chord)
    off_line = np.linalg.norm(np.cross(positions - start, chord / length), axis=1).max()
    assert off_line < 0.0003                                               # 0.3 mm; MoveIt's own points are within 0.05
    # never faster than scale * limit
    assert (np.abs(out['eval'](dense_t)[1]) / (0.05 * ts.VMAX)).max() <= 1.001
    # the planner's wobble is gone: third differences of the joint path
    wobble_in = np.abs(np.diff(q, 3, axis=0)).max()
    spline_q = out['eval'](np.linspace(0, t[-1], 4 * len(q)))[0]
    assert np.abs(np.diff(q, 3, axis=0)).max() > 0.004                     # the input really was jittery
    assert wobble_in > 5 * np.abs(np.diff(spline_q, 3, axis=0)).max()


def test_constant_hand_speed_along_the_line():
    q, poses = planner_like_path(27, (0.10, 0.04, -0.37), 25.0)
    out = ts.smooth_trajectory(q, 0.05)
    assert out['info']['mode'] == 'smooth'
    times = np.linspace(2.0, out['t'][-1] - 2.0, 200)                      # inside the cruise
    positions = hand_positions(out['eval'](times)[0])
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    assert speed.std() / speed.mean() < 0.06                               # +-25 % before; now a few percent (the turn adds a bit)


def test_much_less_jerk_than_the_old_trapezoid_on_the_same_path():
    q, _ = planner_like_path(13, (0.0, 0.0, 0.10), 0.0)
    old_t, old_v = ts.fallback_timing(q, 0.05)
    old = CubicHermiteSpline(old_t, q, old_v, axis=0)                      # what the controller made of the old trajectory
    grid = np.arange(0.0, old_t[-1], 0.001)
    old_jerk = np.abs(np.gradient(old(grid, 2), 0.001, axis=0)).max()
    out = ts.smooth_trajectory(q, 0.05)
    new_grid = np.arange(0.0, out['t'][-1], 0.001)
    new_acc = out['eval'](new_grid)[2]
    new_jerk = np.abs(np.gradient(new_acc, 0.001, axis=0)).max()
    assert new_jerk < old_jerk / 20
    assert np.abs(new_acc).max() < np.abs(old(grid, 2)).max()


def test_the_duration_is_about_what_the_old_one_was():
    q, _ = planner_like_path(13, (0.0, 0.0, 0.10), 0.0)
    old_t, _ = ts.fallback_timing(q, 0.05)
    new_t = ts.smooth_trajectory(q, 0.05)['t']
    assert 0.6 * old_t[-1] < new_t[-1] < 2.0 * old_t[-1]


def test_a_tiny_correction_and_few_points():
    q, _ = planner_like_path(4, (0.0, 0.01, 0.0), 0.0, wobble=0.002)
    out = ts.smooth_trajectory(q, 0.05)
    assert out['info']['mode'] == 'smooth' and out['a'] is not None
    assert out['t'][-1] > 0.5 and np.abs(out['v']).max() < 0.05
    q3, _ = planner_like_path(3, (0.0, 0.005, 0.0), 0.0, wobble=0.0)
    out3 = ts.smooth_trajectory(q3, 0.05)
    assert out3['info']['mode'] == 'smooth', out3['info']
    assert out3['q'][-1] == pytest.approx(q3[-1], abs=1e-12)


def test_it_falls_back_to_the_old_behaviour_instead_of_guessing():
    q, _ = planner_like_path(10, (0.0, 0.0, 0.10), 0.0)
    for bad, reason in ((q[:2], 'planned points'), (np.vstack([q[:5], q[4:5], q[5:]]), 'repeated'),
                        (np.vstack([q[:1]] * 6), 'short'), (np.where(np.arange(70).reshape(10, 7) == 3, np.nan, q), 'unexpected')):
        out = ts.smooth_trajectory(bad, 0.05)
        assert out['info']['mode'] == 'fallback' and reason in out['info']['reason']
        assert out['a'] is None and len(out['t']) == len(bad)               # the planner's own points, as before
    rough = q.copy()
    rough[4] += np.array([0.3, -0.3, 0.3, 0.0, 0.3, 0.0, 0.0])             # a point whose pose is far off the line: not smoothable
    out = ts.smooth_trajectory(rough, 0.05)
    assert out['info']['mode'] == 'fallback'


def test_the_fallback_is_exactly_the_old_retime():
    sm_spec = importlib.util.spec_from_file_location('smooth_motion_for_ts', ROOT / 'scripts' / 'smooth_motion.py')
    sm = importlib.util.module_from_spec(sm_spec)
    sm_spec.loader.exec_module(sm)
    q, _ = planner_like_path(10, (0.0, 0.0, 0.10), 0.0)
    old_t, old_v = sm.retime(q, 0.05)
    new_t, new_v = ts.fallback_timing(q, 0.05)
    assert new_t == pytest.approx(old_t) and new_v == pytest.approx(old_v)
