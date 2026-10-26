"""Smooth, constant-speed playback of the straight-line joint path that MoveIt's Cartesian planner returns.

Why: compute_cartesian_path solves the inverse kinematics point by point with a numerical solver (LMA, 5 ms).
The hand lands on the straight line every time (0.01-0.05 mm), but the 7-axis arm has one spare degree of freedom,
the elbow, and the solver lets it wander by several milliradians from point to point. Joints reverse direction in the
middle of a move, and the velocities worked out from that noise make the arm jerk and the hand speed vary by +-25 %.

What this does (all checked before it is used; anything that does not hold falls back to the old behaviour):
  1. Keep the hand EXACTLY on the planned poses: take the planned poses (forward kinematics of the planned joint
     points), smooth the joint path with a low-order polynomial, and pull it back onto those poses with a few
     damped least-squares steps. Only the elbow wobble changes; the pose of the hand does not.
  2. Interpolate those points with a quintic spline (continuous velocity AND acceleration).
  3. Move along the path at ONE constant speed (the fastest that keeps every joint under scale * its limit
     at the worst point of the path), with quintic ramps at both ends: speed and acceleration start and end at zero
     and jerk stays finite.
  4. Send positions, velocities and accelerations every 50 ms, so the controller follows a quintic spline.

No ROS here: pure numpy / scipy, unit tested.
"""
import math

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

# FR3 joint speed limits (rad/s). Moves run at a small fraction of these.
VMAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
RAMP_SECONDS = 1.5

# FR3 kinematics (Franka's modified DH table; the FR3 has the same link parameters as the Panda).
DH_A = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])
DH_D = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0])
DH_ALPHA = np.array([0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2])
FLANGE_D = 0.107
ROTATION_LEVER = 0.1        # m: one radian of hand rotation counts like this much path length


# ----------------------------------------------------------------------------- kinematics
def fk_flange(q):
    """Pose of the flange (link 8) in the base frame for joint angles q (N x 7 or 7): an (N, 4, 4) array."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    n = len(q)
    transform = np.tile(np.eye(4), (n, 1, 1))
    for i in range(7):
        ct, st = np.cos(q[:, i]), np.sin(q[:, i])
        ca, sa = math.cos(DH_ALPHA[i]), math.sin(DH_ALPHA[i])
        link = np.zeros((n, 4, 4))
        link[:, 0, 0], link[:, 0, 1], link[:, 0, 3] = ct, -st, DH_A[i]
        link[:, 1, 0], link[:, 1, 1], link[:, 1, 2], link[:, 1, 3] = st * ca, ct * ca, -sa, -DH_D[i] * sa
        link[:, 2, 0], link[:, 2, 1], link[:, 2, 2], link[:, 2, 3] = st * sa, ct * sa, ca, DH_D[i] * ca
        link[:, 3, 3] = 1.0
        transform = transform @ link
    flange = np.eye(4)
    flange[2, 3] = FLANGE_D
    return transform @ flange


def _rotation_error(target, current):
    """Rotation vector that turns `current` into `target` (both (N, 3, 3))."""
    return Rotation.from_matrix(target @ current.transpose(0, 2, 1)).as_rotvec()


def pose_error(t_target, t_current):
    """(N, 6): position error and rotation-vector error that take `t_current` to `t_target`."""
    return np.concatenate([t_target[:, :3, 3] - t_current[:, :3, 3],
                           _rotation_error(t_target[:, :3, :3], t_current[:, :3, :3])], axis=1)


def jacobian(q, eps=1e-6):
    """(N, 6, 7) Jacobian of the flange pose (position, rotation vector) by finite differences."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    base = fk_flange(q)
    jac = np.zeros((len(q), 6, 7))
    for j in range(7):
        moved = q.copy()
        moved[:, j] += eps
        jac[:, :, j] = pose_error(fk_flange(moved), base) / eps
    return jac


def project_to_poses(q_guess, t_target, iterations=5, damping=1e-3):
    """Move q_guess (N x 7) the least possible onto the poses t_target (N, 4, 4): damped least squares, one
    step per iteration. Returns (q, worst position error m, worst rotation error rad)."""
    q = np.array(q_guess, dtype=float)
    eye = np.eye(6)
    for _ in range(iterations):
        err = pose_error(t_target, fk_flange(q))
        if np.abs(err[:, :3]).max() < 1e-9 and np.abs(err[:, 3:]).max() < 1e-9:
            break
        jac = jacobian(q)
        jjt = jac @ jac.transpose(0, 2, 1) + damping ** 2 * eye
        step = np.einsum('nij,nj->ni', jac.transpose(0, 2, 1), np.linalg.solve(jjt, err[..., None])[..., 0])
        q += step
    err = pose_error(t_target, fk_flange(q))
    return q, float(np.linalg.norm(err[:, :3], axis=1).max()), float(np.linalg.norm(err[:, 3:], axis=1).max())


def path_parameter(poses):
    """(x in [0, 1], total length) along planned poses: translation plus ROTATION_LEVER * rotation angle."""
    step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    turn = Rotation.from_matrix(poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1)).magnitude()
    s = np.concatenate([[0.0], np.cumsum(step + ROTATION_LEVER * turn)])
    return (s / s[-1] if s[-1] > 0 else s), float(s[-1])


# ------------------------------------------------------------------------------ time law
def smootherstep(tau):
    """h(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 and its integral and first derivative: a velocity ramp 0 -> 1 with zero
    acceleration at both ends."""
    tau = np.clip(tau, 0.0, 1.0)
    return (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5,
            2.5 * tau ** 4 - 3 * tau ** 5 + tau ** 6,
            30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4)


def speed_profile(full_speed_time, ramp=RAMP_SECONDS):
    """The plan for moving once along a path of length 1 when the fastest allowed speed would take
    `full_speed_time` seconds. Returns (peak rate v, ramp time, total duration). Same durations as the old trapezoid:
    long moves cruise at the full speed (duration = full_speed_time + ramp); short ones never reach it
    (duration = 2 sqrt(full_speed_time * ramp))."""
    if full_speed_time >= ramp:
        return 1.0 / full_speed_time, ramp, full_speed_time + ramp
    ramp_eff = math.sqrt(full_speed_time * ramp)
    return 1.0 / ramp_eff, ramp_eff, 2 * ramp_eff


def profile_state(t, v, ramp, total):
    """Position x in [0, 1] along the path, its rate and its acceleration at times t (arrays)."""
    t = np.asarray(t, dtype=float)
    x, xd, xdd = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    up = t < ramp
    down = t > total - ramp
    middle = ~up & ~down
    h, g, hp = smootherstep(t[up] / ramp)
    x[up], xd[up], xdd[up] = v * ramp * g, v * h, v * hp / ramp
    x[middle], xd[middle] = v * ramp / 2 + v * (t[middle] - ramp), v
    h, g, hp = smootherstep((total - t[down]) / ramp)
    x[down], xd[down], xdd[down] = 1.0 - v * ramp * g, v * h, -v * hp / ramp
    return x, xd, xdd


def fallback_timing(q, scale, ramp=RAMP_SECONDS):
    """The old behaviour: the planner's own points with a trapezoid speed profile (velocities only)."""
    q = np.asarray(q, dtype=float)
    if len(q) < 2:
        return np.zeros(len(q)), np.zeros_like(q)
    step = np.max(np.abs(np.diff(q, axis=0)) / (VMAX * scale), axis=1)
    u = np.concatenate([[0.0], np.cumsum(step)])
    total = u[-1]
    if total < 1e-9:
        return np.zeros(len(q)), np.zeros_like(q)
    if total >= ramp:
        end = total + ramp
        t = np.where(u <= ramp / 2, np.sqrt(2 * u * ramp),
                     np.where(u >= total - ramp / 2, end - np.sqrt(np.maximum(2 * (total - u) * ramp, 0)),
                              ramp + (u - ramp / 2)))
    else:
        end = 2 * math.sqrt(total * ramp)
        t = np.where(u <= total / 2, np.sqrt(2 * u * ramp), end - np.sqrt(np.maximum(2 * (total - u) * ramp, 0)))
    t = np.maximum.accumulate(t)
    t = t + np.arange(len(t)) * 1e-4
    v = np.zeros_like(q)
    v[1:-1] = (q[2:] - q[:-2]) / (t[2:] - t[:-2])[:, None]
    return t, v


# ------------------------------------------------------------------------------- main entry
def smooth_trajectory(q, scale, ramp=RAMP_SECONDS, dt=0.05, vmax=VMAX, max_line_error=0.0005, max_turn_error=0.0006):
    """Turn the planner's joint points q (N x 7, a straight Cartesian line) into a smooth timed trajectory.

    Returns a dict: t (K,), q (K, 7), v (K, 7), a (K, 7) or None (a is None in fallback), info (dict with 'mode' =
    'smooth' or 'fallback' and 'reason', plus the measured 'line_error_m', 'jitter_removed_rad', 'duration'), and
    'eval' (a function t -> (q, v, a) of the smooth trajectory, only when smooth).
    Fallback means: the old planner points with the old trapezoid timing, because the smoothing could not be shown
    to keep the hand on the line."""
    q = np.asarray(q, dtype=float)

    def fallback(reason, trivial=False):
        t, v = fallback_timing(q, scale, ramp)
        return {'t': t, 'q': q, 'v': v, 'a': None, 'info': {'mode': 'fallback', 'reason': reason, 'trivial': trivial}}

    n = len(q)
    if n < 3:
        return fallback(f'only {n} planned points', trivial=True)
    if q.shape[1] != 7 or not np.all(np.isfinite(q)):
        return fallback('unexpected joint data')
    poses = fk_flange(q)
    x, length = path_parameter(poses)
    if length < 5e-4 or np.any(np.diff(x) < 1e-6):
        return fallback('path too short or with repeated points', trivial=length < 5e-4)

    # 1. a smooth guess of the joint path, pinned to the exact start and end
    degree = max(1, min(6, n - 2))
    coef = np.polynomial.chebyshev.chebfit(2 * x - 1, q, degree)
    guess = np.polynomial.chebyshev.chebval(2 * x - 1, coef).T
    guess += (1 - x)[:, None] * (q[0] - guess[0]) + x[:, None] * (q[-1] - guess[-1])
    # 2. back onto the planned poses of the hand
    smooth_q, err_pos, err_rot = project_to_poses(guess, poses)
    if err_pos > 1e-6 or err_rot > 1e-5:
        return fallback(f'could not return to the planned poses ({err_pos * 1000:.3f} mm)')
    smooth_q[0], smooth_q[-1] = q[0], q[-1]
    jitter = float(np.abs(smooth_q - q).max())
    if jitter > 0.15:
        return fallback(f'smoothing would move a joint by {jitter:.2f} rad')
    # 3. a quintic spline through them
    try:
        spline = make_interp_spline(x, smooth_q, k=min(5, n - 1) if n - 1 != 2 else 2)
    except ValueError as problem:
        return fallback(f'spline failed: {problem}')
    d1, d2 = spline.derivative(1), spline.derivative(2)
    grid = np.linspace(0.0, 1.0, 801)
    slope = np.abs(d1(grid))
    rate_limit = scale * float(np.min(vmax[None, :] / np.maximum(slope, 1e-9)))       # path fractions per second
    if not rate_limit > 0:
        return fallback('no usable speed limit')
    v_peak, ramp_eff, total = speed_profile(1.0 / rate_limit, ramp)
    count = max(int(math.ceil(total / dt)), 2)
    t = np.linspace(0.0, total, count + 1)
    xs, xd, xdd = profile_state(t, v_peak, ramp_eff, total)
    xs = np.clip(xs, 0.0, 1.0)
    qs = spline(xs)
    vs = d1(xs) * xd[:, None]
    accs = d2(xs) * (xd ** 2)[:, None] + d1(xs) * xdd[:, None]
    qs[0], qs[-1] = q[0], q[-1]
    vs[0] = vs[-1] = 0.0
    accs[0] = accs[-1] = 0.0

    # 4. prove it before using it: the hand stays on the planned line, the joints stay in range and under the limits
    mid = np.concatenate([x[:-1, None] + f * np.diff(x)[:, None] for f in (0.25, 0.5, 0.75)], axis=0).ravel()
    mid_poses = fk_flange(spline(mid))
    ideal_position, ideal_rotation = np.empty((len(mid), 3)), np.empty((len(mid), 3, 3))
    for k, xm in enumerate(mid):
        i = min(int(np.searchsorted(x, xm, side='right')) - 1, n - 2)
        f = (xm - x[i]) / (x[i + 1] - x[i])
        ideal_position[k] = (1 - f) * poses[i, :3, 3] + f * poses[i + 1, :3, 3]
        rel = Rotation.from_matrix(poses[i, :3, :3].T @ poses[i + 1, :3, :3]).as_rotvec()
        ideal_rotation[k] = poses[i, :3, :3] @ Rotation.from_rotvec(f * rel).as_matrix()
    line_error = float(np.linalg.norm(mid_poses[:, :3, 3] - ideal_position, axis=1).max())
    turn_error = float(np.linalg.norm(_rotation_error(ideal_rotation, mid_poses[:, :3, :3]), axis=1).max())
    if line_error > max_line_error or turn_error > max_turn_error:
        return fallback(f'the smoothed path leaves the line by {line_error * 1000:.2f} mm / {turn_error * 1000:.2f} mrad')
    margin = 0.03
    if np.any(qs < q.min(axis=0) - margin) or np.any(qs > q.max(axis=0) + margin):
        return fallback('the smoothed path leaves the range of the planned joint angles')
    speed_ratio = float((np.abs(vs) / (scale * vmax)).max())
    if speed_ratio > 1.02:
        return fallback(f'speed {speed_ratio:.2f} times over the limit')
    info = {'mode': 'smooth', 'duration': float(total), 'line_error_m': line_error, 'turn_error_rad': turn_error,
            'jitter_removed_rad': jitter, 'path_length_m': length, 'speed_m_s': length * v_peak,
            'peak_acceleration': float(np.abs(accs).max())}

    def evaluate(times):
        ts = np.asarray(times, dtype=float)
        xe, xde, xdde = profile_state(np.clip(ts, 0.0, total), v_peak, ramp_eff, total)
        xe = np.clip(xe, 0.0, 1.0)
        return spline(xe), d1(xe) * xde[:, None], d2(xe) * (xde ** 2)[:, None] + d1(xe) * xdde[:, None]
    return {'t': t, 'q': qs, 'v': vs, 'a': accs, 'info': info, 'eval': evaluate}
