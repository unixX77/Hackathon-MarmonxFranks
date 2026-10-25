"""Inspect DisplayTrajectory dictionaries without ROS or hardware access."""
import hashlib
import copy
import json
import math
from bisect import bisect_right


class PreviewSampler:
    """Sample saved joint positions for visualization at an elapsed wall time.

    Uses cubic Hermite interpolation when waypoint velocities are present,
    otherwise linear interpolation. This is a display sampler, not a controller.
    """
    def __init__(self, display):
        audit_display(display)
        if len(display['trajectory']) != 1:
            raise ValueError('State playback requires exactly one trajectory')
        self.display, _ = align_display_start(display)
        joint = self.display['trajectory'][0]['joint_trajectory']
        self.names = joint['joint_names']
        self.points = joint['points']
        self.times = [p['time_from_start']['sec'] + p['time_from_start']['nanosec'] / 1e9
                      for p in self.points]
        self.duration = self.times[-1]

    def state(self, elapsed):
        if not math.isfinite(elapsed):
            raise ValueError('Playback time must be finite')
        if elapsed <= self.times[0]:
            values = self.points[0]['positions']
        elif elapsed >= self.duration:
            values = self.points[-1]['positions']
        else:
            index = bisect_right(self.times, elapsed) - 1
            a, b = self.points[index:index + 2]
            dt = self.times[index + 1] - self.times[index]
            u = (elapsed - self.times[index]) / dt
            if a.get('velocities') and b.get('velocities'):
                values = [(2*u**3 - 3*u**2 + 1)*qa + (u**3 - 2*u**2 + u)*dt*va
                          + (-2*u**3 + 3*u**2)*qb + (u**3 - u**2)*dt*vb
                          for qa, qb, va, vb in zip(a['positions'], b['positions'],
                                                   a['velocities'], b['velocities'])]
            else:
                values = [(1-u)*qa + u*qb for qa, qb in zip(a['positions'], b['positions'])]
        state = copy.deepcopy(self.display['trajectory_start'])
        joint = state['joint_state']
        for name, value in zip(self.names, values):
            joint['position'][joint['name'].index(name)] = value
        return state


def align_display_start(display):
    """Copy a preview and align its start joints with its first waypoint.

    GetCartesianPath can return its final state in response.start_state. Keep
    unplanned joints (such as fingers), but use the path's actual first arm pose.
    This changes visualization metadata only, never the trajectory.
    """
    result = copy.deepcopy(display)
    joint = result['trajectory'][0]['joint_trajectory']
    state = result.setdefault('trajectory_start', {})
    measured = state.setdefault('joint_state', {})
    names = measured.setdefault('name', [])
    positions = measured.setdefault('position', [])
    if len(names) != len(positions):
        raise ValueError('Preview start joint names and positions differ in length')
    largest = 0.
    for name, value in zip(joint['joint_names'], joint['points'][0]['positions']):
        if name in names:
            index = names.index(name)
            largest = max(largest, abs(positions[index] - value))
            positions[index] = value
        else:
            names.append(name)
            positions.append(value)
    measured['velocity'] = []
    measured['effort'] = []
    state['is_diff'] = False
    return result, largest


def audit_display(display):
    reports = []
    for trajectory in display['trajectory']:
        joint = trajectory['joint_trajectory']
        names, points = joint['joint_names'], joint['points']
        if not names or len(set(names)) != len(names) or len(points) < 2:
            raise ValueError('Preview needs unique joints and at least two points')
        stamps = []
        for point in points:
            stamp = point['time_from_start']
            sec, ns = stamp['sec'], stamp['nanosec']
            if not isinstance(sec, int) or not isinstance(ns, int) or sec < 0 or not 0 <= ns < 10**9:
                raise ValueError('Invalid ROS duration')
            stamps.append(sec * 10**9 + ns)
            for field in ('positions', 'velocities', 'accelerations', 'effort'):
                values = point.get(field, [])
                if (field == 'positions' or values) and len(values) != len(names):
                    raise ValueError(f'Incorrect {field} length')
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f'Nonfinite {field}')
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        if min(gaps) <= 0:
            raise ValueError('Preview timestamps must strictly increase')
        reports.append(dict(point_count=len(points), first_stamp_s=stamps[0] / 1e9,
                            last_stamp_s=stamps[-1] / 1e9,
                            min_segment_s=min(gaps) / 1e9, max_segment_s=max(gaps) / 1e9,
                            fixed_005s_nominal_duration_s=len(points) * .05))
    if not reports:
        raise ValueError('Empty display trajectory')
    digest = hashlib.sha256(json.dumps(display, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    return dict(display_sha256=digest, trajectories=reports)
