"""Offline-testable single-brick policy and nominal URDF Jacobian checks."""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

ARM = [f'fr3_joint{i}' for i in range(1, 8)]


class PickFailure(RuntimeError):
    """Request failed; process can remain available."""


class Intervention(PickFailure):
    """No automatic movement allowed after uncertain execution/contact."""


class DescentStopped(PickFailure):
    """Cancellation/completion and a settled state have been confirmed."""


@dataclass(frozen=True)
class Policy:
    ik_attempts: int = 8
    max_candidates: int = 4
    corrections: int = 2
    descent_retries: int = 1
    joint_margin_rad: float = .03
    min_sigma: float = .025
    jacobian_length_m: float = .3
    max_joint_travel_rad: float = 2.5
    correction_xyz_m: float = .005
    correction_angle_deg: float = 1.
    monitor_xy_m: float = .003
    monitor_angle_deg: float = 1.
    monitor_joint_error_rad: float = .03
    monitor_period_s: float = .1
    motion_timeout_s: float = 120.
    relative_jump_factor: float = 2.

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            raise ValueError('pick policy must be a mapping')
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown pick policy options: {sorted(unknown)}')
        result = cls(**config)
        for key in ('ik_attempts', 'max_candidates', 'corrections', 'descent_retries'):
            v = getattr(result, key)
            low = 1 if key in ('ik_attempts', 'max_candidates') else 0
            if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= 12:
                raise ValueError(f'{key} must be an integer in [{low},12]')
        for key in set(cls.__dataclass_fields__) - {'ik_attempts', 'max_candidates', 'corrections', 'descent_retries'}:
            v = getattr(result, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 1. <= result.relative_jump_factor <= 10.:
            raise ValueError('relative_jump_factor must be in [1,10]; absolute joint-step checks remain enabled')
        return result


def angle_deg(a, b):
    return math.degrees(2*math.acos(min(1., abs(sum(x*y for x, y in zip(a, b))))))


class PickWorkflow:
    """Adapter performs planning/motion; policy never silently ignores errors."""
    def __init__(self, adapter, policy):
        self.a, self.p = adapter, policy

    def align(self, target):
        previous = float('inf')
        for attempt in range(self.p.corrections + 1):
            error = self.a.verify(target)
            if error['accepted']:
                return
            if (error['xyz_m'] > self.p.correction_xyz_m
                    or error['angle_deg'] > self.p.correction_angle_deg
                    or error['joint_rad'] > .01):
                raise PickFailure('APPROACH_ALIGNMENT_FAILED: outside correction envelope')
            score = error['xyz_m']/self.p.correction_xyz_m + error['angle_deg']/self.p.correction_angle_deg
            if score >= previous - 1e-3 or attempt == self.p.corrections:
                raise PickFailure('APPROACH_ALIGNMENT_FAILED: correction budget exhausted or no improvement')
            previous = score
            self.a.status('CORRECTING_APPROACH', f'Correcting approach ({attempt+1}/{self.p.corrections})')
            plan = self.a.find_approach(target)
            self.a.move_arm(plan, 'Approach correction')

    def run(self):
        self.a.status('FINDING_APPROACH', 'Finding an approach and checking descent feasibility')
        above = self.a.pregrasp()
        plan = self.a.find_approach(above)
        self.a.open_hand()
        self.a.status('APPROACHING', 'Moving to pre-grasp')
        self.a.move_arm(plan, 'Approach')
        self.align(above)
        for attempt in range(self.p.descent_retries + 1):
            self.a.status('PLANNING_DESCENT', 'Planning descent from measured joints')
            # No movement has occurred if this planning call fails.
            try:
                descent = self.a.plan_descent()
            except Intervention:
                raise
            except PickFailure:
                if attempt == self.p.descent_retries:
                    raise
                self.a.reject_approach()
                self.a.status('RESELECTING_APPROACH', 'Descent planning failed; finding another approach')
                self.a.move_arm(self.a.find_approach(above), 'Alternative approach')
                self.align(above)
                continue
            try:
                self.a.status('DESCENDING', 'Descending to the grasp pose')
                self.a.move_arm(descent, 'Descent', vertical=True)
                error = self.a.verify(self.a.grasp_target())
                if not error['accepted']:
                    raise DescentStopped('Grasp pose is outside acceptance tolerance')
            except DescentStopped as exc:
                self.a.status('RECOVERING', f'Recovering from failed descent: {exc}')
                self.a.recover(above)
                if attempt == self.p.descent_retries:
                    raise PickFailure('DESCENT_FAILED: returned to pre-grasp; retry budget exhausted') from exc
                self.a.reject_approach()
                self.a.status('RESELECTING_APPROACH', 'Retreated; checking another approach')
                self.a.move_arm(self.a.find_approach(above), 'Alternative approach')
                self.align(above)
                continue
            self.a.status('GRASPING', 'Grasp pose verified; closing gripper')
            self.a.close_hand()
            if not getattr(self.a, 'hardware', True):
                self.a.status('PREVIEW_COMPLETE', 'Approach, descent and grasp preview complete; no hardware commands sent')
                return 'PREVIEW_COMPLETE'
            self.a.status('HOLDING_OBJECT', 'Grasp action succeeded; stopped holding brick, without lifting')
            return 'HOLDING_OBJECT'
        raise PickFailure('No descent completed')


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


class Kinematics:
    """Geometric TCP Jacobian from the loaded URDF; no hardware-specific metric."""
    def __init__(self, urdf, base='fr3_link0', tcp='fr3_hand_tcp'):
        root = ET.fromstring(urdf)
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, link, seen = [], tcp, set()
        while link != base:
            if link in seen or link not in by_child:
                raise ValueError('URDF has no valid base-to-TCP chain')
            seen.add(link)
            j = by_child[link]
            chain.append(j)
            link = j.find('parent').get('link')
        self.chain = []
        active = []
        for j in reversed(chain):
            kind, name = j.get('type'), j.get('name')
            if kind not in ('fixed', 'revolute'):
                raise ValueError(f'Unsupported chain joint type: {kind}')
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            if len(xyz) != 3 or len(rpy) != 3 or not np.isfinite(xyz).all() or not np.isfinite(rpy).all():
                raise ValueError('Invalid URDF origin')
            t = np.eye(4)
            t[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
            t[:3, 3] = xyz
            axis_node = j.find('axis')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0') if axis_node is not None else '1 0 0', sep=' ')
            if len(axis) != 3 or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
                raise ValueError('Invalid URDF axis')
            axis /= np.linalg.norm(axis)
            self.chain.append((name, kind, t, axis))
            if kind != 'fixed':
                active.append(name)
        if set(active) != set(ARM) or len(active) != 7:
            raise ValueError('Expected seven FR3 arm joints in TCP chain')

    def evaluate(self, joints, length=.3):
        t = np.eye(4)
        axes = {}
        for name, kind, origin, axis in self.chain:
            t = t @ origin
            if kind != 'fixed':
                axes[name] = (t[:3, 3].copy(), t[:3, :3] @ axis)
                r = np.eye(4)
                r[:3, :3] = rotation(axis, joints[name])
                t = t @ r
        jac = np.empty((6, 7))
        for i, name in enumerate(ARM):
            position, axis = axes[name]
            jac[:3, i] = np.cross(axis, t[:3, 3]-position)/length
            jac[3:, i] = axis
        values = np.linalg.svd(jac, compute_uv=False)
        return t, jac, float(values[-1])


def seeds(current, bounds, count):
    """Deterministic varied seeds, clipped inside soft joint limits."""
    offsets = [(0, 0., 0., 0.), (1, .25, -.25, .2), (-1, -.25, .25, -.2),
               (1, -.3, .3, -.25), (-1, .3, -.3, .25), (1, .45, -.15, -.3),
               (-1, -.45, .15, .3), (1, -.15, -.4, .35), (-1, .15, .4, -.35),
               (1, .4, .2, -.4), (-1, -.4, -.2, .4), (1, -.4, -.2, .4)]
    for direction, shoulder, elbow, wrist in offsets[:count]:
        q = dict(current)
        for name, delta in zip(ARM, (direction*.25, shoulder, -direction*.25, elbow, direction*.2, 0., wrist)):
            low, high = bounds[name]
            q[name] = min(high-.001, max(low+.001, current[name]+delta))
        yield q
