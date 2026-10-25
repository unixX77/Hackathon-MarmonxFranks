#!/usr/bin/python3 -s
"""Pose-driven pick and place. No camera, no detection: you give the poses, the arm does the rest.

A teammate gives the brick pose (and where it should go); this script moves there one small safe
step at a time. Nothing here looks for the brick, so the poses you type are trusted, and the
inspection stop is where you check them against the real brick and correct them.

  0. CHECK   the hand must already point down (else it refuses: move to a ready pose first)
  1. OPEN    open the hand
  2. HOVER   straight line to above the pick pose (clearance above the grasp height)
  3. INSPECT straight down to a few cm above the grasp height and STOP. Look at the fingers
             against the brick, then type small corrections (x, y, yaw) until it is right.
  4. LOWER   straight down to the grasp height
  5. GRASP   close the hand (on a real robot the hand reports whether it closed on something)
  6. LIFT    straight back up. Without --place the brick stays in the hand.
  With --place X Y [Z] [--place-yaw DEG] it goes on:
  7. CARRY   straight line at hover height to above the place pose
  8. LOWER   straight down until the brick rests on the table (or at Z), place_margin above it
  9. RELEASE open the hand, then back off straight up
The correction you needed is printed and appended to the offsets log (offsets_log in the config),
so you can tell your teammate what the pose was off by.

Poses are in fr3_link0 (the robot base), metres; +x away from the robot, +y to its left:
  --pose X Y [Z]       brick CENTRE to pick (Z left out = the brick rests on the table)
  --yaw DEG | --quat X Y Z W   brick long-side angle from base X (only the yaw of a quaternion is used)
  --place X Y [Z]      brick CENTRE to put it down (Z left out = on the table; give Z to stack)
  --place-offset DX DY put it down this far from where it was picked (base frame, metres), e.g.
                       0 0.05 = 5 cm to the robot's left. It follows any correction you make at the
                       inspection stop. Height: on the table.
  --place-yaw DEG      long-side angle of the placed brick (default: the angle it was picked at)
Stacks (a brick lying on other bricks; the brick height in the config is one layer):
  --layers N           the brick to pick is in layer N (1 = on the table, 2 = on top of one brick, ...). With a
                       camera point the measured height must agree with that layer (table_check_tolerance) or it
                       refuses; the grasp then uses the exact layer height, not the noisy measured one
  --layers auto        work the layer out from the camera height; refuses when the height is between two layers
  --place-layers N     put the brick down as layer N (works with --place, --place-offset, --place-camera-xyz)
  --pose-is-tcp / --place-is-tcp   the given Z is the hand (TCP) height, not the brick centre

or the same poses given in the CAMERA frame (camera_color_optical_frame: x right, y down, z forward):
  --camera-xyz X Y Z          the point at the centre of the brick's TOP face, to pick
  --camera-quat X Y Z W       its orientation (optional; else use --yaw). Only the direction of the
                              long side (--camera-long-axis x|y, default x) on the table is used
  --place-camera-xyz X Y Z    the same for where the brick should end up (top-face centre there)
  --place-camera-quat X Y Z W
  --ee-pose X Y Z QX QY QZ QW the hand (fr3_hand_tcp) pose in fr3_link0 when the picture was taken.
                              Default: the arm's pose right now, so the arm must not have moved since.
Conversion: p_base = T_base_hand @ T_hand_camera @ p_camera, with T_hand_camera from calibration_file
(your hand-eye result). The converted top face must agree with the table model (table_check_tolerance)
or it refuses (--use-measured-z: trust the measured height instead). Nothing is detected here.

Correction commands at the inspection stop:
  x 0.002   y -0.001   yaw 5      any combination on one line; metres / degrees, small steps only
  ok = continue to lower and grasp      q = stop here (nothing is released or closed)

Every arm move is a straight, collision-checked Cartesian line (smooth_motion.py, through MoveIt) at
low speed, and after each move the real hand position is compared with the target. Real robot: you
type GO and STEP MODE is on (Enter = do it, a = rest automatically, q = stop). The hand is never
opened automatically after an error. The pick pose is drawn in RViz as a green box and the place
pose as a blue box (/lego_estimate_marker).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ------------------------------------------------------------------ pure logic (no ROS)
def yaw_from_quaternion(q):
    """Long-axis yaw (radians, in (-pi/2, pi/2]) and tilt (degrees) of an xyzw quaternion."""
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise ValueError('Quaternion is all zeros')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    return wrap_line_angle(yaw), tilt


def wrap_line_angle(angle):
    """The brick's long axis repeats every pi: report it in (-pi/2, pi/2]."""
    wrapped = (angle + math.pi / 2) % math.pi - math.pi / 2
    return math.pi / 2 if wrapped <= -math.pi / 2 + 1e-12 else wrapped


def hand_tilt_deg(rotation):
    """Angle between the tool Z axis and straight down (0 = hand pointing down)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(rotation[2, 2])))))


def plan_pick(c, x, y, z=None, yaw=0.0, pose_is_tcp=False):
    """Everything the mission needs for one brick pose. Raises ValueError for a bad pose."""
    top, height = c['table_top_z'], c['block_size'][2]
    offset = c['grasp_tcp_height'] - height / 2        # TCP sits this far above the brick centre
    if pose_is_tcp and z is None:
        raise ValueError('--pose-is-tcp / --place-is-tcp needs a z value')
    if z is None:
        centre_z = top + height / 2
    elif pose_is_tcp:
        centre_z = z - offset
    else:
        centre_z = z
    grip_z = centre_z + offset
    for name, value, (low, high) in (('x', x, c['workspace_x']), ('y', y, c['workspace_y'])):
        if not low <= value <= high:
            raise ValueError(f'{name}={value:.3f} is outside the allowed window [{low}, {high}] '
                             f'(workspace_{name} in the config)')
    if centre_z - height / 2 < top - 0.002:
        raise ValueError(f'Brick centre z={centre_z:.4f} puts the brick {1000 * (top - (centre_z - height / 2)):.0f} mm '
                         f'below the table top ({top:.3f})')
    if grip_z < top + c['min_tcp_above_table']:
        raise ValueError(f'Grasp height z={grip_z:.4f} would put the fingertips in the table '
                         f'(needs at least {top + c["min_tcp_above_table"]:.4f})')
    if grip_z + c['clearance'] > c['max_hover_z']:
        raise ValueError(f'Hover height {grip_z + c["clearance"]:.3f} is above max_hover_z {c["max_hover_z"]}')
    inspect = c.get('inspect_height', 0.04)
    if not 0 < inspect < c['clearance']:
        raise ValueError('inspect_height must be above the grasp and below the hover clearance')
    return {
        'centre': [float(x), float(y), float(centre_z)],
        'grip': [float(x), float(y), float(grip_z)],
        'inspect': [float(x), float(y), float(grip_z + inspect)],
        'above': [float(x), float(y), float(grip_z + c['clearance'])],
        'yaw': float(wrap_line_angle(yaw)),
        'block_in_tcp': [0.0, 0.0, float(offset)],      # tool Z points down
    }


def plan_place(c, x, y, yaw, z=None, pose_is_tcp=False):
    """Where the carried brick is put down. Same rules as a pick pose (workspace window, table, fingertips);
    the hand stops place_margin above the height at which the brick rests at the given pose."""
    plan = plan_pick(c, x, y, z, yaw, pose_is_tcp)
    margin = float(c.get('place_margin', 0.0))
    if margin < 0:
        raise ValueError('place_margin must not be negative: it would push the brick into what it rests on')
    plan['release'] = [plan['grip'][0], plan['grip'][1], plan['grip'][2] + margin]
    return plan


def layer_centre_z(c, layers):
    """Height of the centre of a brick lying in layer `layers` of a stack (1 = on the table)."""
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError('layers must be a whole number, 1 or more (1 = on the table)')
    return c['table_top_z'] + (layers - 0.5) * c['block_size'][2]


def layer_top_z(c, layers):
    """Height of the top face of a brick lying in layer `layers`."""
    return layer_centre_z(c, layers) + c['block_size'][2] / 2


def nearest_layer(c, top_z):
    """For a measured top-face height: (nearest layer, signed distance in m from that layer's top face, layers
    above the table as a fraction: 1.0 = a brick on the table, 2.0 = one on top of another)."""
    levels = (top_z - c['table_top_z']) / c['block_size'][2]
    n = max(1, int(round(levels)))
    return n, top_z - layer_top_z(c, n), levels


def layers_arg(text):
    """argparse type for --layers / --place-layers: a whole number 1..10, or 'auto'."""
    import argparse as _argparse
    if text.strip().lower() == 'auto':
        return 'auto'
    try:
        n = int(text)
    except ValueError:
        raise _argparse.ArgumentTypeError(f"'{text}' is not a whole number (or 'auto')") from None
    if not 1 <= n <= 10:
        raise _argparse.ArgumentTypeError('layers must be from 1 (on the table) to 10')
    return n


def resolve_place(c, pick_plan, place):
    """The place plan for the pick as it stands now. A place given as an offset follows the pick position
    (including any inspection correction, and on the table); a place yaw left out keeps the picked angle."""
    yaw = place['yaw'] if place['yaw'] is not None else pick_plan['yaw']
    if place.get('offset') is not None:
        return plan_place(c, pick_plan['centre'][0] + place['offset'][0], pick_plan['centre'][1] + place['offset'][1], yaw,
                          place.get('z'))
    return plan_place(c, *place['xy'], yaw, place['z'], place['is_tcp'])


def check_offset_place(c, pick_plan, place):
    """An offset place moves with the inspection correction, so it must stay valid for every correction the
    operator may make (max_total_nudge). Refuse now, not later with the brick already in the hand."""
    reach = float(c['max_total_nudge'])
    for sx in (-reach, reach):
        for sy in (-reach, reach):
            shifted = dict(pick_plan, centre=[pick_plan['centre'][0] + sx, pick_plan['centre'][1] + sy,
                                              pick_plan['centre'][2]])
            try:
                resolve_place(c, shifted, place)
            except ValueError as problem:
                raise ValueError(f'The place offset {list(place["offset"])} would leave the allowed area if the '
                                 f'pick is corrected by up to {reach * 1000:.0f} mm: {problem}') from None


def parse_nudge(text, max_step, max_yaw_step_deg):
    """'x 0.002 y -0.001 yaw 5' -> ('nudge', dx, dy, dyaw_deg); 'ok' -> ('ok',); 'q' -> ('quit',).
    Raises ValueError with a message for anything else."""
    words = text.lower().replace(',', ' ').split()
    if words in (['ok'], ['go']):
        return ('ok',)
    if words in (['q'], ['quit']):
        return ('quit',)
    if not words or len(words) % 2:
        raise ValueError('Type e.g.  x 0.002   or  y -0.001 yaw 5   or  ok   or  q')
    values = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    for key, number in zip(words[0::2], words[1::2]):
        if key not in values:
            raise ValueError(f"Unknown axis '{key}': use x, y or yaw")
        try:
            values[key] += float(number)
        except ValueError:
            raise ValueError(f"'{number}' is not a number") from None
    if max(abs(values['x']), abs(values['y'])) > max_step:
        raise ValueError(f'One correction may move at most {max_step * 1000:.0f} mm in x or y '
                         '(metres: 0.002 is 2 mm)')
    if abs(values['yaw']) > max_yaw_step_deg:
        raise ValueError(f'One correction may turn at most {max_yaw_step_deg:.0f} degrees')
    return ('nudge', values['x'], values['y'], values['yaw'])


def apply_nudge(plan, dx, dy, dyaw_deg):
    """Shift the plan sideways / turn it. Heights are untouched."""
    for key in ('centre', 'grip', 'inspect', 'above'):
        plan[key][0] += dx
        plan[key][1] += dy
    plan['yaw'] = float(wrap_line_angle(plan['yaw'] + math.radians(dyaw_deg)))
    return plan


STACK_NODES = ('move_group', 'controller_manager', 'robot_state_publisher')


def duplicate_stack_nodes(names):
    """Core robot nodes that exist more than once in this ROS domain. That means two robot stacks (for
    example a simulation next to the real robot) are sharing the domain, and a command could reach either."""
    names = list(names)
    return sorted(name for name in STACK_NODES if names.count(name) > 1)


def task_config(c, plan):
    """The mapping smooth_motion.Mover and pick_place.validate expect."""
    cfg = {k: c[k] for k in ('block_size', 'table_center', 'table_size', 'clearance', 'velocity_scale',
                             'acceleration_scale', 'grasp_force', 'grasp_width', 'gripper_speed',
                             'open_width', 'base_frame', 'ee_frame')}
    cfg['pick'] = cfg['place'] = list(plan['grip'])
    if 'min_open_clearance' in c:
        cfg['min_open_clearance'] = c['min_open_clearance']
    return cfg


def check_tcp_correction(c):
    """tcp_correction_hand must be two small numbers (metres). More than 1 cm is a typo or a unit mistake."""
    value = c.get('tcp_correction_hand')
    if value is None:
        return
    if not (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise ValueError('tcp_correction_hand must be two numbers [along, across] in metres')
    if max(abs(value[0]), abs(value[1])) > 0.010:
        raise ValueError(f'tcp_correction_hand {list(value)} is more than 10 mm: it is in METRES (0.002 is 2 mm)')


def jaw_clearance_note(c):
    """One line about how tight the open hand is around the brick, or None when it is roomy (>= 2 mm a side)."""
    side = (c['open_width'] - c['block_size'][1]) / 2
    if side >= 0.002:
        return None
    return (f'NOTE: the hand opens to {c["open_width"] * 1000:.0f} mm and closes to {c["grasp_width"] * 1000:.0f} mm for a '
            f'{c["block_size"][1] * 1000:.0f} mm brick: only {side * 1000:.1f} mm of room per side. Sideways or yaw '
            'errors over that put a finger on the brick: use the inspection stop.')


# --------------------------------------------------------------------------- ROS part
def lookup_hand(node, buffer, c, seconds=10.0):
    """T_base_hand (4x4) from TF, i.e. the arm's forward kinematics right now."""
    import rclpy
    from rclpy.time import Time
    from scipy.spatial.transform import Rotation
    end = time.monotonic() + seconds
    while not buffer.can_transform(c['base_frame'], c['ee_frame'], Time()):
        if time.monotonic() > end:
            raise RuntimeError('No hand pose from TF: is the robot (joint states) running?')
        rclpy.spin_once(node, timeout_sec=0.1)
    tf = buffer.lookup_transform(c['base_frame'], c['ee_frame'], Time()).transform
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix()
    pose[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
    return pose


def resolve_camera_target(c, t_base_hand, t_hand_cam, xyz, quat, long_axis, use_measured_z, label, layers=None):
    """A top-face point (and optional orientation) in the camera frame -> (x, y, centre_z or None, yaw or None).
    Prints what it did; raises ValueError when the converted height disagrees with where the brick should be.

    layers=None: a brick on the table (as before; use_measured_z trusts the measured height instead).
    layers=N: the brick is in layer N of a stack. The measured height must be within table_check_tolerance of
    that layer's top face, and the exact layer height is used (the camera height is only the cross-check).
    layers='auto': the layer is worked out from the measured height; refused when it is not clearly one layer
    (layer_auto_tolerance), because 3-7 mm of camera height error is a large part of one 9.6 mm layer."""
    from camera_frame import camera_point_to_base, camera_yaw_to_base, table_check
    top = camera_point_to_base(t_base_hand, t_hand_cam, xyz)
    tol = c.get('table_check_tolerance', 0.008)
    height = c['block_size'][2]
    where = f'{label}: camera point {list(xyz)} -> base frame x={top[0]:.4f} y={top[1]:.4f} z={top[2]:.4f}'
    centre_z = None
    if layers is None:
        err = table_check(c, top[2])
        print(f'{where} (top face; table check {err * 1000:+.1f} mm)')
        if abs(err) > tol and not use_measured_z:
            n, residual, levels = nearest_layer(c, top[2])
            raise ValueError(f'{label}: the top face is {err * 1000:+.1f} mm from where a brick on the table should be '
                             f'({levels:.2f} layers above the table): the camera point, the arm posture it was taken '
                             'from, or the calibration is wrong. Retake the point with the arm still. If the brick '
                             f'lies on other bricks, say how many layers: --layers {n} (or --place-layers {n}); '
                             '--use-measured-z trusts the measured height instead.')
        centre_z = float(top[2] - height / 2) if use_measured_z else None
    else:
        n, residual, levels = nearest_layer(c, top[2])
        if layers == 'auto':
            if abs(residual) > c.get('layer_auto_tolerance', 0.003):
                other = n + 1 if residual > 0 else max(1, n - 1)
                raise ValueError(f'{label}: the top face is {levels:.2f} layers above the table: nearest is layer {n} but '
                                 f'{residual * 1000:+.1f} mm off, too close to call from the camera height. Say how '
                                 f'many bricks are stacked: --layers {n} or --layers {other}.')
            layers = n
        expected = layer_top_z(c, layers)
        err = top[2] - expected
        print(f'{where} (top face, {levels:.2f} layers above the table; layer {layers} should be at '
              f'z={expected:.4f}: {err * 1000:+.1f} mm)')
        if abs(err) > tol:
            raise ValueError(f'{label}: the top face is {err * 1000:+.1f} mm from where layer {layers} should be '
                             f'(measured {levels:.2f} layers above the table): wrong layer count, or the camera point, '
                             'the arm posture it was taken from, or the calibration is wrong.')
        centre_z = layer_centre_z(c, layers) if layers > 1 else None
        print(f'{label}: using the exact layer {layers} height (brick centre z={layer_centre_z(c, layers):.4f})')
    yaw = None
    if quat is not None:
        yaw, tilt = camera_yaw_to_base(t_base_hand, t_hand_cam, quat, long_axis)
        print(f'{label}: orientation -> long side at {math.degrees(yaw):.1f} deg from base X')
        if tilt > 10:
            print(f'WARNING: {label}: the brick long axis leans {tilt:.0f} deg out of the table plane; '
                  'only its direction on the table is used.')
    return float(top[0]), float(top[1]), centre_z, yaw


class Mission:
    def __init__(self, node, c, plan, buffer, inspect, given, place=None):
        from visualization_msgs.msg import Marker
        import smooth_motion
        from smooth_motion import Mover
        self.node, self.c, self.plan, self.inspect, self.given = node, c, plan, inspect, given
        self.place = place        # {'xy': (x, y), 'z': metres or None, 'is_tcp': bool, 'yaw': radians or None}
        self.nudge = np.zeros(3)                                    # total dx, dy (m), dyaw (deg)
        self.grasp_hand_yaw_deg = None
        smooth_motion.MAX_JOINT_SWING = float(c['max_joint_swing'])  # refuse big turns of joints 1-6
        smooth_motion.MAX_WRIST_SWING = float(c['max_wrist_swing'])  # joint 7 may turn further
        self.marker_type = Marker
        self.marker_pub = node.create_publisher(Marker, '/lego_estimate_marker', 1)
        self.mover = Mover(node, c, buffer, plan['block_in_tcp'])
        self.mover.prepare_scene()

    def say(self, text):
        print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)

    def draw_box(self, ident, name, centre, yaw, colour):
        from scipy.spatial.transform import Rotation
        m = self.marker_type()
        m.header.frame_id, m.header.stamp = self.c['base_frame'], self.node.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = name, ident, m.CUBE, m.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, centre)
        q = Rotation.from_euler('z', yaw).as_quat()
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, q)
        m.scale.x, m.scale.y, m.scale.z = [float(v) for v in self.c['block_size']]
        m.color.r, m.color.g, m.color.b, m.color.a = colour
        self.marker_pub.publish(m)

    def show_target(self):
        """Green box in RViz at the (corrected) pick pose; blue box where the brick will be put down."""
        self.draw_box(0, 'pose_target', self.plan['centre'], self.plan['yaw'], (0.1, 0.9, 0.2, 0.55))
        if self.place is not None:
            where = self.place_plan()
            self.draw_box(1, 'place_target', where['centre'], where['yaw'], (0.1, 0.4, 1.0, 0.55))

    def place_plan(self):
        """The place pose; a place yaw left out means the brick keeps the angle it was picked at."""
        return resolve_place(self.c, self.plan, self.place)

    def spin(self, seconds):
        import rclpy
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def preflight(self):
        tilt = hand_tilt_deg(self.mover.hand_pose()[:3, :3])
        if tilt > self.c['max_tilt_deg']:
            raise RuntimeError(f'The hand is tilted {tilt:.0f} deg from pointing down (limit '
                               f'{self.c["max_tilt_deg"]} deg). Move the arm to a ready pose first (hand pointing '
                               'down, above the table, near the brick), then run again. Refusing to swing the '
                               'arm there in one go.')
        self.say(f'Start check OK: hand is {tilt:.0f} deg from pointing down.')

    def go(self, label, xyz, yaw=None):
        """One straight, approved move, then compare where the hand really is with the target."""
        commanded = self.mover.move(label, xyz, yaw=self.plan['yaw'] if yaw is None else yaw)
        self.spin(0.5)
        err = self.mover.hand_pose()[:3, 3] - np.array(commanded if commanded is not None else xyz)
        xy, z = math.hypot(err[0], err[1]), float(err[2])
        self.say(f'  arrived: {xy * 1000:.1f} mm sideways (dx={err[0] * 1000:+.1f} dy={err[1] * 1000:+.1f}), '
                 f'{z * 1000:+.1f} mm in height from the target')
        if max(xy, abs(z)) > self.c['arrival_abort']:
            raise RuntimeError(f'{label}: the hand is {max(xy, abs(z)) * 1000:.0f} mm from the target '
                               f'(limit {self.c["arrival_abort"] * 1000:.0f} mm); stopping')
        if max(xy, abs(z)) > self.c['arrival_warn']:
            self.say(f'  WARNING: more than {self.c["arrival_warn"] * 1000:.0f} mm off the target')

    def correct_offset(self):
        from smooth_motion import StepAbort
        c = self.c
        self.say('INSPECT: look at the fingers against the brick (from the side and from above).')
        print('  Type a small correction, e.g.  x 0.002   y -0.001   yaw 5   (metres / degrees, base axes:\n'
              '  +x away from the robot, +y to its left),  ok = lower and grasp,  q = stop here.')
        while True:
            try:
                answer = input('  correction> ')
            except EOFError:
                raise StepAbort('the correction prompt needs a terminal to answer in') from None
            try:
                command = parse_nudge(answer, c['max_nudge_step'], c['max_nudge_yaw_step_deg'])
            except ValueError as problem:
                print(f'  {problem}')
                continue
            if command[0] == 'ok':
                return
            if command[0] == 'quit':
                raise StepAbort('stopped at your request during the inspection')
            _, dx, dy, dyaw = command
            total = self.nudge + np.array([dx, dy, dyaw])
            if math.hypot(total[0], total[1]) > c['max_total_nudge'] or abs(total[2]) > 45:
                print(f'  Total correction would be {math.hypot(total[0], total[1]) * 1000:.0f} mm / '
                      f'{total[2]:.0f} deg: more than allowed ({c["max_total_nudge"] * 1000:.0f} mm / 45 deg). '
                      'The pose is probably wrong; stop and check it.')
                continue
            apply_nudge(self.plan, dx, dy, dyaw)
            self.nudge = total
            self.show_target()
            self.go('Correction', self.plan['inspect'])
            print(f'  Total correction so far: dx={total[0] * 1000:+.1f} mm  dy={total[1] * 1000:+.1f} mm  '
                  f'dyaw={total[2]:+.1f} deg')

    def save_log(self):
        path = Path(self.c.get('offsets_log', 'pose_pick_offsets.json'))
        try:
            history = json.loads(path.read_text()) if path.exists() else []
        except ValueError:
            history = []
        entry = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'pose_given': self.given,
                 'correction_dx_dy_m': [float(self.nudge[0]), float(self.nudge[1])],
                 'correction_yaw_deg': float(self.nudge[2]),
                 'final_grip_xyz': self.plan['grip'], 'final_yaw_deg': math.degrees(self.plan['yaw']),
                 'hand_yaw_deg': self.grasp_hand_yaw_deg,
                 'tcp_correction_used': list(self.c.get('tcp_correction_hand') or [0.0, 0.0])}
        if self.place is not None:
            where = self.place_plan()
            entry['place_xyz'], entry['place_yaw_deg'] = where['grip'], math.degrees(where['yaw'])
        history.append(entry)
        path.write_text(json.dumps(history, indent=2))
        self.say(f'Correction logged in {path}')

    def place_brick(self):
        """Carry the brick sideways at hover height, set it down, let go and back off straight up."""
        where = self.place_plan()
        self.show_target()
        self.go('Carry the brick above the place pose', where['above'], where['yaw'])
        self.go('Lower the brick to the place pose', where['release'], where['yaw'])
        self.mover.hand()                       # open: the brick stays where it is
        self.mover.detach_block()
        self.say('Brick released')
        self.go('Back off straight up', where['above'], where['yaw'])

    def run(self):
        p = self.plan
        for _ in range(5):                      # let RViz connect to the marker topic
            self.show_target()
            self.spin(0.1)
        self.preflight()
        self.mover.hand()
        self.go('Hover above the brick', p['above'])
        self.go('Come down to the inspection height', p['inspect'])
        if self.inspect:
            self.correct_offset()
        self.say(f'Correction needed: dx={self.nudge[0] * 1000:+.1f} mm dy={self.nudge[1] * 1000:+.1f} mm '
                 f'dyaw={self.nudge[2]:+.1f} deg')
        self.go('Lower to the brick', p['grip'])
        from smooth_motion import hand_yaw
        self.grasp_hand_yaw_deg = math.degrees(hand_yaw(self.mover.hand_pose()[:3, :3]))   # for measure_grasp_offset.py
        self.mover.hand(close=True)
        self.say('Grasp reported success')
        self.mover.attach_block()
        self.go('Lift the brick', p['above'])
        if self.place is None:
            self.save_log()
            self.say('DONE: brick is held above its place')
            return
        self.place_brick()
        self.save_log()
        self.say('DONE: brick picked and placed')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument('--pose', nargs='+', type=float, metavar='V',
                       help='X Y [Z] in metres, fr3_link0 (brick centre to pick; z optional)')
    where.add_argument('--camera-xyz', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Top-face centre of the brick to pick, in camera_color_optical_frame (metres)')
    put = parser.add_mutually_exclusive_group()
    put.add_argument('--place', nargs='+', type=float, metavar='V',
                     help='X Y [Z] brick centre to put it down at (fr3_link0, metres; z optional)')
    put.add_argument('--place-camera-xyz', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                     help='Top-face centre of the brick where it should end up, in camera_color_optical_frame')
    put.add_argument('--place-offset', nargs=2, type=float, metavar=('DX', 'DY'),
                     help='Put it down this far from the pick position (fr3_link0 axes, metres), e.g. 0 0.05')
    parser.add_argument('--camera-quat', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'W'),
                        help='Orientation of the brick to pick in the camera frame (xyzw)')
    parser.add_argument('--place-camera-quat', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'W'),
                        help='Orientation of the placed brick in the camera frame (xyzw)')
    parser.add_argument('--camera-long-axis', choices=['x', 'y'], default='x',
                        help="Which axis of the camera-frame brick orientation runs along its long side")
    parser.add_argument('--ee-pose', nargs=7, type=float, metavar=('X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW'),
                        help='Hand (fr3_hand_tcp) pose in fr3_link0 when the camera picture was taken '
                             '(default: the arm right now)')
    parser.add_argument('--use-measured-z', action='store_true',
                        help='With camera input: use the measured brick height instead of the table model')
    parser.add_argument('--calibration', help='Hand-eye JSON (default: calibration_file in the config)')
    parser.add_argument('--layers', type=layers_arg, metavar='N|auto',
                        help='The brick to pick is in layer N of a stack (1 = on the table, 2 = on top of one brick, ...), '
                             'or auto = work it out from the camera height (refuses when it cannot tell)')
    parser.add_argument('--place-layers', type=layers_arg, metavar='N|auto',
                        help='Put the brick down as layer N of a stack (1 = on the table, 2 = on top of one brick, ...)')
    parser.add_argument('--place-yaw', type=float, metavar='DEG',
                        help='Long-axis angle of the placed brick (default: the angle it was picked at)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--yaw', type=float, metavar='DEG', help='Brick long-axis angle from base X (default 0)')
    group.add_argument('--quat', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'W'),
                       help='Orientation as xyzw quaternion in the base frame (only its yaw is used)')
    parser.add_argument('--pose-is-tcp', action='store_true', help='With --pose: Z is the hand (TCP) height at the grasp')
    parser.add_argument('--place-is-tcp', action='store_true', help='With --place: Z is the hand (TCP) height at the release')
    parser.add_argument('--yes', action='store_true',
                        help='Skip the GO prompt (only honoured when the simulated gripper is running)')
    parser.add_argument('--step', action='store_true', help='Ask before every move (default on a real robot)')
    parser.add_argument('--no-step', action='store_true', help='Never ask before moves')
    parser.add_argument('--inspect', action='store_true', help='Stop above the brick for corrections (default on a real robot)')
    parser.add_argument('--no-inspect', action='store_true', help='Do not stop for corrections')
    args, ros_args = parser.parse_known_args()
    if args.step and args.no_step or args.inspect and args.no_inspect:
        raise ValueError('Choose either --step or --no-step, and either --inspect or --no-inspect')
    if args.pose is not None and len(args.pose) not in (2, 3):
        raise ValueError('--pose takes X Y or X Y Z')
    if args.place is not None and len(args.place) not in (2, 3):
        raise ValueError('--place takes X Y or X Y Z')
    if args.place_yaw is not None and args.place is None and args.place_camera_xyz is None \
            and args.place_offset is None:
        raise ValueError('--place-yaw needs --place, --place-camera-xyz or --place-offset')
    if args.place_offset is not None and max(abs(args.place_offset[0]), abs(args.place_offset[1])) > 0.5:
        raise ValueError('--place-offset is in metres: 0.05 is 5 cm (more than 0.5 m is surely a typo)')
    if args.place_is_tcp and args.place is None:
        raise ValueError('--place-is-tcp needs --place X Y Z')
    if args.pose_is_tcp and args.pose is None:
        raise ValueError('--pose-is-tcp needs --pose X Y Z')
    if args.camera_quat is not None and (args.camera_xyz is None or args.yaw is not None or args.quat):
        raise ValueError('--camera-quat goes with --camera-xyz and replaces --yaw / --quat')
    if args.place_camera_quat is not None and (args.place_camera_xyz is None or args.place_yaw is not None):
        raise ValueError('--place-camera-quat goes with --place-camera-xyz and replaces --place-yaw')
    camera_used = args.camera_xyz is not None or args.place_camera_xyz is not None
    if args.ee_pose is not None and not camera_used:
        raise ValueError('--ee-pose is only used with --camera-xyz / --place-camera-xyz')
    if args.layers is not None:
        if args.use_measured_z:
            raise ValueError('--layers and --use-measured-z are two ways to give the height: use one')
        if args.pose is not None and (args.layers == 'auto' or len(args.pose) == 3):
            raise ValueError('With --pose, --layers needs a number and no Z (it sets the height); '
                             '--layers auto needs --camera-xyz')
    if args.place_layers is not None:
        if args.place is None and args.place_camera_xyz is None and args.place_offset is None:
            raise ValueError('--place-layers needs --place, --place-camera-xyz or --place-offset')
        if args.use_measured_z:
            raise ValueError('--place-layers and --use-measured-z are two ways to give the height: use one')
        if args.place_layers == 'auto' and args.place_camera_xyz is None:
            raise ValueError('--place-layers auto needs --place-camera-xyz')
        if args.place is not None and len(args.place) == 3:
            raise ValueError('--place-layers sets the height: leave out the Z of --place')
    c = yaml.safe_load(open(args.config, encoding='utf-8'))
    check_tcp_correction(c)
    yaw = None
    if args.quat:
        yaw, tilt = yaw_from_quaternion(args.quat)
        if tilt > 5:
            print(f'WARNING: the quaternion is tilted {tilt:.0f} deg; only its yaw is used '
                  '(the hand always comes straight down).')
    elif args.yaw is not None:
        yaw = wrap_line_angle(math.radians(args.yaw))

    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener
    from pick_place import validate
    from smooth_motion import StepAbort
    exit_code = 0
    rclpy.init(args=ros_args)
    node = rclpy.create_node('pose_pick')
    buffer = Buffer(cache_time=Duration(seconds=30))
    listener = TransformListener(buffer, node)         # keep a reference alive
    try:
        end = time.monotonic() + 5.0                   # DDS discovery can take a few seconds: do not
        while time.monotonic() < end and 'sim_gripper' not in node.get_node_names():   # mistake a slow start for a real robot
            rclpy.spin_once(node, timeout_sec=0.1)
        doubled = duplicate_stack_nodes(name for name, _ in node.get_node_names_and_namespaces())
        if doubled:
            raise RuntimeError(f'Two robot stacks are running in this ROS domain (more than one {", ".join(doubled)}): '
                               'for example a simulation next to the real robot. A command could reach the wrong '
                               'one. Stop one of them, then run again. Nothing was moved.')
        simulated = 'sim_gripper' in node.get_node_names()
        if simulated:
            c['velocity_scale'] = c['acceleration_scale'] = float(c.get('sim_speed_scale', 0.1))
            print('SIMULATION: sim_gripper found, using speed scale', c['velocity_scale'])
        else:
            print('REAL ROBOT.')

        # ---- turn whatever was typed into pick / place plans. Nothing moves until the GO below.
        t_base_hand = t_hand_cam = None
        if camera_used:
            from camera_frame import load_calibration, pose_matrix
            calibration, t_hand_cam = load_calibration(args.calibration or c['calibration_file'], c,
                                                       c.get('camera_frame', 'camera_color_optical_frame'))
            print(f'Hand-eye: status {calibration["status"]}, typical error '
                  f'{calibration["all_samples"]["translation_rms_m"] * 1000:.1f} mm / '
                  f'{calibration["all_samples"]["rotation_rms_deg"]:.1f} deg')
            if args.ee_pose is not None:
                t_base_hand = pose_matrix(args.ee_pose[:3], args.ee_pose[3:])
            else:
                t_base_hand = lookup_hand(node, buffer, c)
            p = t_base_hand[:3, 3]
            print(f'Hand when the picture was taken ({"typed in" if args.ee_pose else "the arm right now"}): '
                  f'x={p[0]:.4f} y={p[1]:.4f} z={p[2]:.4f}')
        given = {}
        if args.pose is not None:
            z = args.pose[2] if len(args.pose) == 3 else (
                layer_centre_z(c, args.layers) if isinstance(args.layers, int) and args.layers > 1 else None)
            plan = plan_pick(c, args.pose[0], args.pose[1], z, yaw or 0.0, args.pose_is_tcp)
            given = {'pose': args.pose, 'yaw_deg': math.degrees(plan['yaw'])}
        else:
            x, y, centre_z, cam_yaw = resolve_camera_target(
                c, t_base_hand, t_hand_cam, args.camera_xyz, args.camera_quat, args.camera_long_axis,
                args.use_measured_z, 'Pick', args.layers)
            plan = plan_pick(c, x, y, centre_z, cam_yaw if cam_yaw is not None else (yaw or 0.0))
            given = {'camera_xyz': args.camera_xyz, 'base_xy': [x, y], 'yaw_deg': math.degrees(plan['yaw'])}
        validate(task_config(c, plan))
        place = None
        stack_z = layer_centre_z(c, args.place_layers) if isinstance(args.place_layers, int) and args.place_layers > 1 else None
        if args.place is not None:
            place = {'xy': (args.place[0], args.place[1]), 'z': args.place[2] if len(args.place) == 3 else stack_z,
                     'is_tcp': args.place_is_tcp,
                     'yaw': wrap_line_angle(math.radians(args.place_yaw)) if args.place_yaw is not None else None}
        elif args.place_camera_xyz is not None:
            x, y, centre_z, cam_yaw = resolve_camera_target(
                c, t_base_hand, t_hand_cam, args.place_camera_xyz, args.place_camera_quat, args.camera_long_axis,
                args.use_measured_z, 'Place', args.place_layers)
            place_yaw = cam_yaw if cam_yaw is not None else (
                wrap_line_angle(math.radians(args.place_yaw)) if args.place_yaw is not None else None)
            place = {'xy': (x, y), 'z': centre_z, 'is_tcp': False, 'yaw': place_yaw}
        elif args.place_offset is not None:
            place = {'offset': tuple(args.place_offset), 'xy': None, 'z': stack_z, 'is_tcp': False,
                     'yaw': wrap_line_angle(math.radians(args.place_yaw)) if args.place_yaw is not None else None}
            check_offset_place(c, plan, place)
        if place is not None:                         # a bad place pose must not leave a brick in the hand
            validate(task_config(c, resolve_place(c, plan, place)))

        inspect = args.inspect or (not simulated and not args.no_inspect)
        mission = Mission(node, c, plan, buffer, inspect, given, place)
        centre = plan['centre']
        print(f'Pick pose: x={centre[0]:.3f} y={centre[1]:.3f} z={centre[2]:.4f}  yaw={math.degrees(plan["yaw"]):.0f} deg\n'
              f'Hand will hover at z={plan["above"][2]:.3f}, stop at z={plan["inspect"][2]:.3f} '
              f'{"for corrections" if inspect else "(no corrections)"}, grasp at z={plan["grip"][2]:.4f}, '
              'then lift back up.')
        if place is not None:
            where = mission.place_plan()
            print(f'Place pose: x={where["centre"][0]:.3f} y={where["centre"][1]:.3f} z={where["centre"][2]:.4f}  '
                  f'yaw={math.degrees(where["yaw"]):.0f} deg   (release with the hand at z={where["release"][2]:.4f})')
        note = jaw_clearance_note(c)
        if note:
            print(note)
        correction = c.get('tcp_correction_hand')
        if correction and any(correction):
            print(f'Grasp centring in force (tcp_correction_hand, hand frame): along the long side '
                  f'{correction[0] * 1000:+.1f} mm, across it {correction[1] * 1000:+.1f} mm')
        print('This will MOVE THE ARM AND HAND.')
        if not (simulated and args.yes):
            if input('Hold the e-stop, clear the workspace. Type GO to start: ').strip() != 'GO':
                raise SystemExit('Cancelled; nothing was moved.')
        mission.mover.step_mode = args.step or (not simulated and not args.no_step)
        if mission.mover.step_mode:
            print('STEP MODE: you will be asked before every arm move and hand action.')
        mission.run()
    except StepAbort as stop:
        print(f'\nSTOPPED ({stop}). Nothing further was commanded. The arm stays where it is and the '
              'hand was not opened or closed automatically.')
    finally:
        del listener
        node.destroy_node()
        rclpy.try_shutdown()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
