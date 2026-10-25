#!/usr/bin/python3 -s
"""Pose-driven pick: go to a brick pose, check the offset, correct it slowly, grasp and lift.

A teammate gives the brick pose; this script does the rest, one small safe step at a time:

  0. CHECK   the hand must already point down (else it refuses: move to a ready pose first)
  1. OPEN    open the hand
  2. HOVER   straight line to above the brick (clearance above the grasp height)
  3. INSPECT straight down to a few cm above the grasp height and STOP. Look at the fingers
             against the brick, then type small corrections (x, y, yaw) until it is right.
  4. LOWER   straight down to the grasp height
  5. GRASP   close the hand (the result is checked)
  6. LIFT    straight back up. Without --place the brick stays in the hand.
  With --place X Y [--place-yaw DEG] it goes on:
  7. CARRY   straight line at hover height to above the place pose
  8. LOWER   straight down until the brick touches the table (place_margin above it)
  9. RELEASE open the hand, then back off straight up
  In simulation the true brick pose (/sim_brick_marker) is then compared with the place pose and the
  run exits with code 2 if it is off by more than sim_check_xy_tol / sim_check_yaw_tol_deg.
The correction you needed is printed and appended to pose_offsets (offsets_log in the config), so
you can tell your teammate (or the calibration) what the pose was off by.

Pose input, either:
  --pose X Y [Z]       brick CENTRE in fr3_link0, metres (z left out = brick rests on the table)
  --camera-xyz X Y Z   the point on the brick's TOP face in the camera frame
                       (camera_color_optical_frame), converted with YOUR hand-eye matrix
                       (calibration_file) and the arm's current pose. The arm must still be in the
                       posture the picture was taken from. The converted top face must agree with
                       the table model (table check) or it refuses.
  --yaw DEG | --quat X Y Z W   brick long-side angle from base X (only the yaw of a quaternion is used)

Correction commands at the inspection stop (base-frame axes: +x away from the robot, +y to its left):
  x 0.002   y -0.001   yaw 5      any combination on one line; metres / degrees, small steps only
  ok = continue to lower and grasp      q = stop here (nothing is released or closed)

Every arm move is a straight, collision-checked Cartesian line (smooth_motion.py) at low speed, and
after each move the real hand position is compared with the target. Real robot: you type GO and STEP
MODE is on (Enter = do it, a = rest automatically, q = stop). The hand is never opened automatically
after an error. The commanded brick is drawn in RViz as a green box (/lego_estimate_marker).
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
        raise ValueError('--pose-is-tcp needs a z value')
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


def plan_place(c, x, y, yaw):
    """Where the carried brick is put down. Same rules as a pick pose (workspace window, table, fingertips);
    the hand stops place_margin above the height at which the brick bottom touches the table."""
    plan = plan_pick(c, x, y, None, yaw)
    margin = float(c.get('place_margin', 0.0))
    if margin < 0:
        raise ValueError('place_margin must not be negative: it would push the brick into the table')
    plan['release'] = [plan['grip'][0], plan['grip'][1], plan['grip'][2] + margin]
    return plan


def yaw_error_deg(a, b):
    """Smallest angle between two brick long-axis directions (they repeat every 180 degrees), in degrees."""
    return abs(math.degrees(wrap_line_angle(a - b)))


def camera_point_to_base(t_base_ee, t_ee_cam, p_cam):
    """p_base = T_base_ee (forward kinematics) @ T_ee_camera (hand-eye) @ p_camera."""
    return (np.asarray(t_base_ee) @ np.asarray(t_ee_cam) @ np.append(np.asarray(p_cam, dtype=float), 1.0))[:3]


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


def task_config(c, plan):
    """The mapping smooth_motion.Mover and pick_place.validate expect."""
    cfg = {k: c[k] for k in ('block_size', 'table_center', 'table_size', 'clearance', 'velocity_scale',
                             'acceleration_scale', 'grasp_force', 'grasp_width', 'gripper_speed',
                             'open_width', 'base_frame', 'ee_frame')}
    cfg['pick'] = cfg['place'] = list(plan['grip'])
    return cfg


# --------------------------------------------------------------------------- ROS part
def lookup_hand(node, buffer, c, seconds=10.0):
    """T_base_ee (4x4) from TF, i.e. the arm's forward kinematics."""
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


class Mission:
    def __init__(self, node, c, plan, buffer, inspect, given, place=None, simulated=False):
        from visualization_msgs.msg import Marker
        import smooth_motion
        from smooth_motion import Mover
        self.node, self.c, self.plan, self.inspect, self.given = node, c, plan, inspect, given
        self.place, self.truth, self.simulated = place, None, simulated   # place = {'xy': (x, y), 'yaw': radians or None}
        self.nudge = np.zeros(3)                                    # total dx, dy (m), dyaw (deg)
        smooth_motion.MAX_JOINT_SWING = float(c['max_joint_swing'])  # refuse big turns of joints 1-6
        smooth_motion.MAX_WRIST_SWING = float(c['max_wrist_swing'])  # joint 7 may turn further
        self.marker_type = Marker
        self.marker_pub = node.create_publisher(Marker, '/lego_estimate_marker', 1)
        if simulated:                             # sim_world publishes where the brick REALLY is
            node.create_subscription(Marker, '/sim_brick_marker', self.on_truth, 10)
        self.mover = Mover(node, c, buffer, plan['block_in_tcp'])
        self.mover.prepare_scene()

    def on_truth(self, marker):
        if marker.ns == 'sim_world' and marker.id == 0:
            q, p = marker.pose.orientation, marker.pose.position
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.truth = (p.x, p.y, p.z, yaw)

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
        """Green box in RViz at the (corrected) brick pose; blue box where it will be put down."""
        self.draw_box(0, 'pose_target', self.plan['centre'], self.plan['yaw'], (0.1, 0.9, 0.2, 0.55))
        if self.place is not None:
            where = self.place_plan()
            self.draw_box(1, 'place_target', where['centre'], where['yaw'], (0.1, 0.4, 1.0, 0.55))

    def place_plan(self):
        """The place pose; a place yaw left out means the brick keeps the angle it was picked at."""
        yaw = self.place['yaw'] if self.place['yaw'] is not None else self.plan['yaw']
        return plan_place(self.c, *self.place['xy'], yaw)

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
        self.mover.move(label, xyz, yaw=self.plan['yaw'] if yaw is None else yaw)
        self.spin(0.5)
        err = self.mover.hand_pose()[:3, 3] - np.array(xyz)
        xy, z = math.hypot(err[0], err[1]), float(err[2])
        self.say(f'  arrived: {xy * 1000:.1f} mm sideways, {z * 1000:+.1f} mm in height from the target')
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
                 'final_grip_xyz': self.plan['grip'], 'final_yaw_deg': math.degrees(self.plan['yaw'])}
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
        self.go('Lower the brick onto the table', where['release'], where['yaw'])
        self.mover.hand()                       # open: the brick stays where it is
        self.mover.detach_block()
        self.say('Brick released')
        self.go('Back off straight up', where['above'], where['yaw'])
        return where

    def check_against_truth(self, where):
        """SIMULATION: compare where the brick really ended up with the place pose that was asked for.
        Returns True when it is within sim_check_xy_tol / sim_check_yaw_tol_deg."""
        self.spin(1.0)                          # let the final marker arrive
        if self.truth is None:
            self.say('SIM CHECK: no /sim_brick_marker received, cannot check the result')
            return False
        x, y, z, yaw = self.truth
        want = where['centre']
        xy_err = math.hypot(x - want[0], y - want[1])
        yaw_err = yaw_error_deg(yaw, where['yaw'])
        z_err = z - want[2]
        tol_xy, tol_yaw = float(self.c.get('sim_check_xy_tol', 0.001)), float(self.c.get('sim_check_yaw_tol_deg', 1.0))
        ok = xy_err <= tol_xy and yaw_err <= tol_yaw and abs(z_err) <= 0.001
        self.say(f'SIM CHECK: brick really is at x={x:.4f} y={y:.4f}, {xy_err * 1000:.2f} mm from the place pose '
                 f'(dx={(x - want[0]) * 1000:+.2f} dy={(y - want[1]) * 1000:+.2f}), yaw off {yaw_err:.2f} deg, '
                 f'height off {z_err * 1000:+.2f} mm -> {"PASS" if ok else "FAIL"} '
                 f'(limits {tol_xy * 1000:.1f} mm / {tol_yaw:.1f} deg)')
        return ok

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
        self.mover.hand(close=True)
        self.say('Grasp reported success')
        self.mover.attach_block()
        self.go('Lift the brick', p['above'])
        if self.place is None:
            self.save_log()
            self.say('DONE: brick is held above its place')
            return True
        where = self.place_brick()
        self.save_log()
        self.say('DONE: brick picked and placed')
        return self.check_against_truth(where) if self.simulated else True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument('--pose', nargs='+', type=float, metavar='V',
                       help='X Y [Z] in metres, fr3_link0 (brick centre; z optional)')
    where.add_argument('--camera-xyz', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Point on the brick top face in camera_color_optical_frame (metres)')
    parser.add_argument('--place', nargs=2, type=float, metavar=('X', 'Y'),
                        help='After the pick, carry the brick to this centre (fr3_link0, metres) and put it down')
    parser.add_argument('--place-yaw', type=float, metavar='DEG',
                        help='Long-axis angle of the placed brick (default: the angle it was picked at)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--yaw', type=float, default=0.0, metavar='DEG', help='Brick long-axis angle from base X')
    group.add_argument('--quat', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'W'),
                       help='Orientation as xyzw quaternion in the base frame (only its yaw is used)')
    parser.add_argument('--pose-is-tcp', action='store_true', help='With --pose: Z is the hand (TCP) height at the grasp')
    parser.add_argument('--use-measured-z', action='store_true',
                        help='With --camera-xyz: use the measured brick height instead of the table model')
    parser.add_argument('--calibration', help='Hand-eye JSON for --camera-xyz (default: calibration_file in the config)')
    parser.add_argument('--yes', action='store_true',
                        help='Skip the GO prompt (only honoured when sim_world is running)')
    parser.add_argument('--step', action='store_true', help='Ask before every move (default on a real robot)')
    parser.add_argument('--no-step', action='store_true', help='Never ask before moves')
    parser.add_argument('--inspect', action='store_true', help='Stop above the brick for corrections (default on a real robot)')
    parser.add_argument('--no-inspect', action='store_true', help='Do not stop for corrections')
    args, ros_args = parser.parse_known_args()
    if args.step and args.no_step or args.inspect and args.no_inspect:
        raise ValueError('Choose either --step or --no-step, and either --inspect or --no-inspect')
    if args.pose is not None and len(args.pose) not in (2, 3):
        raise ValueError('--pose takes X Y or X Y Z')
    if args.place_yaw is not None and args.place is None:
        raise ValueError('--place-yaw needs --place X Y')
    c = yaml.safe_load(open(args.config, encoding='utf-8'))
    if args.quat:
        yaw, tilt = yaw_from_quaternion(args.quat)
        if tilt > 5:
            print(f'WARNING: the quaternion is tilted {tilt:.0f} deg; only its yaw is used '
                  '(the hand always comes straight down).')
    else:
        yaw = wrap_line_angle(math.radians(args.yaw))

    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener
    from pick_place import validate
    from smooth_motion import StepAbort
    plan = given = None
    exit_code = 0
    place = None
    if args.place is not None:                         # fail early: a bad place pose must not leave the brick in the hand
        place = {'xy': tuple(args.place),
                 'yaw': wrap_line_angle(math.radians(args.place_yaw)) if args.place_yaw is not None else None}
        validate(task_config(c, plan_place(c, *place['xy'], place['yaw'] or 0.0)))
    if args.pose is not None:                          # fail early: no ROS needed to check a typed pose
        z = args.pose[2] if len(args.pose) == 3 else None
        plan = plan_pick(c, args.pose[0], args.pose[1], z, yaw, args.pose_is_tcp)
        validate(task_config(c, plan))
        given = {'pose': args.pose, 'yaw_deg': math.degrees(yaw)}
    rclpy.init(args=ros_args)
    node = rclpy.create_node('pose_pick')
    buffer = Buffer(cache_time=Duration(seconds=30))
    listener = TransformListener(buffer, node)         # keep a reference alive
    try:
        end = time.monotonic() + 5.0                   # DDS discovery can take a few seconds: do not
        while time.monotonic() < end and 'sim_world' not in node.get_node_names():   # mistake a slow start for a real robot
            rclpy.spin_once(node, timeout_sec=0.1)
        simulated = 'sim_world' in node.get_node_names()
        if simulated:
            c['velocity_scale'] = c['acceleration_scale'] = float(c.get('sim_speed_scale', 0.1))
            print('SIMULATION: sim_world found, using speed scale', c['velocity_scale'])
        else:
            print('REAL ROBOT.')
        if args.camera_xyz is not None:
            from pixel_to_base import load_calibration, table_check
            calibration, t_ee_cam = load_calibration(args.calibration or c['calibration_file'], c,
                                                     c.get('camera_frame', 'camera_color_optical_frame'))
            print(f'Hand-eye: status {calibration["status"]}, typical error '
                  f'{calibration["all_samples"]["translation_rms_m"] * 1000:.1f} mm / '
                  f'{calibration["all_samples"]["rotation_rms_deg"]:.1f} deg')
            top = camera_point_to_base(lookup_hand(node, buffer, c), t_ee_cam, args.camera_xyz)
            err = table_check(c, top[2])
            print(f'Camera point {args.camera_xyz} -> base frame x={top[0]:.4f} y={top[1]:.4f} z={top[2]:.4f} '
                  f'(top face; table check {err * 1000:+.1f} mm)')
            if abs(err) > c.get('table_check_tolerance', 0.008) and not args.use_measured_z:
                raise ValueError(f'The top face is {err * 1000:+.1f} mm from where a brick on the table should be: '
                                 'the camera point, the arm posture it was taken from, or the calibration is wrong. '
                                 'Retake the point with the arm still, or use --use-measured-z if the brick is not '
                                 'on the table.')
            centre_z = top[2] - c['block_size'][2] / 2 if args.use_measured_z else None
            plan = plan_pick(c, float(top[0]), float(top[1]), centre_z, yaw)
            validate(task_config(c, plan))
            given = {'camera_xyz': args.camera_xyz, 'base_top': top.tolist(), 'yaw_deg': math.degrees(yaw)}
        inspect = args.inspect or (not simulated and not args.no_inspect)
        mission = Mission(node, c, plan, buffer, inspect, given, place, simulated)
        centre = plan['centre']
        print(f'Brick pose: x={centre[0]:.3f} y={centre[1]:.3f} z={centre[2]:.4f}  yaw={math.degrees(plan["yaw"]):.0f} deg\n'
              f'Hand will hover at z={plan["above"][2]:.3f}, stop at z={plan["inspect"][2]:.3f} '
              f'{"for corrections" if inspect else "(no corrections)"}, grasp at z={plan["grip"][2]:.4f}, '
              'then lift back up.\nThis will MOVE THE ARM AND HAND.')
        if not (simulated and args.yes):
            if input('Hold the e-stop, clear the workspace. Type GO to start: ').strip() != 'GO':
                raise SystemExit('Cancelled; nothing was moved.')
        mission.mover.step_mode = args.step or (not simulated and not args.no_step)
        if mission.mover.step_mode:
            print('STEP MODE: you will be asked before every arm move and hand action.')
        if not mission.run():
            exit_code = 2                              # the simulated brick did not end up where it was asked to
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
