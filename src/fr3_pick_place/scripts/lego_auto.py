#!/usr/bin/python3 -s
"""Autonomous Lego pick and place: the wrist camera keeps watching while the arm works.

  1. SEARCH    visit the scan poses (config: scan_poses) until the brick is seen.
  2. SERVO     step down through hover heights (servo_heights). At every height the arm stops,
               the camera processes every frame for a moment, and the arm re-aims at the improved
               brick position. The camera is closer each time, so calibration error shrinks.
  3. PICK      descend, close the hand (the grasp result is checked), lift.
  4. PLACE     carry to --place-xy, lower, open, retreat.
  5. VERIFY    go back to a scan pose and report where the brick ended up.

Every arm move is a straight, collision-checked Cartesian line with a smooth speed profile (see
smooth_motion.py); a move that cannot be done that way is refused instead of improvised.
The camera's estimate of the brick is drawn in RViz as a green box (/lego_estimate_marker).
Works unchanged in simulation (sim_world.py provides camera + hand) and on the real robot.
Real robot: you must type GO, and STEP MODE is on: before every arm move and hand action the
script shows the planned move and waits for Enter (a = continue automatically, q = stop).
Use --no-step to turn that off. Simulation: --yes skips GO (only if sim_world is running) and
step mode is off unless you give --step.
Any failure stops the sequence. The hand is never opened automatically after an error.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lego_pick_place as lp                      # noqa: E402  (camera, detection, task config)
from smooth_motion import Mover, StepAbort       # noqa: E402  (straight-line moves, hand)


class Tracker:
    """Turns the live camera stream into a stable brick estimate."""

    def __init__(self, eye, c):
        self.eye, self.c, self.last_stamp, self.debug = eye, c, None, None

    def watch(self, seconds, min_frames=8, max_spread=0.006, near=None):
        """Process every new frame for `seconds` while the arm is still. Returns (brick, status)."""
        found, shown = [], 0
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.eye.spin(0.03)
            frame = self.eye.latest_frame(self.last_stamp)
            if frame is None:
                continue
            self.last_stamp, bgr, k, t_base_cam = frame
            shown += 1
            accepted, rejected = lp.detect_bricks(bgr, k, t_base_cam, self.c)
            self.debug = lp.annotate(bgr, accepted, rejected)
            if near is not None:
                accepted = [a for a in accepted if np.linalg.norm(a['centre'][:2] - near[:2]) < 0.05]
            if accepted:
                found.append(accepted[0])
        if not found:
            return None, f'no brick in {shown} frames'
        xy = np.array([f['centre'][:2] for f in found])
        median = np.median(xy, axis=0)
        keep = [f for f in found if np.linalg.norm(f['centre'][:2] - median) < 0.015]
        if len(keep) < min_frames:
            return None, f'brick seen in only {len(keep)} of {shown} frames (need {min_frames})'
        centres = np.array([f['centre'] for f in keep])
        spread = float(np.max(np.linalg.norm(centres[:, :2] - centres[:, :2].mean(axis=0), axis=1)))
        if spread > max_spread:
            return None, f'brick position jitters {spread * 1000:.1f} mm'
        brick = {'centre': centres.mean(axis=0), 'yaw': lp.circular_mean_line([f['yaw'] for f in keep]),
                 'size': np.mean([f['size'] for f in keep], axis=0)}
        return brick, f'{len(keep)}/{shown} frames, spread {spread * 1000:.1f} mm'


class Mission:
    def __init__(self, node, c, calibration, place_xy, validate):
        self.node, self.c, self.place_xy = node, c, place_xy
        self.validate = validate
        self.eye = lp.Eye(node, c, calibration)
        self.tracker = Tracker(self.eye, c)
        top = c['table_top_z'] + c['block_size'][2]
        placeholder = {'centre': np.array([0.45, 0.0, top]), 'yaw': 0.0, 'size': (0.032, 0.016)}
        self.cfg = validate(lp.build_task_config(c, placeholder, place_xy))
        self.mover = Mover(node, c, self.eye.buffer, self.cfg['block_in_tcp'])
        self.mover.prepare_scene()
        self.grip_z = self.cfg['pick'][2]
        from visualization_msgs.msg import Marker
        self.marker_type = Marker
        self.marker_pub = node.create_publisher(Marker, '/lego_estimate_marker', 1)

    def say(self, text):
        print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)

    def aim(self, brick):
        """Point the pick position at the current brick estimate."""
        self.cfg.update(self.validate(lp.build_task_config(self.c, brick, self.place_xy)))

    def show_estimate(self, brick):
        """Green box in RViz where the CAMERA believes the brick is."""
        from scipy.spatial.transform import Rotation
        m = self.marker_type()
        m.header.frame_id, m.header.stamp = self.c['base_frame'], self.node.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'lego_estimate', 0, m.CUBE, m.ADD
        m.pose.position.x, m.pose.position.y = float(brick['centre'][0]), float(brick['centre'][1])
        m.pose.position.z = float(self.c['table_top_z'] + self.c['block_size'][2] / 2 + 0.002)
        q = Rotation.from_euler('z', brick['yaw']).as_quat()
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, q)
        m.scale.x, m.scale.y, m.scale.z = [float(v) for v in self.c['block_size']]
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.55
        self.marker_pub.publish(m)

    def save_debug(self, name):
        if self.tracker.debug is not None:
            cv2.imwrite(name, self.tracker.debug)

    def settle_and_watch(self, label, near=None, seconds=None):
        self.eye.spin(self.c['settle_seconds'])
        brick, status = self.tracker.watch(seconds or self.c['watch_seconds'], near=near)
        self.say(f'{label}: ' + (f'brick at x={brick["centre"][0]:.4f} y={brick["centre"][1]:.4f} '
                                 f'yaw={math.degrees(brick["yaw"]):.0f} deg ({status})'
                                 if brick else f'not found ({status})'))
        if brick:
            self.show_estimate(brick)
        return brick

    # ----------------------------------------------------------------- phases
    def search(self):
        for n, xyz in enumerate(self.c['scan_poses'], 1):
            self.mover.move(f'Scan pose {n}', xyz)
            brick = self.settle_and_watch(f'Search {n}/{len(self.c["scan_poses"])}')
            if brick:
                self.save_debug('lego_auto_first_look.png')
                return brick
        self.save_debug('lego_auto_failed.png')
        raise RuntimeError('Brick not found from any scan pose (see lego_auto_failed.png)')

    def servo(self, brick):
        first = brick
        for height in self.c['servo_heights']:
            for attempt in range(1, self.c['servo_max_moves'] + 1):
                self.aim(brick)
                target = [brick['centre'][0], brick['centre'][1], self.grip_z + height]
                self.mover.move(f'Hover {height * 100:.0f} cm above the brick', target, yaw=brick['yaw'])
                better = self.settle_and_watch(f'Servo {height * 100:.0f} cm', near=brick['centre'])
                if better is None:
                    raise RuntimeError('Lost the brick while approaching; stopping')
                shift = float(np.linalg.norm(better['centre'][:2] - brick['centre'][:2]))
                brick = better
                if shift <= self.c['servo_tolerance']:
                    self.say(f'  settled (moved {shift * 1000:.1f} mm since last look)')
                    break
                self.say(f'  estimate moved {shift * 1000:.1f} mm, re-aiming ({attempt})')
        move = float(np.linalg.norm(brick['centre'][:2] - first['centre'][:2]))
        self.say(f'Camera refinement corrected the first estimate by {move * 1000:.1f} mm')
        self.save_debug('lego_auto_last_look.png')
        return brick

    def pick_and_place(self, brick):
        self.aim(brick)
        pick, place = self.cfg['pick'], self.cfg['place']
        yaw_pick, yaw_place = self.cfg['pick_yaw'], self.cfg['place_yaw']
        above_pick = [pick[0], pick[1], pick[2] + self.c['clearance']]
        above_place = [place[0], place[1], place[2] + self.c['clearance']]
        self.mover.move('Lower to the brick', pick, yaw=yaw_pick)
        self.mover.hand(close=True)
        self.say('Grasp reported success')
        self.mover.attach_block()
        self.mover.move('Lift the brick', above_pick, yaw=yaw_pick)
        self.mover.move('Carry to the drop position', above_place, yaw=yaw_place)
        self.mover.move('Lower to the table', place, yaw=yaw_place)
        self.mover.hand()
        self.mover.detach_block()
        self.mover.move('Retreat', above_place, yaw=yaw_place)

    def verify(self):
        self.mover.move('Back to the scan pose', self.c['scan_poses'][0])
        brick = self.settle_and_watch('Verify')
        if brick is None:
            self.say('Could not see the brick afterwards (it may be outside the allowed window).')
            return
        error = float(np.linalg.norm(brick['centre'][:2] - np.array(self.place_xy)))
        self.say(f'Brick is {error * 1000:.1f} mm from the requested drop position')

    def run(self):
        self.mover.hand()
        brick = self.search()
        brick = self.servo(brick)
        self.pick_and_place(brick)
        self.verify()
        self.say('MISSION COMPLETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--place-xy', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--yes', action='store_true',
                        help='Skip the GO prompt (only honoured when sim_world is running)')
    parser.add_argument('--step', action='store_true', help='Ask before every move (default on a real robot)')
    parser.add_argument('--no-step', action='store_true', help='Never ask before moves')
    args, ros_args = parser.parse_known_args()
    if args.step and args.no_step:
        raise ValueError('Choose either --step or --no-step')
    c = yaml.safe_load(open(args.config, encoding='utf-8'))
    place_xy = args.place_xy or c.get('place_xy')
    if not place_xy:
        raise ValueError('Give the drop position with --place-xy X Y (metres, in fr3_link0)')
    calibration = json.load(open(c['calibration_file'], encoding='utf-8'))
    if calibration['child_frame'] != c['camera_frame'] or calibration['parent_frame'] != c['ee_frame']:
        raise ValueError('Calibration frames do not match the config')
    if calibration['status'] != 'passed_quality_gates' and not c.get('allow_rejected_calibration', False):
        raise ValueError('Calibration did not pass its quality gates; see allow_rejected_calibration')

    import rclpy
    from pick_place import validate
    rclpy.init(args=ros_args)
    node = rclpy.create_node('lego_auto')
    try:
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
        simulated = 'sim_world' in node.get_node_names()
        if simulated:
            c['velocity_scale'] = c['acceleration_scale'] = float(c.get('sim_speed_scale', 0.1))
            print('SIMULATION: sim_world found, using speed scale', c['velocity_scale'])
        else:
            print(f'REAL ROBOT. Calibration status: {calibration["status"]}.')
        mission = Mission(node, c, calibration, place_xy, validate)
        print('This will MOVE THE ARM AND HAND, searching for the brick and carrying it to '
              f'x={place_xy[0]:.3f} y={place_xy[1]:.3f}.')
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
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
