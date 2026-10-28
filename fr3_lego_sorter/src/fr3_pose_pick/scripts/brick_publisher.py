#!/usr/bin/python3 -s
"""Run the camera pipeline (vision/run_pipeline_classifier.sh) and PUBLISH the bricks it found on a topic, for palletize_listener.py.

Start it with the arm at the pose the picture is taken from (hand pointing down, held still):

  ros2 run fr3_pose_pick brick_publisher.py --config src/fr3_pose_pick/config/pose_pick.yaml --review codex

Every option it does not know goes to run_pipeline_classifier (--review codex, --second-opinion codex, --recording DIR --frame N, ...).
What it does:
  1. reads the hand pose (fr3_hand_tcp in fr3_link0) from TF, about ten times over a second, refused if the arm is moving;
  2. runs the pipeline (the D435 capture window, SAM3, the classifier, the review) and shows its output as it goes;
  3. reads the hand pose again: if the arm moved by more than 1 mm or 0.5 deg the pictures do not belong to one pose and NOTHING is published;
  4. takes the final brick lines  #ID COLOUR good|bad x=.. y=.. z=.. angle=..  (G for good, B for bad; a brick with neither is left out
     and said so, it must not sort silently) and publishes them once, as JSON in a std_msgs/String, on --topic (default /brick_poses):

     {"run": final result folder, "time": ..., "frame": "fr3_link0", "ee_pose": [x, y, z, qx, qy, qz, qw],
      "bricks": [{"id": 1, "colour": "red", "quality": "good", "label": "G", "camera_xyz": [x, y, z], "angle_deg": 80.0,
                  "note": "CHECK: ..." or ""}, ...]}

Own options:
  --config FILE      pose_pick.yaml (for base_frame / ee_frame)
  --pipeline FILE    run_pipeline_classifier.sh (default vision/run_pipeline_classifier.sh of this repository)
  --topic NAME       default /brick_poses
  --ee-pose X Y Z QX QY QZ QW   the hand pose of the picture, instead of TF; needed with --recording (an old picture)
  --wait SECONDS     how long to wait for a listener before giving up (default 30)
  --manual-capture   show the capture window (Enter saves, q quits) instead of taking one frame by itself after about a second
Nothing here moves the arm.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

import record_pose as rp

TOPIC = '/brick_poses'
PIPELINE = Path(__file__).resolve().parents[3] / 'vision' / 'run_pipeline_classifier.sh'   # through the --symlink-install link
SAMPLES = 10
MAX_MOVE = 0.001               # m: the hand may differ by this much between before and after the pipeline
MAX_TURN_DEG = 0.5
MAX_TF_AGE = 2.0               # s: an older hand pose means the joint states stopped
HEADER = 'bricks. x y z = top-face centre'
LINE = re.compile(r'^#(\d+) (.*?) ?x=(\S+) y=(\S+) z=(\S+) angle=(\S+)(.*)$')
LABELS = {'good': 'G', 'bad': 'B'}


def parse_bricks(lines):
    """The final brick lines of the pipeline output -> (bricks, problems). Only the lines after its header count."""
    bricks, problems, started = [], [], False
    for line in lines:
        if HEADER in line:
            started, bricks, problems = True, [], []     # a second header: only the last list counts
            continue
        match = LINE.match(line.strip()) if started else None
        if not match:
            continue
        number, words, x, y, z, angle, rest = match.groups()
        words = words.split()
        quality = words[-1] if words else ''
        if quality not in LABELS:
            problems.append(f'brick #{number} has no good / bad ({line.strip()!r}): left out')
            continue
        bricks.append({'id': int(number), 'colour': words[0] if len(words) > 1 else '', 'quality': quality,
                       'label': LABELS[quality], 'camera_xyz': [float(x), float(y), float(z)], 'angle_deg': float(angle),
                       'note': rest.strip()})
    return bricks, problems, started


def run_pipeline(pipeline, args):
    """Run the pipeline, show its output, return (exit code, the output lines, the final result folder)."""
    print('Running: ' + ' '.join([str(pipeline), *args]), flush=True)
    process = subprocess.Popen([str(pipeline), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env=dict(os.environ, PYTHONUNBUFFERED='1'))
    lines, final = [], ''
    for line in process.stdout:
        print(line, end='', flush=True)
        lines.append(line)
        if line.startswith('Final results:'):
            final = line.split(':', 1)[1].strip()
    return process.wait(), lines, final


# ------------------------------------------------------------------------------------------------- ROS part
def hand_samples(node, buffer, c, count=SAMPLES, interval=0.1):
    """The hand pose from TF `count` times (4x4). The node is spun by another thread; a pose older than MAX_TF_AGE is refused."""
    from rclpy.time import Time
    from scipy.spatial.transform import Rotation
    samples, end = [], time.monotonic() + 10.0
    while len(samples) < count:
        if buffer.can_transform(c['base_frame'], c['ee_frame'], Time()):
            stamped = buffer.lookup_transform(c['base_frame'], c['ee_frame'], Time())
            age = (node.get_clock().now() - Time.from_msg(stamped.header.stamp)).nanoseconds * 1e-9
            if age > MAX_TF_AGE:
                raise RuntimeError(f'The hand pose from TF is {age:.0f} s old: are the joint states running?')
            tf = stamped.transform
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix()
            pose[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
            samples.append(pose)
        elif time.monotonic() > end:
            raise RuntimeError('No hand pose from TF: is the robot (pose_bringup.launch.py) running?')
        time.sleep(interval)
    return samples


def turn_deg(quat_a, quat_b):
    dot = abs(float(np.dot(quat_a, quat_b)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def main(argv=None):
    from rclpy.utilities import remove_ros_args
    argv = remove_ros_args(list(sys.argv if argv is None else ['brick_publisher.py', *argv]))[1:]
    if '-h' in argv or '--help' in argv:
        print(__doc__)
        return 0
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    own.add_argument('--config', default='src/fr3_pose_pick/config/pose_pick.yaml')
    own.add_argument('--pipeline', type=Path, default=PIPELINE)
    own.add_argument('--topic', default=TOPIC)
    own.add_argument('--ee-pose', nargs=7, type=float, metavar=('X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW'))
    own.add_argument('--wait', type=float, default=30.0)
    own.add_argument('--manual-capture', action='store_true')
    args, pipeline_args = own.parse_known_args(argv)
    if not args.manual_capture and not any(a.split('=', 1)[0] == '--recording' for a in pipeline_args):
        pipeline_args = ['--auto-capture', *pipeline_args]           # no window, no Enter / q
    if not args.pipeline.is_file():
        print(f'ERROR: no pipeline at {args.pipeline}: give it with --pipeline', file=sys.stderr)
        return 2
    if any(a.split('=', 1)[0] == '--recording' for a in pipeline_args) and args.ee_pose is None:
        print('ERROR: --recording uses an old picture, which was not taken from the arm pose of now: give the hand pose of that '
              'picture with --ee-pose X Y Z QX QY QZ QW', file=sys.stderr)
        return 2
    import yaml
    c = yaml.safe_load(open(args.config, encoding='utf-8'))

    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = rclpy.create_node('brick_publisher')
    buffer = Buffer(cache_time=Duration(seconds=30))
    listener = TransformListener(buffer, node)         # keep a reference alive
    publisher = node.create_publisher(String, args.topic, 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:            # the shutdown at the end
            pass
    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        before = None
        if args.ee_pose is None:
            before = rp.pose_record(hand_samples(node, buffer, c))
            print(f'Hand at the picture: x={before["xyz"][0]:.4f} y={before["xyz"][1]:.4f} z={before["xyz"][2]:.4f} '
                  f'yaw {before["yaw_deg"]:.1f} deg. Keep the arm still until the pipeline is done.', flush=True)
        code, lines, final = run_pipeline(args.pipeline, pipeline_args)
        if code:
            raise RuntimeError(f'the pipeline failed (exit code {code}): nothing published')
        if before is not None:
            after = rp.pose_record(hand_samples(node, buffer, c))
            moved = float(np.linalg.norm(np.subtract(after['xyz'], before['xyz'])))
            turned = turn_deg(before['quat_xyzw'], after['quat_xyzw'])
            if moved > MAX_MOVE or turned > MAX_TURN_DEG:
                raise RuntimeError(f'the arm moved during the pipeline ({moved * 1000:.1f} mm, {turned:.1f} deg): the picture pose is '
                                   'not known, nothing published. Hold the arm still and run again.')
            ee_pose = list(before['xyz']) + list(before['quat_xyzw'])
        else:
            ee_pose = list(args.ee_pose)
        bricks, problems, found = parse_bricks(lines)
        if not found:
            raise RuntimeError('the pipeline output has no brick list: nothing published')
        for problem in problems:
            print('WARNING: ' + problem, flush=True)
        if not bricks:
            print('No brick to publish.')
            return 0
        message = {'run': final, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'frame': c['base_frame'], 'ee_pose': ee_pose,
                   'bricks': bricks}
        end = time.monotonic() + args.wait
        while publisher.get_subscription_count() == 0:
            if time.monotonic() > end:
                raise RuntimeError(f'no listener on {args.topic} after {args.wait:.0f} s (start palletize_listener.py): nothing published')
            time.sleep(0.1)
        publisher.publish(String(data=json.dumps(message)))
        time.sleep(1.0)                                  # let the message go out before the node is gone
        print(f'\nPublished {len(bricks)} brick{"s" if len(bricks) != 1 else ""} on {args.topic}: '
              + ', '.join(f'#{b["id"]} {b["label"]}' for b in bricks))
        return 0
    except KeyboardInterrupt:
        print('\nCANCELLED (Ctrl+C): nothing published.')
        return 130
    except (RuntimeError, ValueError, OSError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 1
    finally:
        rclpy.try_shutdown()                             # wakes the spinning thread first: destroying under it aborts
        spinner.join(timeout=2.0)
        del listener
        node.destroy_node()


if __name__ == '__main__':
    sys.exit(main())
