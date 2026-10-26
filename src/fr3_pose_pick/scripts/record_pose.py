#!/usr/bin/python3 -s
"""Record the pose the arm is in RIGHT NOW under a name (default: bin) in poses.yaml next to the config.

Move the arm where you want it (hand pointing down), hold it still, and:

  ros2 run fr3_pose_pick record_pose.py bin --config src/fr3_pose_pick/config/pose_pick.yaml

It reads the hand pose (fr3_hand_tcp in fr3_link0) from TF about twenty times over two seconds, refuses if the arm is moving (the
samples spread by more than 1 mm), and saves the average: position, quaternion, the yaw and how far the hand is tilted from
pointing down. Nothing is commanded: this only reads. The file is written next to the config, so the pose is permanent and in git.

  --show           print the recorded poses and exit
  --delete NAME    forget a pose

pose_pick_palletize.py uses the pose called "bin" for the BAD bricks: it carries the brick there, lets go and goes back to the good ones.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

POSES_FILE = 'poses.yaml'
MAX_SPREAD = 0.001            # m: the samples may differ by this much (the arm must be still)
SAMPLES = 20
MAX_REACH = 0.80              # m from the base axis: the FR3 reaches 0.855 m at full stretch
MAX_BIN_Z = 0.55              # m: the same ceiling as max_hover_z


def poses_path(config_path):
    return Path(config_path).with_name(POSES_FILE)


def load_poses(path):
    """{name: record}. An absent file is an empty dict; a broken one is an error, never silently ignored."""
    import yaml
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        assert isinstance(data, dict) and all(isinstance(v, dict) and len(v.get('xyz', [])) == 3 for v in data.values())
    except (yaml.YAMLError, AssertionError, AttributeError, TypeError):
        raise ValueError(f'{path} is damaged: every pose needs a name and an xyz. Fix it, or record the pose again.') from None
    return data


def save_pose(path, name, record):
    """Add or replace one pose in the file, keeping the others."""
    import yaml
    path = Path(path)
    data = load_poses(path)
    data[name] = record
    path.write_text('# Poses recorded with record_pose.py (hand = fr3_hand_tcp in fr3_link0, metres and degrees). Do not edit by hand\n'
                    '# unless you know what you are doing: record the pose again instead.\n'
                    + yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
    return data


def pose_record(samples):
    """samples: list of 4x4 hand poses. Returns the record of their average, or raises ValueError when the arm was moving."""
    from scipy.spatial.transform import Rotation
    positions = np.array([s[:3, 3] for s in samples], dtype=float)
    spread = float(np.max(np.ptp(positions, axis=0)))
    if spread > MAX_SPREAD:
        raise ValueError(f'The arm is moving: the hand position varied by {spread * 1000:.1f} mm during the reading (limit '
                         f'{MAX_SPREAD * 1000:.0f} mm). Hold it still, wait a second, and try again.')
    mean = positions.mean(axis=0)
    rotations = Rotation.from_matrix(np.array([s[:3, :3] for s in samples]))
    rotation = rotations.mean()
    matrix = rotation.as_matrix()
    quat = rotation.as_quat()
    if quat[3] < 0:
        quat = -quat
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -matrix[2, 2]))))        # 0 = the hand's z points straight down
    yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))                # the jaw-axis angle of a hand pointing down
    return {'xyz': [round(float(v), 5) for v in mean], 'quat_xyzw': [round(float(v), 5) for v in quat],
            'yaw_deg': round(yaw, 3), 'tilt_deg': round(tilt, 3), 'spread_mm': round(spread * 1000, 3),
            'recorded': time.strftime('%Y-%m-%d %H:%M:%S')}


def check_bin_pose(c, record):
    """Is the recorded pose one the palletizer can use for the bad bricks? Raises ValueError saying what is wrong."""
    x, y, z = record['xyz']
    tilt = float(record.get('tilt_deg', 0.0))
    if tilt > float(c.get('max_tilt_deg', 10)):
        raise ValueError(f'The bin pose has the hand tilted {tilt:.0f} deg from pointing down (limit {c.get("max_tilt_deg", 10)} deg): '
                         'the straight-line moves keep the hand pointing down. Record it again with the hand pointing down.')
    if math.hypot(x, y) > MAX_REACH or x < 0.05:
        raise ValueError(f'The bin pose x={x:.3f} y={y:.3f} is out of reach (more than {MAX_REACH} m from the base, or behind it)')
    if z < c['table_top_z'] + 0.03:
        raise ValueError(f'The bin pose is z={z:.3f}: too close to the table (surface at {c["table_top_z"]}); the hand would hit it')
    if z > MAX_BIN_Z:
        raise ValueError(f'The bin pose is z={z:.3f}, above the ceiling of {MAX_BIN_Z} m')


def read_samples(count=SAMPLES, interval=0.1):
    """The hand pose from TF `count` times (needs the robot software running). Only reads."""
    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener
    import pose_pick as pp
    rclpy.init()
    node = rclpy.create_node('record_pose')
    buffer = Buffer(cache_time=Duration(seconds=30))
    listener = TransformListener(buffer, node)
    config = {'base_frame': 'fr3_link0', 'ee_frame': 'fr3_hand_tcp'}
    try:
        samples = []
        for _ in range(count):
            samples.append(pp.lookup_hand(node, buffer, config))
            end = time.monotonic() + interval
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.02)
        return samples
    finally:
        del listener
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', nargs='?', default='bin', help='the name to record the pose under (default: bin)')
    parser.add_argument('--config', default='src/fr3_pose_pick/config/pose_pick.yaml',
                        help='the config; the poses are stored in poses.yaml next to it')
    parser.add_argument('--show', action='store_true', help='print the recorded poses and exit')
    parser.add_argument('--delete', metavar='NAME', help='forget a recorded pose and exit')
    args = parser.parse_args(argv)
    path = poses_path(args.config)
    try:
        if args.show or args.delete:
            data = load_poses(path)
            if args.delete:
                if args.delete not in data:
                    raise ValueError(f'There is no pose called {args.delete!r} in {path}')
                del data[args.delete]
                import yaml
                path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
                print(f'Forgot {args.delete!r}.')
            for name, record in data.items():
                print(f'{name}: x={record["xyz"][0]:.4f} y={record["xyz"][1]:.4f} z={record["xyz"][2]:.4f}  yaw={record.get("yaw_deg", 0):.2f} deg  '
                      f'tilt={record.get("tilt_deg", 0):.2f} deg  recorded {record.get("recorded", "?")}')
            if not data:
                print(f'No poses recorded in {path}.')
            return 0
        record = pose_record(read_samples())
        save_pose(path, args.name, record)
    except (ValueError, RuntimeError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 1
    print(f'Recorded {args.name!r} in {path}:\n  hand x={record["xyz"][0]:.4f} y={record["xyz"][1]:.4f} z={record["xyz"][2]:.4f} m, '
          f'yaw {record["yaw_deg"]:.2f} deg, tilted {record["tilt_deg"]:.2f} deg from pointing down '
          f'(the readings varied by {record["spread_mm"]:.2f} mm)')
    if record['tilt_deg'] > 3:
        print(f'  NOTE: the hand is tilted {record["tilt_deg"]:.1f} deg: the robot will keep it pointing straight down when it goes there.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
