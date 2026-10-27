#!/usr/bin/python3 -s
"""Move the hand to a pose recorded with record_pose.py (default: observe), with the scripts' straight, collision-checked lines.

  ros2 run fr3_pose_pick go_to_pose.py observe --config src/fr3_pose_pick/config/pose_pick.yaml

First straight up to the height of the pose if it is higher (never down first, so nothing below is swept), then one straight line to it,
at transit_velocity_scale, the hand pointing down all the way. The hand ends at the recorded angle itself, not the one turned 180 deg:
the camera is on the hand, so the picture must be taken the same way round. The hand is not opened or closed. Refused: a hand that is
tilted at the start, a pose that is tilted, a move that would swing a joint more than max_joint_swing, an arrival more than 5 mm or
5 deg off.
"""
import argparse
import math
import sys

import numpy as np

import pose_pick as pp
import record_pose as rp

MAX_ARRIVAL_ERROR = 0.005      # m
MAX_YAW_ERROR = 5.0            # deg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', nargs='?', default='observe', help='the recorded pose (default: observe)')
    parser.add_argument('--config', default='src/fr3_pose_pick/config/pose_pick.yaml')
    args = parser.parse_args(argv)
    import yaml
    try:
        c = pp.apply_baseplate(yaml.safe_load(open(args.config, encoding='utf-8')))
        record = rp.load_poses(rp.poses_path(args.config)).get(args.name)
        if record is None:
            raise ValueError(f'No pose called {args.name!r}: hold the arm there and run  record_pose.py {args.name} --config {args.config}')
        if float(record.get('tilt_deg', 0.0)) > float(c['max_tilt_deg']):
            raise ValueError(f'The pose {args.name!r} is tilted {record["tilt_deg"]:.1f} deg from pointing down: record it again')
    except (ValueError, OSError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 2
    target, yaw = [float(v) for v in record['xyz']], math.radians(float(record.get('yaw_deg', 0.0)))

    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener
    import smooth_motion as sm
    rclpy.init()
    node = rclpy.create_node('go_to_pose')
    buffer = Buffer(cache_time=Duration(seconds=30))
    listener = TransformListener(buffer, node)         # keep a reference alive
    try:
        sm.MAX_JOINT_SWING = float(c['max_joint_swing'])
        sm.MAX_WRIST_SWING = float(c['max_wrist_swing'])
        mover = sm.Mover(node, pp.scene_config(c), buffer, None)
        mover.prepare_scene()
        transit = float(c.get('transit_velocity_scale') or c['velocity_scale'])
        now = mover.hand_pose()
        tilt = pp.hand_tilt_deg(now[:3, :3])
        if tilt > c['max_tilt_deg']:
            raise RuntimeError(f'The hand is tilted {tilt:.0f} deg from pointing down: move it to a ready pose first. Nothing was moved.')
        if now[2, 3] < target[2] - 0.005:
            mover.scale = transit                         # straight up stays at the transit speed (fast, it failed: pose_pick_auto.FAST)
            mover.move(f'Straight up to the height of {args.name}', [now[0, 3], now[1, 3], target[2]])
        mover.scale = float(c.get('fast_velocity_scale') or transit)
        nearest = sm.nearest_yaw
        sm.nearest_yaw = lambda axis_yaw, current_yaw: axis_yaw        # this yaw itself, not the jaw-equivalent one 180 deg away
        try:
            mover.move(f'To {args.name}', target, yaw)
        finally:
            sm.nearest_yaw = nearest
        hand = mover.hand_pose()
        off = float(np.linalg.norm(hand[:3, 3] - target))
        turn = abs((math.degrees(sm.hand_yaw(hand[:3, :3]) - yaw) + 180.0) % 360.0 - 180.0)
        if off > MAX_ARRIVAL_ERROR or turn > MAX_YAW_ERROR:
            raise RuntimeError(f'Arrived {off * 1000:.1f} mm and {turn:.1f} deg from {args.name}: more than {MAX_ARRIVAL_ERROR * 1000:.0f} mm '
                               f'/ {MAX_YAW_ERROR:.0f} deg. The arm stays there.')
        print(f'At {args.name}: {off * 1000:.1f} mm and {turn:.1f} deg from the recorded pose.', flush=True)
        return 0
    except (RuntimeError, TimeoutError, ValueError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 1
    finally:
        del listener
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
