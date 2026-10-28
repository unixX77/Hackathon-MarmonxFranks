#!/usr/bin/python3 -s
"""LISTEN for the bricks published by brick_publisher.py and palletize them: every message becomes one pose_pick_palletize.py run.

  ros2 run fr3_pose_pick palletize_listener.py --config src/fr3_pose_pick/config/pose_pick.yaml
  ros2 run fr3_pose_pick palletize_listener.py --config src/fr3_pose_pick/config/pose_pick.yaml -- --countdown 10   # after -- : more
                                                                                          # options for pose_pick_palletize.py

Each message is run as

  pose_pick_palletize.py --config CONFIG --ee-pose <the hand pose of the picture> --brick X Y Z A G|B --brick ...

so the bricks are picked from the pose the picture was really taken from, wherever the arm is when the run starts; G goes to the good
row, B to the bin, exactly as when you type the command. Messages are done one after the other, in the order they arrived. Run it from
the folder you run pose_pick_palletize.py from (the repository root): the row files pose_pick_palletize*.json are kept there.

A run that fails stops the listener (exit code of the run): the arm stays where it is, as after a failed pose_pick_palletize.py,
and no later message moves it until you look and start the listener again. Messages published while it is not running are lost.

Own options:
  --config FILE   pose_pick.yaml (required)
  --topic NAME    default /brick_poses
  --once          do one message and exit
  --dry-run       print the command of each message, move nothing
  --all-layers    pick every brick of the message; by default only the TOP layer is picked: a brick whose camera depth is more
                  than half a brick height (4.8 mm) deeper than the nearest brick of the picture is left out
"""
import argparse
import json
import queue
import subprocess
import sys
from pathlib import Path

TOPIC = '/brick_poses'
PALLETIZE = Path(__file__).resolve().with_name('pose_pick_palletize.py')
TOP_LAYER_MARGIN = 0.5         # layers: a brick this much lower than the highest one is not on the top layer


def fmt(value):
    return f'{float(value):.6f}'


def top_layer_only(message, config):
    """Keep only the bricks of the highest layer of the picture. Compared by the camera's own depth (camera_xyz z: the hand points
    down, so the camera looks straight at the layers): the bricks of one layer read the same depth within about a millimetre, the
    layer below one brick height (9.6 mm) deeper. Not in the base frame: the hand-eye result and the hand's tilt add a slope of about
    2 deg there (6 mm over one picture of the 10-27 lattice, with every brick on the same layer). A brick more than half a brick height
    deeper than the nearest one is left out. Returns (kept, [(brick, mm below the top layer), ...] of all)."""
    import yaml
    height = float(yaml.safe_load(open(config, encoding='utf-8'))['block_size'][2])
    top = min(float(brick['camera_xyz'][2]) for brick in message['bricks'])
    below = [(brick, (float(brick['camera_xyz'][2]) - top) * 1000) for brick in message['bricks']]
    return [brick for brick, mm in below if mm <= TOP_LAYER_MARGIN * height * 1000], below


def command(message, config, extra):
    """The pose_pick_palletize.py arguments for one message. Raises ValueError for a message it cannot trust."""
    bricks = message.get('bricks') or []
    arguments = ['--config', config]
    ee_pose = message.get('ee_pose')
    if ee_pose is not None:
        if len(ee_pose) != 7:
            raise ValueError(f'ee_pose needs x y z qx qy qz qw, got {ee_pose}')
        arguments += ['--ee-pose', *map(fmt, ee_pose)]
    for brick in bricks:
        if brick.get('label') not in ('G', 'B') or len(brick.get('camera_xyz', [])) != 3:
            raise ValueError(f'brick {brick} needs camera_xyz and a label G or B')
        arguments += ['--brick', *map(fmt, brick['camera_xyz']), fmt(brick['angle_deg']), brick['label']]
    return arguments + extra


def main(argv=None):
    from rclpy.utilities import remove_ros_args
    argv = remove_ros_args(list(sys.argv if argv is None else ['palletize_listener.py', *argv]))[1:]
    if '-h' in argv or '--help' in argv:
        print(__doc__)
        return 0
    extra = []
    if '--' in argv:
        argv, extra = argv[:argv.index('--')], argv[argv.index('--') + 1:]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--topic', default=TOPIC)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--all-layers', action='store_true')
    args = parser.parse_args(argv)
    config = str(Path(args.config).resolve())
    if not Path(config).is_file():
        print(f'ERROR: no config at {config}', file=sys.stderr)
        return 2

    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node('palletize_listener')
    waiting = queue.Queue()
    node.create_subscription(String, args.topic, lambda message: waiting.put(message.data), 10)
    print(f'Listening on {args.topic}{" (dry run: nothing moves)" if args.dry_run else ""}. Ctrl+C stops.', flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if waiting.empty():
                continue
            text = waiting.get()
            try:
                message = json.loads(text)
                arguments = command(message, config, extra)
            except (ValueError, KeyError, TypeError) as problem:
                print(f'ERROR: a message that cannot be used, skipped: {problem}\n  {text[:300]}', file=sys.stderr, flush=True)
                continue
            bricks = message.get('bricks') or []
            print(f'\n=== {len(bricks)} brick{"s" if len(bricks) != 1 else ""} from {message.get("run") or "?"} '
                  f'({message.get("time", "?")}): ' + ', '.join(f'#{b.get("id", "?")} {b.get("colour", "")} {b["label"]}' for b in bricks)
                  + ' ===', flush=True)
            if not bricks:
                continue
            if message.get('ee_pose') is None:
                print('NOTE: the message has no hand pose: the arm pose of NOW is taken as the picture pose.', flush=True)
            if not args.all_layers:
                kept, below = top_layer_only(message, config)
                print('Depth below the top layer (camera, mm): ' + ', '.join(
                    f'#{b.get("id", "?")} {mm:.1f}{"" if b in kept else " LEFT OUT"}' for b, mm in below), flush=True)
                if len(kept) < len(bricks):
                    message = dict(message, bricks=kept)
                    arguments = command(message, config, extra)
                    print(f'{len(bricks) - len(kept)} brick{"s" if len(bricks) - len(kept) != 1 else ""} below the top layer left out: '
                          f'{len(kept)} to pick.', flush=True)
            run = [sys.executable, '-s', str(PALLETIZE), *arguments]
            print('Running: ' + ' '.join(run), flush=True)
            if args.dry_run:
                code = 0
            else:
                code = subprocess.run(run).returncode
            if code:
                print(f'\nLISTENER STOPPED: pose_pick_palletize.py ended with exit code {code}. The arm stays where it is; look at the '
                      'output above (and --status), then start the listener again.', file=sys.stderr, flush=True)
                return code
            if args.once:
                return 0
            print(f'\nDone. Listening on {args.topic} for the next picture.', flush=True)
        return 0
    except KeyboardInterrupt:
        print('\nListener stopped (Ctrl+C).')
        return 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
