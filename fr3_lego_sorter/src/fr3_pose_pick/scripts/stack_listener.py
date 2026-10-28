#!/usr/bin/python3 -s
"""LISTEN for the bricks published by brick_publisher.py and sort them BY COLOUR into stacks (pose_pick_stacks.py).

  ros2 run fr3_pose_pick stack_listener.py --config src/fr3_pose_pick/config/pose_pick.yaml

Exactly palletize_listener.py (the same options: --topic, --once, --dry-run, --all-layers, -- options for the script; only the top layer
of each picture is picked), but every message becomes one pose_pick_stacks.py run, with each brick's colour: a good brick goes onto the
stack of its colour (colour_stacks in the config), a bad one into the bin. Run it from the repository root: the stacks are remembered in
pose_pick_stacks.json there (--status / --reset with pose_pick_stacks.py).

It refuses to start while anything else listens on the topic: palletize_listener.py and this one must never both get the same bricks.
Its messages still name pose_pick_palletize.py in a few places (they come from palletize_listener.py, which is used unchanged).
"""
import sys
from pathlib import Path

import palletize_listener as listener

STACKS = Path(__file__).resolve().with_name('pose_pick_stacks.py')


def command(message, config, extra):
    """palletize_listener.command, with each brick's colour after its label (pose_pick_stacks.py --brick X Y Z A G COLOUR)."""
    arguments = ['--config', config]
    ee_pose = message.get('ee_pose')
    if ee_pose is not None:
        if len(ee_pose) != 7:
            raise ValueError(f'ee_pose needs x y z qx qy qz qw, got {ee_pose}')
        arguments += ['--ee-pose', *map(listener.fmt, ee_pose)]
    for brick in message.get('bricks') or []:
        if brick.get('label') not in ('G', 'B') or len(brick.get('camera_xyz', [])) != 3:
            raise ValueError(f'brick {brick} needs camera_xyz and a label G or B')
        colour = str(brick.get('colour') or 'unknown').replace(' ', '_')
        arguments += ['--brick', *map(listener.fmt, brick['camera_xyz']), listener.fmt(brick['angle_deg']), brick['label'], colour]
    return arguments + extra


def someone_else_listening(topic):
    import time
    import rclpy
    rclpy.init()
    node = rclpy.create_node('stack_listener_check')
    try:
        end = time.monotonic() + 2.0                     # DDS discovery takes a moment
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.count_subscribers(topic)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    if '-h' in sys.argv or '--help' in sys.argv:
        print(__doc__)
        return 0
    topic = sys.argv[sys.argv.index('--topic') + 1] if '--topic' in sys.argv[:-1] else listener.TOPIC
    others = someone_else_listening(topic)
    if others:
        print(f'ERROR: {others} other listener(s) on {topic} (palletize_listener.py?): stop it first, the same bricks must not be '
              'given to two scripts. Nothing was started.', file=sys.stderr)
        return 2
    listener.PALLETIZE, listener.command = STACKS, command
    return listener.main()


if __name__ == '__main__':
    sys.exit(main())
