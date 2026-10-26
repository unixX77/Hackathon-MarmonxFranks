#!/usr/bin/python3 -s
"""Measure what the hand reads when the (printed) fingertips touch each other: hand_zero_width.

With the stock fingertips the hand reads about 0 mm when the fingers touch. Printed tips of thickness t on
each finger make it read 2 t, and every width in the config (open_width, grasp_width) is the hand's own
reading, so it includes that: the brick (16 mm) is gripped at 2 t + 16 mm, not at 16 mm.

This closes the hand GENTLY (5 N) with NOTHING between the fingers, waits, reads the finger joints, prints the
value for the config and the open_width / grasp_width that follow from it, and opens the hand again.
Keep your fingers and every brick out of the hand. Needs the robot software (the Franka hand actions) running.

    ros2 run fr3_pose_pick measure_hand_zero.py            (asks you to type YES first)
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import yaml


# ------------------------------------------------------------------ pure logic (no ROS)
def suggest_widths(zero, brick_short, room=0.002, squeeze=0.001):
    """open_width and grasp_width for a hand that reads `zero` with the tips touching: the open hand has `room`
    (total, both sides) more than the brick, and the closed hand squeezes the brick by `squeeze`."""
    return {'hand_zero_width': zero, 'open_width': zero + brick_short + room, 'grasp_width': zero + brick_short - squeeze}


def format_report(zero, brick_short):
    s = suggest_widths(zero, brick_short)
    lines = [f'With the tips touching the hand reads {zero * 1000:.1f} mm (the tips are about {zero * 500:.1f} mm thick each).',
             'Put these in config/pose_pick.yaml:',
             f'  hand_zero_width: {s["hand_zero_width"]:.4f}',
             f'  open_width: {s["open_width"]:.4f}      # brick {brick_short * 1000:.0f} mm + tips + 1 mm of room per side',
             f'  grasp_width: {s["grasp_width"]:.4f}     # squeezes the brick by 1 mm']
    if zero > 0.030:
        lines.append('WARNING: more than 30 mm is surprising for tips: was something between the fingers?')
    if zero < -0.001:
        lines.append('WARNING: negative: the hand is not homed. Home the hand and measure again.')
    return lines


# --------------------------------------------------------------------------- ROS part
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent.parent / 'config' / 'pose_pick.yaml'))
    parser.add_argument('--force', type=float, default=5.0, help='closing force in N (default 5)')
    parser.add_argument('--yes', action='store_true', help='do not ask (only for a simulated hand)')
    args, ros_args = parser.parse_known_args()
    if not 0 < args.force <= 10:
        sys.exit('--force must be between 0 and 10 N')
    brick_short = yaml.safe_load(open(args.config, encoding='utf-8'))['block_size'][1]
    if not args.yes:
        print('The hand will CLOSE with', args.force, 'N. Nothing must be between the fingers.')
        if input('Type YES to close the hand: ').strip() != 'YES':
            sys.exit('Cancelled; the hand was not moved.')

    import rclpy
    from franka_msgs.action import Grasp, Move
    from rclpy.action import ActionClient
    from sensor_msgs.msg import JointState
    rclpy.init(args=ros_args)
    node = rclpy.create_node('measure_hand_zero')
    readings = []
    node.create_subscription(JointState, '/franka_gripper/joint_states',
                             lambda m: readings.append(float(sum(m.position[:2]))) if len(m.position) >= 2 else None, 10)
    try:
        grasp, move = ActionClient(node, Grasp, 'franka_gripper/grasp'), ActionClient(node, Move, 'franka_gripper/move')
        if not (grasp.wait_for_server(timeout_sec=10) and move.wait_for_server(timeout_sec=10)):
            sys.exit('The Franka hand actions are not available: is the robot software running?')
        goal = Grasp.Goal()
        goal.width, goal.speed, goal.force = 0.0, 0.02, float(args.force)
        goal.epsilon.inner, goal.epsilon.outer = 0.0, 0.08          # stopping anywhere on the way counts as success
        handle_future = grasp.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, handle_future, timeout_sec=10)
        handle = handle_future.result()
        if handle is None or not handle.accepted:
            sys.exit('The hand refused the close command.')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=30)
        if not result_future.done():
            sys.exit('The hand did not finish closing in 30 s.')
        del readings[:]
        end = time.monotonic() + 1.5                                   # let the fingers settle, then read
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        if len(readings) < 5:
            sys.exit('No finger readings on /franka_gripper/joint_states.')
        zero = statistics.median(readings)
        print('\n'.join(format_report(zero, brick_short)))
        open_goal = Move.Goal()
        open_goal.width, open_goal.speed = 0.05, 0.05                  # open again; leave the hand open
        future = move.send_goal_async(open_goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10)
        if future.result() is not None and future.result().accepted:
            rclpy.spin_until_future_complete(node, future.result().get_result_async(), timeout_sec=15)
        print('The hand is open again.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
