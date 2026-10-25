#!/usr/bin/python3 -s
"""SIMULATION ONLY: print the exact pose of the simulated brick (what a perfect perception team
would hand over), as arguments for pose_pick.py:   --pose X Y --yaw DEG

Reads /sim_brick_marker from sim_world, so it only works next to the simulation. sim_world's own
HIDDEN TRUTH line is rounded to 1 mm / 1 degree, which is too coarse for judging accuracy.
"""
import math
import sys

import rclpy
from visualization_msgs.msg import Marker


def main():
    rclpy.init()
    node = rclpy.create_node('sim_truth')
    found = []

    def on_marker(m):
        if m.ns == 'sim_world' and m.id == 0 and not found:
            q, p = m.pose.orientation, m.pose.position
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2          # long axis repeats every 180 degrees
            found.append((p.x, p.y, p.z, math.degrees(yaw)))

    node.create_subscription(Marker, '/sim_brick_marker', on_marker, 10)
    end = node.get_clock().now().nanoseconds + 10 * 10 ** 9
    while not found and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.try_shutdown()
    if not found:
        sys.exit('No /sim_brick_marker: is the simulation (sim_world:=true) running in this ROS domain?')
    x, y, _, yaw = found[0]
    print(f'--pose {x:.5f} {y:.5f} --yaw {yaw:.3f}')


if __name__ == '__main__':
    main()
