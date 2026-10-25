#!/usr/bin/python3 -s
"""SIMULATION ONLY: a fake Franka Hand for the pose-driven pick and place. No brick, no camera.

The fake-hardware stack has an arm but nothing that answers the hand's actions. This node serves
franka_gripper/move and franka_gripper/grasp and publishes the finger joint states, so the hand
visibly opens and closes in RViz. There is nothing to grip, so a grasp simply closes to the width it
was asked for and reports success; a real hand only reports success when it closed on something.
The node is named 'sim_gripper': pose_pick.py takes that as "this is a simulation".
"""
import os
import signal
import threading
import time

import rclpy
from franka_msgs.action import Grasp, Move
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

FINGER_JOINTS = ['fr3_finger_joint1', 'fr3_finger_joint2']
OPEN_WIDTH = 0.08


def silence_fake_gripper():
    """The stock fake gripper publisher reports 'always open' on the same topic and would fight
    with the animated one. Stop it, but only this user's copy running in this ROS domain."""
    mine = os.environ.get('ROS_DOMAIN_ID', '')
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f'/proc/{entry}/cmdline', 'rb') as f:
                cmd = f.read().decode(errors='ignore')
            if 'fake_gripper_state_publisher' not in cmd:
                continue
            with open(f'/proc/{entry}/environ', 'rb') as f:
                env = dict(item.split(b'=', 1) for item in f.read().split(b'\0') if b'=' in item)
            if env.get(b'ROS_DOMAIN_ID', b'').decode() == mine:
                os.kill(int(entry), signal.SIGTERM)
                print(f'Stopped the stock fake gripper publisher (pid {entry}) so the hand can animate.')
        except (OSError, ValueError):
            continue


class SimGripper(Node):
    def __init__(self):
        super().__init__('sim_gripper')
        cb = ReentrantCallbackGroup()
        self.lock = threading.RLock()
        self.width, self.width_goal, self.width_speed = OPEN_WIDTH, OPEN_WIDTH, 0.05
        self.finger_pub = self.create_publisher(JointState, '/franka_gripper/joint_states', 10)
        self.create_timer(0.02, self.step_fingers, callback_group=cb)
        ActionServer(self, Move, 'franka_gripper/move', self.on_move, callback_group=cb)
        ActionServer(self, Grasp, 'franka_gripper/grasp', self.on_grasp, callback_group=cb)
        self.get_logger().info('Simulated Franka Hand ready (no brick: a grasp closes to the requested width).')

    def step_fingers(self):
        with self.lock:
            step = self.width_speed * 0.02
            self.width += max(-step, min(step, self.width_goal - self.width))
            half = self.width / 2
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = FINGER_JOINTS, [half, half]
        msg.velocity, msg.effort = [0.0, 0.0], [0.0, 0.0]
        self.finger_pub.publish(msg)

    def go_to(self, width, speed):
        with self.lock:
            self.width_goal = max(0.0, min(OPEN_WIDTH, float(width)))
            self.width_speed = float(speed) if speed > 0 else 0.05
        end = time.monotonic() + 20.0
        while time.monotonic() < end:
            with self.lock:
                if abs(self.width - self.width_goal) < 5e-4:
                    return
            time.sleep(0.02)

    def on_move(self, goal_handle):
        goal = goal_handle.request
        self.go_to(goal.width, goal.speed)
        goal_handle.succeed()
        return Move.Result(success=True)

    def on_grasp(self, goal_handle):
        goal = goal_handle.request
        self.go_to(goal.width, goal.speed)
        self.get_logger().info(f'GRASP: closed to {goal.width * 1000:.0f} mm (simulated: nothing is checked)')
        goal_handle.succeed()
        return Grasp.Result(success=True)


def main():
    rclpy.init()
    silence_fake_gripper()
    node = SimGripper()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
