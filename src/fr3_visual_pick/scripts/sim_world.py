#!/usr/bin/python3 -s
"""SIMULATION ONLY: fake wrist camera, fake Franka Hand and a physical red brick.

Run it next to MoveIt with use_fake_hardware:=true so lego_auto.py can be tested without the
robot. It
  * renders what the hand-mounted camera would see (from the live hand pose, with a deliberate
    calibration error) on the same topics as the RealSense driver: the brick, plus decoys the
    detector must ignore (a red object of the wrong size and a blue brick of the right size),
    on a lit, noisy background,
  * puts the brick at a RANDOM place and angle (sim_brick_random) that only this node knows:
    lego_auto.py gets the brick position from camera images alone,
  * serves franka_gripper/move and franka_gripper/grasp with real finger motion: the finger
    joint states are published, so the hand visibly opens and closes in RViz. A grasp only
    succeeds when the fingers really are around the brick, which then follows the hand,
  * publishes the true brick and the decoys as RViz markers on /sim_brick_marker.
It refuses to start if a real camera is already publishing.
"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from franka_msgs.action import Grasp, Move
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

K = np.array([[928.2, 0, 640.9], [0, 928.1, 368.3], [0, 0, 1.0]])   # the real D435 colour stream
SIZE = (1280, 720)
FINGER_JOINTS = ['fr3_finger_joint1', 'fr3_finger_joint2']
RED, BLUE, SHADOW = (30, 30, 200), (190, 70, 30), (150, 154, 152)    # BGR


def matrix(rotation, translation):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, translation
    return t


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


class Item:
    """A box lying on the table (centre xy, long-axis yaw, size L x W x H)."""

    def __init__(self, xy, yaw, size, colour, z_bottom):
        self.xy, self.yaw, self.size, self.colour, self.z_bottom = np.array(xy, float), yaw, size, colour, z_bottom

    def pose(self):
        return matrix(Rotation.from_euler('z', self.yaw).as_matrix(),
                      [*self.xy, self.z_bottom + self.size[2] / 2])


class SimWorld(Node):
    def __init__(self, c):
        super().__init__('sim_world')
        self.c = c
        cb = ReentrantCallbackGroup()
        calibration = json.load(open(c['calibration_file'], encoding='utf-8'))
        nominal = np.array(calibration['T_ee_camera'], dtype=float)
        err = c.get('sim_cal_error', {})
        wrong = matrix(Rotation.from_rotvec(np.radians(err.get('rotation_deg', [0, 0, 0]))).as_matrix(),
                       np.array(err.get('translation_mm', [0, 0, 0])) / 1000.0)
        self.t_ee_cam_true = wrong @ nominal       # what the simulated camera REALLY is
        self.size = np.array(c['block_size'], dtype=float)
        self.top = float(c['table_top_z'])
        self.place_scene()
        self.grasp = None                          # brick pose in the hand frame while held
        self.width, self.width_goal, self.width_speed = float(c['open_width']), float(c['open_width']), 0.05
        self.lock = threading.RLock()
        self.bridge = CvBridge()
        self.tf = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.tf, self, spin_thread=False)
        self.image_pub = self.create_publisher(Image, c['image_topic'], qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, c['camera_info_topic'], qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(Marker, '/sim_brick_marker', 10)
        self.finger_pub = self.create_publisher(JointState, '/franka_gripper/joint_states', 10)
        self.create_timer(0.1, self.publish_camera, callback_group=cb)
        self.create_timer(0.05, self.publish_markers, callback_group=cb)
        self.create_timer(0.02, self.step_fingers, callback_group=cb)
        ActionServer(self, Move, 'franka_gripper/move', self.on_move, callback_group=cb)
        ActionServer(self, Grasp, 'franka_gripper/grasp', self.on_grasp, callback_group=cb)
        self.get_logger().info(
            f'Simulated world ready. HIDDEN TRUTH: brick x={self.brick.xy[0]:.3f} y={self.brick.xy[1]:.3f} '
            f'yaw={math.degrees(self.brick.yaw):.0f} deg (the script must find this from the camera). '
            f'Camera error {err.get("translation_mm", [0, 0, 0])} mm / {err.get("rotation_deg", [0, 0, 0])} deg')

    # ---------------------------------------------------------------- scene
    def place_scene(self):
        c, rng = self.c, np.random.default_rng(self.c.get('sim_seed'))
        if c.get('sim_brick_random', False):
            xy = [rng.uniform(0.38, 0.52), rng.uniform(-0.10, 0.18)]
            yaw = math.radians(rng.uniform(-80, 80))
        else:
            xy, yaw = c['sim_brick_xy'], math.radians(c['sim_brick_yaw_deg'])
        self.brick = Item(xy, yaw, self.size, RED, self.top)
        self.decoys = []
        if c.get('sim_decoys', True):
            taken = [np.array(xy)]
            for size, colour in (((0.045, 0.045, 0.02), RED), (tuple(self.size), BLUE)):
                for _ in range(200):
                    cand = np.array([rng.uniform(0.36, 0.56), rng.uniform(-0.14, 0.22)])
                    if all(np.linalg.norm(cand - t) > 0.09 for t in taken):
                        taken.append(cand)
                        self.decoys.append(Item(cand, rng.uniform(-1.5, 1.5), size, colour, self.top))
                        break

    def hand_pose(self):
        t = self.tf.lookup_transform(self.c['base_frame'], self.c['ee_frame'], Time()).transform
        q, p = t.rotation, t.translation
        return matrix(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [p.x, p.y, p.z])

    def brick_pose(self):
        with self.lock:
            if self.grasp is None:
                return self.brick.pose()
        try:
            return self.hand_pose() @ self.grasp
        except Exception:
            return self.brick.pose()

    # --------------------------------------------------------------- camera
    def project(self, cam, xy, yaw, size, z):
        """Pixel polygon of a horizontal rectangle at height z, or None if behind the camera."""
        rot = Rotation.from_euler('z', yaw).as_matrix()[:2, :2]
        half = np.array([size[0] / 2, size[1] / 2])
        inverse = np.linalg.inv(cam)
        poly = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corner = np.asarray(xy) + rot @ (np.array([sx, sy]) * half)
            p = inverse @ np.array([corner[0], corner[1], z, 1.0])
            if p[2] < 0.05:
                return None
            uvw = K @ p[:3]
            poly.append(uvw[:2] / uvw[2])
        return np.round(poly).astype(np.int32)

    def publish_camera(self):
        stamp = self.get_clock().now().to_msg()
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, self.c['camera_frame']
        info.width, info.height = SIZE
        info.distortion_model, info.d = 'plumb_bob', [0.0] * 5
        info.k = K.flatten().tolist()
        info.p = [float(v) for v in (K[0, 0], 0, K[0, 2], 0, 0, K[1, 1], K[1, 2], 0, 0, 0, 1, 0)]
        info.r = np.eye(3).flatten().tolist()
        self.info_pub.publish(info)
        try:
            cam = self.hand_pose() @ self.t_ee_cam_true
        except Exception:
            return
        # a table lit unevenly: brighter toward one side
        gradient = np.tile(np.linspace(0.86, 1.04, SIZE[0]), (SIZE[1], 1))[:, :, None]
        image = np.clip(np.array((176, 180, 178), float) * gradient, 0, 255).astype(np.uint8)
        brick = self.brick_pose()
        shapes = []
        for item in self.decoys:
            shapes.append((item.xy, item.yaw, item.size, item.colour, item.z_bottom + item.size[2]))
        shapes.append((brick[:2, 3], math.atan2(brick[1, 0], brick[0, 0]), self.size, RED,
                       brick[2, 3] + self.size[2] / 2))
        for xy, yaw, size, colour, z in shapes:
            if z < self.top + 0.05:          # objects resting on the table cast a short shadow
                shadow = self.project(cam, np.asarray(xy) + [0.004, -0.004], yaw, size, self.top)
                if shadow is not None:
                    cv2.fillPoly(image, [shadow], SHADOW)
            poly = self.project(cam, xy, yaw, size, z)
            if poly is not None:
                cv2.fillPoly(image, [poly], colour)
        image = np.clip(image + np.random.normal(0, 2.0, image.shape), 0, 255).astype(np.uint8)
        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        msg.header.stamp, msg.header.frame_id = stamp, self.c['camera_frame']
        self.image_pub.publish(msg)

    def box_marker(self, ident, pose, size, colour):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = self.c['base_frame'], self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = 'sim_world', ident, Marker.CUBE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = pose[:3, 3]
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        marker.pose.orientation.x, marker.pose.orientation.y = float(q[0]), float(q[1])
        marker.pose.orientation.z, marker.pose.orientation.w = float(q[2]), float(q[3])
        marker.scale.x, marker.scale.y, marker.scale.z = [float(v) for v in size]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = colour
        return marker

    def publish_markers(self):
        self.marker_pub.publish(self.box_marker(0, self.brick_pose(), self.size, (0.9, 0.1, 0.1, 1.0)))
        for n, item in enumerate(self.decoys, 1):
            bgr = item.colour
            self.marker_pub.publish(self.box_marker(
                n, item.pose(), item.size, (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255, 1.0)))

    # -------------------------------------------------------------- gripper
    def step_fingers(self):
        with self.lock:
            error = self.width_goal - self.width
            step = self.width_speed * 0.02
            self.width += max(-step, min(step, error))
            half = self.width / 2
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = FINGER_JOINTS, [half, half]
        msg.velocity, msg.effort = [0.0, 0.0], [0.0, 0.0]
        self.finger_pub.publish(msg)

    def wait_for_fingers(self, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                if abs(self.width - self.width_goal) < 5e-4:
                    return True
            time.sleep(0.02)
        return False

    def on_move(self, goal_handle):
        goal = goal_handle.request
        with self.lock:
            if self.grasp is not None and goal.width > self.width + 0.002:
                held = self.brick_pose()
                self.brick.xy = held[:2, 3].copy()
                self.brick.yaw = math.atan2(held[1, 0], held[0, 0])
                self.grasp = None
                self.get_logger().info(f'RELEASE: brick dropped at x={held[0, 3]:.3f} y={held[1, 3]:.3f}, '
                                       f'{(held[2, 3] - self.size[2] / 2 - self.top) * 1000:.0f} mm above the table')
            self.width_goal = float(goal.width)
            self.width_speed = float(goal.speed) if goal.speed > 0 else 0.05
        self.wait_for_fingers()
        goal_handle.succeed()
        return Move.Result(success=True)

    def on_grasp(self, goal_handle):
        goal = goal_handle.request
        result = Grasp.Result()
        try:
            hand = self.hand_pose()
        except Exception as exc:
            goal_handle.abort()
            result.error = f'no hand pose: {exc}'
            return result
        with self.lock:
            brick = self.brick.pose()
            rel = np.linalg.inv(hand) @ brick
            offset = float(np.hypot(rel[0, 3], rel[1, 3]))
            jaw_axis = hand[:3, 1]                                    # fingers close along hand Y
            short_axis = brick[:3, :3] @ np.array([0.0, 1.0, 0.0])
            misalign = math.degrees(math.acos(min(1.0, abs(float(jaw_axis @ short_axis)))))
            low_enough = -0.03 < rel[2, 3] < 0.03                     # brick centre near the TCP height
            inside = goal.width + goal.epsilon.outer >= self.size[1] >= goal.width - goal.epsilon.inner
            ok = offset < 0.012 and misalign < 15.0 and low_enough and inside
            self.width_goal = float(self.size[1]) if ok else 0.0      # closes on the brick, or on nothing
            self.width_speed = float(goal.speed) if goal.speed > 0 else 0.05
        self.wait_for_fingers()
        if ok:
            with self.lock:
                self.grasp = rel
            self.get_logger().info(f'GRASP ok: brick centre {offset * 1000:.1f} mm off the TCP, '
                                   f'jaws {misalign:.1f} deg off the short side')
            goal_handle.succeed()
            result.success = True
            return result
        self.get_logger().warn(f'GRASP FAILED: offset {offset * 1000:.1f} mm, jaws {misalign:.0f} deg, '
                               f'height offset {rel[2, 3] * 1000:.0f} mm, width ok={inside}')
        goal_handle.abort()
        result.success, result.error = False, 'no object between the fingers'
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    args, ros_args = parser.parse_known_args()
    c = yaml.safe_load(open(args.config, encoding='utf-8'))
    rclpy.init(args=ros_args)
    probe = rclpy.create_node('sim_world_probe')
    for _ in range(20):
        rclpy.spin_once(probe, timeout_sec=0.1)       # let DDS discovery find any real camera
    if probe.get_publishers_info_by_topic(c['image_topic']):
        probe.destroy_node()
        rclpy.shutdown()
        sys.exit('A camera is already publishing on ' + c['image_topic'] +
                 ': sim_world is for simulation only and will not start next to a real camera.')
    probe.destroy_node()
    silence_fake_gripper()
    node = SimWorld(c)
    executor = MultiThreadedExecutor(num_threads=6)
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
