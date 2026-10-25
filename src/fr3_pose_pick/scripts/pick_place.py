#!/usr/bin/env python3
"""One measured-position cycle. Planning only unless --execute is supplied."""
import argparse
import copy
import math

import yaml
import rclpy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import (
    AllowedCollisionEntry, AttachedCollisionObject, CollisionObject, Constraints,
    MoveItErrorCodes, OrientationConstraint, PlanningSceneComponents, PositionConstraint,
)
from franka_msgs.action import Move, Grasp


def validate(c):
    if not isinstance(c, dict):
        raise ValueError('Configuration must be a YAML mapping')
    for key in ('pick', 'place', 'block_size', 'table_center', 'table_size'):
        v = c.get(key)
        if not isinstance(v, list) or len(v) != 3 or any(
                isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
                for x in v):
            raise ValueError(f'{key} must be three finite measured values in metres')
    for key in ('block_size', 'table_size'):
        if min(c[key]) <= 0:
            raise ValueError(f'{key} dimensions must be positive')
    limits = {'clearance': (0, 0.3), 'velocity_scale': (0, 0.1),
              'acceleration_scale': (0, 0.1), 'grasp_force': (0, 20),
              'gripper_speed': (0, 0.05), 'open_width': (0, 0.08),
              'grasp_width': (0, 0.08)}
    for key, (low, high) in limits.items():
        v = c.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not low < v <= high:
            raise ValueError(f'{key} must be in ({low}, {high}]')
    needed = c.get('min_open_clearance', 0.005)
    if c['block_size'][1] + needed > c['open_width'] + 1e-9:
        raise ValueError(f'The open hand ({c["open_width"] * 1000:.1f} mm) needs at least {needed * 1000:.1f} mm '
                         f'more than the brick short side ({c["block_size"][1] * 1000:.1f} mm) '
                         '(min_open_clearance in the config)')
    if c['grasp_width'] >= c['open_width']:
        raise ValueError('grasp_width must be smaller than open_width')
    top = c['table_center'][2] + c['table_size'][2] / 2
    for key in ('pick', 'place'):
        if c[key][2] - c['block_size'][2] / 2 < top - 0.001:
            raise ValueError(f'{key}: block would intersect the table')
    return c


def box(name, size, xyz, frame='fr3_link0'):
    obj = CollisionObject()
    obj.id = name
    obj.header.frame_id = frame
    obj.operation = CollisionObject.ADD
    primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(x) for x in size])
    pose = Pose()
    pose.orientation.w = 1.0
    pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
    obj.primitives = [primitive]
    obj.primitive_poses = [pose]
    return obj


class Task:
    def __init__(self, node, config, execute):
        self.node, self.c, self.execute = node, config, execute
        self.joints = {}
        self.arm = ActionClient(node, MoveGroup, 'move_action')
        if not self.arm.wait_for_server(timeout_sec=15):
            raise RuntimeError('MoveIt move_action unavailable; launch MoveIt first')
        self.scene_client = node.create_client(GetPlanningScene, 'get_planning_scene')
        if not self.scene_client.wait_for_service(timeout_sec=15):
            raise RuntimeError('get_planning_scene unavailable')
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        self.acm = self.wait(self.scene_client.call_async(req), 10).scene.allowed_collision_matrix
        # Permit contact with the block only for the two grasping fingers.
        name = 'fr3_demo_block'
        if name not in self.acm.entry_names:
            for row in self.acm.entry_values:
                row.enabled.append(False)
            self.acm.entry_names.append(name)
            self.acm.entry_values.append(AllowedCollisionEntry(
                enabled=[False] * len(self.acm.entry_names)))
        index = self.acm.entry_names.index(name)
        for finger in ('fr3_leftfinger', 'fr3_rightfinger'):
            if finger not in self.acm.entry_names:
                for row in self.acm.entry_values:
                    row.enabled.append(False)
                self.acm.entry_names.append(finger)
                self.acm.entry_values.append(AllowedCollisionEntry(
                    enabled=[False] * len(self.acm.entry_names)))
            j = self.acm.entry_names.index(finger)
            self.acm.entry_values[index].enabled[j] = True
            self.acm.entry_values[j].enabled[index] = True
        self.move = self.grasp = None
        if execute:
            self.move = ActionClient(node, Move, 'franka_gripper/move')
            self.grasp = ActionClient(node, Grasp, 'franka_gripper/grasp')
            if not all(client.wait_for_server(timeout_sec=10) for client in (self.move, self.grasp)):
                raise RuntimeError('Franka Hand action servers unavailable')

    def wait(self, future, timeout):
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError('ROS operation timed out; inspect robot state before retrying')
        return future.result()

    def action(self, client, goal):
        handle = self.wait(client.send_goal_async(goal), 10)
        if not handle.accepted:
            raise RuntimeError('Action rejected')
        try:
            result = self.wait(handle.get_result_async(), 120)
        except (TimeoutError, KeyboardInterrupt):
            self.wait(handle.cancel_goal_async(), 5)
            raise
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            detail = getattr(result.result, 'error_code', None)
            code = f', MoveIt code {detail.val}' if detail is not None else ''
            raise RuntimeError(f'Action ended with status {result.status}{code}')
        return result.result

    def hand(self, close=False):
        width = self.c['grasp_width'] if close else self.c['open_width']
        if self.execute:
            goal = Grasp.Goal() if close else Move.Goal()
            goal.width, goal.speed = float(width), float(self.c['gripper_speed'])
            if close:
                goal.force = float(self.c['grasp_force'])
                goal.epsilon.inner = goal.epsilon.outer = 0.003
            result = self.action(self.grasp if close else self.move, goal)
            if not result.success:
                raise RuntimeError(f'Gripper failed: {result.error}')
        else:
            for name in ('fr3_finger_joint1', 'fr3_finger_joint2'):
                self.joints[name] = width / 2

    def motion(self, label, xyz, carrying=False, placed=False, yaw=0.0, with_block=True):
        self.node.get_logger().info(f'{label}: {xyz} (execute={self.execute})')
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = 'fr3_arm'
        req.allowed_planning_time = 10.0
        req.num_planning_attempts = 5
        req.max_velocity_scaling_factor = float(self.c['velocity_scale'])
        req.max_acceleration_scaling_factor = float(self.c['acceleration_scale'])
        req.start_state.is_diff = True
        if not self.execute:
            req.start_state.joint_state.name = list(self.joints)
            req.start_state.joint_state.position = list(self.joints.values())
        position = PositionConstraint()
        position.header.frame_id = 'fr3_link0'
        position.link_name = 'fr3_hand_tcp'
        region = box('goal', [0.002] * 3, xyz)
        position.constraint_region.primitives = region.primitives
        position.constraint_region.primitive_poses = region.primitive_poses
        position.weight = 1.0
        orientation = OrientationConstraint()
        orientation.header.frame_id = 'fr3_link0'
        orientation.link_name = 'fr3_hand_tcp'
        # Tool Z points down; at yaw 0 the jaws travel along base Y. Quaternion of Rz(yaw) * Rx(pi).
        orientation.orientation.x = math.cos(yaw / 2)
        orientation.orientation.y = math.sin(yaw / 2)
        orientation.orientation.w = 0.0
        orientation.absolute_x_axis_tolerance = 0.02
        orientation.absolute_y_axis_tolerance = 0.02
        orientation.absolute_z_axis_tolerance = 0.02
        orientation.weight = 1.0
        req.goal_constraints = [Constraints(
            position_constraints=[position], orientation_constraints=[orientation])]
        scene = goal.planning_options.planning_scene_diff
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = copy.deepcopy(self.acm)
        scene.world.collision_objects = [box(
            'fr3_demo_table', self.c['table_size'], self.c['table_center'])]
        if carrying:
            attached = AttachedCollisionObject()
            attached.link_name = 'fr3_hand_tcp'
            attached.touch_links = ['fr3_leftfinger', 'fr3_rightfinger']
            attached.object = box('fr3_demo_block', self.c['block_size'],
                                  self.c.get('block_in_tcp', [0, 0, 0]), 'fr3_hand_tcp')
            req.start_state.attached_collision_objects = [attached]
            scene.robot_state.attached_collision_objects = [copy.deepcopy(attached)]
        elif with_block:
            centre = self.c.get('place_block_center' if placed else 'pick_block_center',
                                self.c['place' if placed else 'pick'])
            scene.world.collision_objects.append(box('fr3_demo_block', self.c['block_size'], centre))
        goal.planning_options.plan_only = not self.execute
        goal.planning_options.replan = False
        result = self.action(self.arm, goal)
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'{label} failed: MoveIt code {result.error_code.val}')
        trajectory = result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            raise RuntimeError(f'{label}: empty trajectory')
        if not self.execute:
            self.joints.update(zip(trajectory.joint_names, trajectory.points[-1].positions))

    def run(self):
        pick, place = self.c['pick'], self.c['place']
        above_pick, above_place = list(pick), list(place)
        above_pick[2] += self.c['clearance']
        above_place[2] += self.c['clearance']
        pick_yaw, place_yaw = self.c.get('pick_yaw', 0.0), self.c.get('place_yaw', 0.0)
        self.hand()
        self.motion('Approach pickup', above_pick, yaw=pick_yaw)
        self.motion('Lower to block', pick, yaw=pick_yaw)
        self.hand(close=True)
        self.motion('Lift block', above_pick, carrying=True, yaw=pick_yaw)
        self.motion('Transfer block', above_place, carrying=True, yaw=place_yaw)
        self.motion('Lower to destination', place, carrying=True, yaw=place_yaw)
        self.hand()
        self.motion('Retreat', above_place, placed=True, yaw=place_yaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execute', action='store_true', help='Enable physical arm AND hand actions')
    args, ros_args = parser.parse_known_args()
    with open(args.config, encoding='utf-8') as stream:
        config = validate(yaml.safe_load(stream))
    rclpy.init(args=ros_args)
    node = rclpy.create_node('fr3_pick_place')
    try:
        Task(node, config, args.execute).run()
        node.get_logger().info('Cycle complete' if args.execute else 'All six motions planned; no motion commanded')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
