#!/usr/bin/python3 -s
"""Add the table and a red Lego brick to the MoveIt planning scene (visible in RViz).

Sizes and the table height come from lego.yaml. This only edits the planning scene;
it never moves the robot. The table uses the same id as pick_place.py ('fr3_demo_table'),
so the planner's own copy simply replaces it. The brick is 'lego_brick'; lego_pick_place.py
reads it with --brick-from-scene and removes it before planning the grasp.
"""
import argparse
import math
import sys
import time

import rclpy
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA

TABLE_ID = 'fr3_demo_table'
BRICK_ID = 'lego_brick'
BASE_PAD_ID = 'fr3_base_pad'


def make_box(name, size, xyz, yaw=0.0, frame='fr3_link0'):
    obj = CollisionObject()
    obj.id, obj.operation = name, CollisionObject.ADD
    obj.header.frame_id = frame
    obj.pose.orientation.w = 1.0
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
    pose.orientation.z, pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(s) for s in size])]
    obj.primitive_poses = [pose]
    return obj


def colour(name, r, g, b, a=1.0):
    return ObjectColor(id=name, color=ColorRGBA(r=r, g=g, b=b, a=a))


def apply_scene(node, add=(), remove=(), colours=(), strict=True):
    client = node.create_client(ApplyPlanningScene, 'apply_planning_scene')
    end = time.monotonic() + 90
    while not client.wait_for_service(timeout_sec=1.0):
        if time.monotonic() > end:
            raise RuntimeError('MoveIt apply_planning_scene service not available (is MoveIt running?)')
    scene = PlanningScene(is_diff=True)
    scene.robot_state.is_diff = True
    scene.world.collision_objects = list(add)
    for name in remove:
        scene.world.collision_objects.append(CollisionObject(id=name, operation=CollisionObject.REMOVE))
    scene.object_colors = list(colours)
    # A fresh client's first request is sometimes never answered (DDS discovery race). The update is a
    # diff (ADD replaces, REMOVE of a missing object is harmless), so asking again is safe.
    for attempt in range(3):
        future = client.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        if future.done():
            break
        client.remove_pending_request(future)
    else:
        raise RuntimeError('MoveIt did not answer the planning-scene update (3 tries of 5 s)')
    if not (future.result().success or not strict):
        raise RuntimeError('MoveIt rejected the planning-scene update')


def table_object(c):
    return make_box(TABLE_ID, c['table_size'], c['table_center'])


def base_pad_objects(c):
    """The spacer/bracket between the robot base and the table (empty if the config has none).
    Same length and width as the base footprint, from the config's base_pad_size / base_pad_center."""
    if 'base_pad_size' not in c:
        return []
    return [make_box(BASE_PAD_ID, c['base_pad_size'], c['base_pad_center'])]


def brick_object(c, x, y, yaw):
    height = c['block_size'][2]
    return make_box(BRICK_ID, c['block_size'], (x, y, c['table_top_z'] + height / 2), yaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--no-brick', action='store_true', help='Spawn only the table')
    parser.add_argument('--brick-xy', nargs=2, type=float, metavar=('X', 'Y'),
                        help='Brick centre in fr3_link0 (default: sim_brick_xy in the config)')
    parser.add_argument('--brick-yaw', type=float, metavar='DEG',
                        help='Long-axis angle in degrees (default: sim_brick_yaw_deg)')
    parser.add_argument('--remove', action='store_true', help='Remove the table and brick instead')
    args, ros_args = parser.parse_known_args()
    c = yaml.safe_load(open(args.config, encoding='utf-8'))
    rclpy.init(args=ros_args)
    node = rclpy.create_node('spawn_lego_scene')
    try:
        if args.remove:
            apply_scene(node, remove=(TABLE_ID, BRICK_ID, BASE_PAD_ID))
            print('Removed table, base pad and brick from the planning scene.')
            return
        add, colours = [table_object(c)], [colour(TABLE_ID, 0.55, 0.55, 0.6, 0.9)]
        text = f'table top at z={c["table_top_z"]:.3f} m'
        pad = base_pad_objects(c)
        if pad:
            add += pad
            colours.append(colour(BASE_PAD_ID, 0.95, 0.65, 0.1, 0.9))
            text += (f', base pad {c["base_pad_size"][0] * 100:.1f} x {c["base_pad_size"][1] * 100:.1f} x '
                     f'{c["base_pad_size"][2] * 100:.1f} cm under the robot base')
        if not args.no_brick:
            x, y = args.brick_xy or c['sim_brick_xy']
            yaw = math.radians(args.brick_yaw if args.brick_yaw is not None else c['sim_brick_yaw_deg'])
            add.append(brick_object(c, x, y, yaw))
            colours.append(colour(BRICK_ID, 0.85, 0.1, 0.1))
            text += f', brick at x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.0f} deg'
        apply_scene(node, add=add, colours=colours)
        print(f'Spawned in the planning scene: {text}.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
