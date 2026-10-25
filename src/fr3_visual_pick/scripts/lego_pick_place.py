#!/usr/bin/python3 -s
"""Find a Lego brick with the wrist D435 and pick-and-place it with MoveIt.

Modes (increasing risk):
  --detect-only   camera + arm pose only; no MoveIt, no motion. Writes a debug image.
  (default)       detect, then PLAN all six motions. No arm or hand command is sent.
  --execute       detect, plan and run the cycle on the real arm and hand.

No camera or calibration is needed for simulation: --brick-from-scene uses the virtual
'lego_brick' that spawn_scene.py (or the launch file) put in the MoveIt scene, and
--brick-xy X Y [--brick-yaw DEG] takes the pose from the command line. Both are plan-only.

The brick position comes from projecting its image pixels onto the table plane
using the arm's current pose and the hand-eye result, so no depth stream is needed.
Hold the arm still, high enough that the camera sees the brick, before running.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))


def hsv_mask(bgr, ranges):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_bricks(bgr, k, t_base_cam, c):
    """Return (accepted, rejected) brick dicts in the base frame, best first."""
    plane_z = c['table_top_z'] + c['block_size'][2]       # top face of the brick
    kinv = np.linalg.inv(k)
    origin, rot = t_base_cam[:3, 3], t_base_cam[:3, :3]

    def on_plane(u, v):
        d = rot @ (kinv @ np.array([u, v, 1.0]))
        if d[2] > -1e-6:
            return None
        return origin + (plane_z - origin[2]) / d[2] * d

    contours, _ = cv2.findContours(hsv_mask(bgr, c['hsv_ranges']), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    accepted, rejected = [], []
    length, width = c['block_size'][0], c['block_size'][1]
    tol, (xlim, ylim) = c['size_tolerance'], (c['workspace_x'], c['workspace_y'])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < c['min_area_px']:
            continue
        rect = cv2.minAreaRect(contour)
        pixels = cv2.boxPoints(rect)
        corners = [on_plane(u, v) for u, v in pixels]
        item = {'pixel_rect': pixels, 'area_px': float(area)}
        if any(p is None for p in corners):
            item['why'] = 'ray does not hit the table'
            rejected.append(item)
            continue
        corners = np.array(corners)
        e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
        if np.linalg.norm(e1) < np.linalg.norm(e2):
            e1, e2 = e2, e1
        long_side, short_side = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        centre = corners.mean(axis=0)
        yaw = math.atan2(e1[1], e1[0])
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2  # long axis is a line, not an arrow
        item.update(centre=centre, yaw=yaw, size=(long_side, short_side))
        if abs(long_side - length) > tol * length or abs(short_side - width) > tol * width:
            item['why'] = f'size {long_side * 1000:.0f}x{short_side * 1000:.0f} mm is not a brick'
        elif not (xlim[0] <= centre[0] <= xlim[1] and ylim[0] <= centre[1] <= ylim[1]):
            item['why'] = 'outside the allowed workspace'
        (rejected if 'why' in item else accepted).append(item)
    accepted.sort(key=lambda i: -i['area_px'])
    return accepted, rejected


def annotate(bgr, accepted, rejected):
    out = bgr.copy()
    for items, colour in ((rejected, (0, 0, 255)), (accepted, (0, 255, 0))):
        for n, item in enumerate(items):
            box = item['pixel_rect'].astype(int)
            cv2.polylines(out, [box], True, colour, 2)
            label = (f'#{n} ' + (f'{item["centre"][0]:.3f},{item["centre"][1]:.3f}'
                                 if colour[1] else item.get('why', '')))
            cv2.putText(out, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return out


def circular_mean_line(angles):
    """Mean of axis angles that repeat every pi."""
    return 0.5 * math.atan2(np.mean(np.sin(2 * np.asarray(angles))),
                            np.mean(np.cos(2 * np.asarray(angles))))


class Eye:
    def __init__(self, node, c, calibration):
        from sensor_msgs.msg import CameraInfo, Image
        from cv_bridge import CvBridge
        from rclpy.duration import Duration
        from rclpy.qos import qos_profile_sensor_data
        from tf2_ros import Buffer, TransformListener
        self.node, self.c, self.bridge = node, c, CvBridge()
        self.image = self.info = None
        self.buffer = Buffer(cache_time=Duration(seconds=30))
        self.listener = TransformListener(self.buffer, node)
        node.create_subscription(Image, c['image_topic'], self._image, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, c['camera_info_topic'], self._info, qos_profile_sensor_data)
        self.t_ee_cam = np.array(calibration['T_ee_camera'], dtype=float)

    def _image(self, msg):
        self.image = msg

    def _info(self, msg):
        self.info = msg

    def spin(self, seconds):
        import rclpy
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def look(self):
        """One (image, K, T_base_cam) triple from the current, stationary arm pose."""
        from rclpy.time import Time
        self.image = None
        self.spin(0.5)
        for _ in range(40):
            if self.image is not None and self.info is not None:
                break
            self.spin(0.1)
        else:
            raise RuntimeError('No camera image/CameraInfo: is the RealSense node running?')
        msg, info = self.image, self.info
        if msg.header.frame_id != self.c['camera_frame']:
            raise RuntimeError(f'Image frame {msg.header.frame_id} != {self.c["camera_frame"]}')
        try:
            tf = self.buffer.lookup_transform(self.c['base_frame'], self.c['ee_frame'],
                                              Time.from_msg(msg.header.stamp))
        except Exception:  # arm is stationary: the newest pose is equivalent
            tf = self.buffer.lookup_transform(self.c['base_frame'], self.c['ee_frame'], Time())
        q, t = tf.transform.rotation, tf.transform.translation
        t_base_ee = np.eye(4)
        t_base_ee[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t_base_ee[:3, 3] = [t.x, t.y, t.z]
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return bgr, np.array(info.k, dtype=float).reshape(3, 3), t_base_ee @ self.t_ee_cam

    def latest_frame(self, last_stamp):
        """The newest camera frame not yet seen, as (stamp, image, K, T_base_cam), else None."""
        from rclpy.time import Time
        msg, info = self.image, self.info
        if msg is None or info is None or msg.header.stamp == last_stamp:
            return None
        if msg.header.frame_id != self.c['camera_frame']:
            raise RuntimeError(f'Image frame {msg.header.frame_id} != {self.c["camera_frame"]}')
        try:
            tf = self.buffer.lookup_transform(self.c['base_frame'], self.c['ee_frame'],
                                              Time.from_msg(msg.header.stamp))
        except Exception:
            return None                      # no arm pose for this exact frame: skip it
        q, t = tf.transform.rotation, tf.transform.translation
        t_base_ee = np.eye(4)
        t_base_ee[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t_base_ee[:3, 3] = [t.x, t.y, t.z]
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return (msg.header.stamp, bgr, np.array(info.k, dtype=float).reshape(3, 3),
                t_base_ee @ self.t_ee_cam)

    def find_brick(self, frames=5):
        """Average several frames; return the chosen brick, the debug image and a summary."""
        runs = []
        for _ in range(frames):
            bgr, k, t_base_cam = self.look()
            accepted, rejected = detect_bricks(bgr, k, t_base_cam, self.c)
            runs.append((bgr, accepted, rejected))
            self.spin(0.2)
        bgr, accepted, rejected = runs[-1]
        debug = annotate(bgr, accepted, rejected)
        if not accepted:
            reasons = '; '.join(sorted({i.get('why', '?') for i in rejected})) or 'nothing red found'
            return None, debug, f'No brick accepted ({reasons})'
        target = accepted[0]['centre']
        matches = []
        for _, acc, _ in runs:
            near = [i for i in acc if np.linalg.norm(i['centre'][:2] - target[:2]) < 0.02]
            if near:
                matches.append(near[0])
        if len(matches) < max(3, frames - 1):
            return None, debug, f'Brick not stable across frames ({len(matches)}/{frames})'
        centres = np.array([m['centre'] for m in matches])
        spread = float(np.max(np.linalg.norm(centres[:, :2] - centres[:, :2].mean(axis=0), axis=1)))
        if spread > 0.004:
            return None, debug, f'Brick position jitters {spread * 1000:.1f} mm between frames'
        yaw = circular_mean_line([m['yaw'] for m in matches])
        return {'centre': centres.mean(axis=0), 'yaw': yaw,
                'size': np.mean([m['size'] for m in matches], axis=0)}, debug, 'ok'


def brick_from_scene(node):
    """Read the virtual brick from the MoveIt planning scene; return (brick, its message)."""
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene
    import rclpy
    from spawn_scene import BRICK_ID
    client = node.create_client(GetPlanningScene, 'get_planning_scene')
    if not client.wait_for_service(timeout_sec=15):
        raise RuntimeError('get_planning_scene unavailable: is MoveIt running?')
    request = GetPlanningScene.Request()
    request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10)
    if not future.done():
        raise RuntimeError('Timed out reading the planning scene')
    objects = {o.id: o for o in future.result().scene.world.collision_objects}
    if BRICK_ID not in objects or not objects[BRICK_ID].primitive_poses:
        raise RuntimeError(f"No '{BRICK_ID}' in the scene. Run: ros2 run fr3_visual_pick spawn_scene.py "
                           f"--config <lego.yaml>")
    obj = objects[BRICK_ID]
    if obj.header.frame_id not in ('', 'fr3_link0', 'base', 'world'):  # 'base' is MoveIt's root
        raise RuntimeError(f'Brick is in frame {obj.header.frame_id}; expected fr3_link0')
    base, local = obj.pose, obj.primitive_poses[0]
    base_r = Rotation.from_quat([base.orientation.x, base.orientation.y, base.orientation.z,
                                 base.orientation.w] if any(
        (base.orientation.x, base.orientation.y, base.orientation.z, base.orientation.w)) else [0, 0, 0, 1])
    local_r = Rotation.from_quat([local.orientation.x, local.orientation.y, local.orientation.z,
                                  local.orientation.w])
    centre = (base_r.apply([local.position.x, local.position.y, local.position.z])
              + [base.position.x, base.position.y, base.position.z])
    yaw = (base_r * local_r).as_euler('xyz')[2]
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return {'centre': centre, 'yaw': float(yaw), 'size': tuple(obj.primitives[0].dimensions[:2])}, obj


def build_task_config(c, brick, place_xy):
    top, height = c['table_top_z'], c['block_size'][2]
    grip_z = top + c['grasp_tcp_height']
    cfg = {k: c[k] for k in ('block_size', 'table_center', 'table_size', 'clearance',
                             'velocity_scale', 'acceleration_scale', 'grasp_force',
                             'grasp_width', 'gripper_speed', 'open_width')}
    cfg['pick'] = [float(brick['centre'][0]), float(brick['centre'][1]), grip_z]
    cfg['pick_block_center'] = [cfg['pick'][0], cfg['pick'][1], top + height / 2]
    cfg['pick_yaw'] = float(brick['yaw'])
    cfg['place'] = [place_xy[0], place_xy[1], grip_z]
    cfg['place_block_center'] = [place_xy[0], place_xy[1], top + height / 2]
    cfg['place_yaw'] = float(c.get('place_yaw', 0.0))
    cfg['block_in_tcp'] = [0.0, 0.0, grip_z - (top + height / 2)]  # tool Z points down
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--debug-image', default='lego_detection.png')
    parser.add_argument('--detect-only', action='store_true')
    parser.add_argument('--place-xy', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--execute', action='store_true', help='Move the real arm and hand')
    parser.add_argument('--brick-from-scene', action='store_true',
                        help="Simulation: use the virtual 'lego_brick' from the MoveIt scene (plan only)")
    parser.add_argument('--brick-xy', nargs=2, type=float, metavar=('X', 'Y'),
                        help='Simulation: brick centre in fr3_link0, skips the camera (plan only)')
    parser.add_argument('--brick-yaw', type=float, default=0.0, metavar='DEG',
                        help='Simulation: brick long-axis angle with --brick-xy')
    args, ros_args = parser.parse_known_args()
    c = yaml.safe_load(open(args.config, encoding='utf-8'))
    simulated = args.brick_from_scene or args.brick_xy is not None
    if simulated and args.execute:
        raise ValueError('Simulated bricks are plan-only: they cannot be combined with --execute')
    calibration = None
    if not simulated:
        calibration = json.load(open(c['calibration_file'], encoding='utf-8'))
        if calibration['child_frame'] != c['camera_frame'] or calibration['parent_frame'] != c['ee_frame']:
            raise ValueError('Calibration frames do not match the config')
        if calibration['status'] != 'passed_quality_gates':
            if not c.get('allow_rejected_calibration', False):
                raise ValueError('Calibration did not pass its quality gates; see allow_rejected_calibration')
            print(f'WARNING: using a calibration with status {calibration["status"]} '
                  f'(typical error {calibration["all_samples"]["translation_rms_m"] * 1000:.1f} mm, '
                  f'{calibration["all_samples"]["rotation_rms_deg"]:.1f} deg).')
    place_xy = args.place_xy or c.get('place_xy')
    if not args.detect_only and not place_xy:
        raise ValueError('Give the drop position with --place-xy X Y (metres, in fr3_link0)')
    import rclpy
    rclpy.init(args=ros_args)
    node = rclpy.create_node('lego_pick_place')
    scene_brick = None
    try:
        if args.brick_from_scene:
            brick, scene_brick = brick_from_scene(node)
            print('Using the virtual brick from the MoveIt scene.')
        elif args.brick_xy is not None:
            top = c['table_top_z'] + c['block_size'][2] / 2
            yaw = (math.radians(args.brick_yaw) + math.pi / 2) % math.pi - math.pi / 2
            brick = {'centre': np.array([*args.brick_xy, top]), 'yaw': yaw,
                     'size': tuple(c['block_size'][:2])}
            print('Using the brick pose from the command line.')
        else:
            eye = Eye(node, c, calibration)
            brick, debug, status = eye.find_brick()
            cv2.imwrite(args.debug_image, debug)
            print(f'Debug image: {Path(args.debug_image).resolve()}')
            if brick is None:
                raise SystemExit(f'Detection failed: {status}')
        print(f'Brick at x={brick["centre"][0]:.4f} y={brick["centre"][1]:.4f} m, '
              f'yaw={math.degrees(brick["yaw"]):.1f} deg, '
              f'size {brick["size"][0] * 1000:.0f}x{brick["size"][1] * 1000:.0f} mm')
        if args.detect_only:
            return
        from pick_place import Task, validate
        cfg = validate(build_task_config(c, brick, place_xy))
        print(f'Pick TCP {np.round(cfg["pick"], 4).tolist()}  place TCP {np.round(cfg["place"], 4).tolist()}')
        if args.execute:
            print('This will MOVE THE REAL ARM AND HAND. Check the debug image, clear the workspace, '
                  'hold the e-stop.')
            if input('Type GO to start: ').strip() != 'GO':
                raise SystemExit('Cancelled; nothing was moved.')
        if scene_brick is not None:
            # The planner adds its own brick; the virtual one would collide with the fingers.
            from spawn_scene import BRICK_ID, apply_scene
            apply_scene(node, remove=(BRICK_ID,))
        Task(node, cfg, args.execute).run()
        print('Cycle complete' if args.execute else 'All six motions planned; no motion commanded')
    finally:
        if scene_brick is not None:
            from spawn_scene import apply_scene
            scene_brick.operation = scene_brick.ADD
            apply_scene(node, add=[scene_brick])  # put the virtual brick back for the next run
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
