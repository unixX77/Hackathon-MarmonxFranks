#!/usr/bin/python3 -s
"""Brick pixel + depth -> pose in the robot base frame (fr3_link0), using the hand-eye result.

    p_base = T_base_ee(forward kinematics) @ T_ee_camera(hand-eye file) @ p_camera
    p_camera = ((u - cx) * Z / fx, (v - cy) * Z / fy, Z)      Z = depth along the optical axis

The full 4x4 chain is used, so a camera that is tilted (not looking straight down) is fine.
The clicked point is on the brick's TOP face, so the brick centre is half a brick height lower.
Sanity check: for a brick lying on the table the top face must come out at
table_top_z + brick height; the difference is reported (a big one means a calibration, intrinsics,
depth or arm-moved problem). The arm must stand still from the picture until the answer is read.

LIVE (RealSense running with aligned depth: align_depth.enable:=true):
  pixel_to_base.py --config pose_pick.yaml
      A window shows the camera. SPACE freezes the picture, click the brick CENTRE, optionally click
      the END OF THE LONG SIDE (gives the yaw), Enter accepts, r = unfreeze, q = quit.
  pixel_to_base.py --config pose_pick.yaml --pixel 640 360 [--yaw DEG]     (no window)

OFFLINE (no ROS, no camera: numbers someone gave you; hand pose from  ros2 run tf2_ros tf2_echo
fr3_link0 fr3_hand_tcp ):
  pixel_to_base.py --config pose_pick.yaml --pixel 640 360 --depth 0.42 \\
      --intrinsics 910 910 640 360 --ee-pose 0.4 0 0.45 1 0 0 0

The answer ends with the matching pose_pick.py command.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_pick import wrap_line_angle                         # noqa: E402


# ------------------------------------------------------------------ pure logic (no ROS)
def deproject(u, v, depth, fx, fy, cx, cy):
    """Pixel + depth along the optical axis -> 3D point in the camera frame (metres)."""
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=float)


def project(point, fx, fy, cx, cy):
    """Inverse of deproject: camera-frame point -> (u, v, depth)."""
    return fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy, point[2]


def transform_point(t_matrix, point):
    return (np.asarray(t_matrix, dtype=float) @ np.append(point, 1.0))[:3]


def pose_matrix(xyz, quat_xyzw):
    from scipy.spatial.transform import Rotation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    m[:3, 3] = xyz
    return m


def robust_depth(depth_m, u, v, half=2, min_valid=6):
    """Median depth (m) in a (2*half+1)^2 window around the pixel, ignoring invalid pixels.
    Returns (depth, spread) where spread is the 90th-percentile deviation from the median
    (one flying pixel does not count; a click on an edge does)."""
    h, w = depth_m.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f'Pixel ({u}, {v}) is outside the {w}x{h} image')
    window = depth_m[max(v - half, 0):v + half + 1, max(u - half, 0):u + half + 1]
    valid = window[np.isfinite(window) & (window > 0.05)]
    if valid.size < min_valid:
        raise ValueError(f'Only {valid.size} valid depth pixels around ({u}, {v}): the depth is missing '
                         'there (shiny, too close/far, or an edge). Click a flatter part of the brick.')
    median = float(np.median(valid))
    return median, float(np.percentile(np.abs(valid - median), 90))


def brick_pose_from_pixels(t_base_cam, intrinsics, centre_px, depth, block_height, long_end_px=None,
                           long_end_depth=None):
    """Top-face point, brick centre and (optional) long-axis yaw in the base frame."""
    fx, fy, cx, cy = intrinsics
    top = transform_point(t_base_cam, deproject(centre_px[0], centre_px[1], depth, fx, fy, cx, cy))
    result = {'top': top, 'centre': top - np.array([0.0, 0.0, block_height / 2]), 'yaw': None, 'yaw_from': None}
    if long_end_px is not None:
        end = transform_point(t_base_cam, deproject(long_end_px[0], long_end_px[1],
                                                    long_end_depth or depth, fx, fy, cx, cy))
        dx, dy = end[0] - top[0], end[1] - top[1]
        if math.hypot(dx, dy) < 0.008:
            raise ValueError('The two clicks are less than 8 mm apart: click the END of the long side')
        result['yaw'], result['yaw_from'] = wrap_line_angle(math.atan2(dy, dx)), 'your 2nd click'
    return result


def table_check(c, top_z):
    """Signed difference (m) between the measured top-face z and where a brick on the table has it."""
    return top_z - (c['table_top_z'] + c['block_size'][2])


def load_calibration(path, c, camera_frame='camera_color_optical_frame'):
    calibration = json.load(open(path, encoding='utf-8'))
    if calibration['child_frame'] != camera_frame or calibration['parent_frame'] != c['ee_frame']:
        raise ValueError(f'Calibration frames {calibration["parent_frame"]} -> {calibration["child_frame"]} '
                         f'do not match {c["ee_frame"]} -> {camera_frame}')
    if calibration['status'] != 'passed_quality_gates' and not c.get('allow_rejected_calibration', False):
        raise ValueError('Calibration did not pass its quality gates; see allow_rejected_calibration')
    return calibration, np.array(calibration['T_ee_camera'], dtype=float)


def report(c, config_path, result, depth, spread, tolerance):
    centre, top = result['centre'], result['top']
    err = table_check(c, top[2])
    lines = [f'Depth {depth * 1000:.0f} mm (window spread {spread * 1000:.1f} mm)',
             f'Top face in base frame : x={top[0]:.4f} y={top[1]:.4f} z={top[2]:.4f}',
             f'Brick centre           : x={centre[0]:.4f} y={centre[1]:.4f} z={centre[2]:.4f}']
    lines.append('Yaw (long side)        : ' + (f'{math.degrees(result["yaw"]):.1f} deg (from {result["yaw_from"]})'
                                                 if result['yaw'] is not None else 'not measured, pass --yaw'))
    verdict = 'OK' if abs(err) <= tolerance else 'CHECK THIS'
    lines.append(f'Table check            : top face is {err * 1000:+.1f} mm from where a brick on the table '
                 f'should be ({c["table_top_z"] + c["block_size"][2]:.4f}) -> {verdict}')
    if abs(err) > tolerance:
        lines.append(f'  More than {tolerance * 1000:.0f} mm off: suspect the calibration, the camera intrinsics, '
                     'the depth (try another pixel), or the arm moved after the picture.')
    yaw_arg = (f' --yaw {math.degrees(result["yaw"]):.1f}' if result['yaw'] is not None else ' --yaw 0')
    base = f'ros2 run fr3_pose_pick pose_pick.py --config {config_path} --pose {centre[0]:.4f} {centre[1]:.4f}'
    lines += ['', 'Pick it (brick assumed to rest on the table, recommended):', f'  {base}{yaw_arg}',
              'or with the measured height:', f'  {base} {centre[2]:.4f}{yaw_arg}']
    return '\n'.join(lines)


# --------------------------------------------------------------------------- ROS part
class Camera:
    """Latest colour frame, aligned depth frame, CameraInfo and the arm pose at the colour stamp."""

    def __init__(self, node, c, t_ee_cam):
        from cv_bridge import CvBridge
        from rclpy.duration import Duration
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener
        self.node, self.c, self.t_ee_cam, self.bridge = node, c, t_ee_cam, CvBridge()
        self.color = self.depth = self.info = None
        self.buffer = Buffer(cache_time=Duration(seconds=30))
        self.listener = TransformListener(self.buffer, node)
        node.create_subscription(Image, c['image_topic'], lambda m: setattr(self, 'color', m), qos_profile_sensor_data)
        node.create_subscription(Image, c['depth_topic'], lambda m: setattr(self, 'depth', m), qos_profile_sensor_data)
        node.create_subscription(CameraInfo, c['camera_info_topic'],
                                 lambda m: setattr(self, 'info', m), qos_profile_sensor_data)

    def spin(self, seconds):
        import rclpy
        import time
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.03)

    def wait_ready(self, seconds=15):
        import time
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.spin(0.2)
            if self.color is not None and self.depth is not None and self.info is not None:
                return
        missing = [n for n, v in (('colour image', self.color), ('aligned depth', self.depth),
                                  ('camera_info', self.info)) if v is None]
        raise RuntimeError(f'No {", ".join(missing)}: is the RealSense running with align_depth.enable:=true? '
                           f'(topics: {self.c["image_topic"]}, {self.c["depth_topic"]})')

    def _arm_pose(self, stamp=None):
        from rclpy.time import Time
        from scipy.spatial.transform import Rotation
        tf = self.buffer.lookup_transform(self.c['base_frame'], self.c['ee_frame'],
                                          Time.from_msg(stamp) if stamp is not None else Time())
        q, t = tf.transform.rotation, tf.transform.translation
        return pose_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])

    def snapshot(self):
        """(bgr, depth_m, intrinsics, T_base_cam, warnings) for the newest frame."""
        color, depth, info = self.color, self.depth, self.info
        if color.header.frame_id != self.c['camera_frame']:
            raise RuntimeError(f'Colour frame {color.header.frame_id} != {self.c["camera_frame"]}')
        if (depth.width, depth.height) != (color.width, color.height):
            raise RuntimeError(f'Depth is {depth.width}x{depth.height} but colour is {color.width}x{color.height}: '
                               'use the ALIGNED depth topic (align_depth.enable:=true)')
        warnings = []
        try:
            t_base_ee = self._arm_pose(color.header.stamp)
        except Exception:
            t_base_ee = self._arm_pose()
            warnings.append('no arm pose at the picture time; used the newest one (fine if the arm is still)')
        newest = self._arm_pose()
        moved = np.linalg.norm(newest[:3, 3] - t_base_ee[:3, 3]) * 1000
        if moved > 1.0:
            warnings.append(f'the hand moved {moved:.1f} mm since the picture: results are not trustworthy')
        raw = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        depth_m = raw.astype(np.float64) * (0.001 if raw.dtype == np.uint16 else 1.0)
        k = info.k
        return (self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8'), depth_m,
                (k[0], k[4], k[2], k[5]), t_base_ee @ self.t_ee_cam, warnings)


def click_pixels(camera):
    """Interactive window. Returns (centre_px, long_end_px, snapshot) of the frozen picture, or None."""
    import cv2
    name = 'pixel_to_base: SPACE freeze | click centre, then end of long side | Enter accept | r reset | q quit'
    clicks, state = [], {'frozen': None}
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, lambda ev, x, y, f, p: clicks.append((x, y))
                         if ev == cv2.EVENT_LBUTTONDOWN and state['frozen'] is not None and len(clicks) < 2 else None)
    while True:
        camera.spin(0.03)
        if state['frozen'] is None:
            shown = camera.bridge.imgmsg_to_cv2(camera.color, desired_encoding='bgr8').copy()
        else:
            shown = state['frozen'].copy()
        for i, (x, y) in enumerate(clicks):
            cv2.drawMarker(shown, (x, y), (0, 255, 0) if i == 0 else (0, 200, 255), cv2.MARKER_CROSS, 24, 2)
        if len(clicks) == 2:
            cv2.line(shown, clicks[0], clicks[1], (0, 200, 255), 1)
        cv2.imshow(name, shown)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        if key == ord(' ') and state['frozen'] is None:
            snap = camera.snapshot()
            state['frozen'], state['snap'] = snap[0], snap
        if key == ord('r'):
            clicks.clear()
            state['frozen'] = None
        if key in (13, 10) and clicks:
            cv2.destroyAllWindows()
            return clicks[0], (clicks[1] if len(clicks) > 1 else None), state['snap']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--calibration', help='Hand-eye JSON (default: calibration_file in the config)')
    parser.add_argument('--pixel', nargs=2, type=float, metavar=('U', 'V'), help='Brick centre pixel (no window)')
    parser.add_argument('--long-end', nargs=2, type=float, metavar=('U', 'V'), help='Pixel at the end of the long side')
    parser.add_argument('--yaw', type=float, metavar='DEG', help='Brick long-axis angle, if no long-end pixel')
    parser.add_argument('--depth', type=float, metavar='M', help='OFFLINE: depth at the pixel, metres')
    parser.add_argument('--intrinsics', nargs=4, type=float, metavar=('FX', 'FY', 'CX', 'CY'), help='OFFLINE')
    parser.add_argument('--ee-pose', nargs=7, type=float, metavar='V', help='OFFLINE: hand pose in fr3_link0 as '
                        'x y z qx qy qz qw')
    parser.add_argument('--save', type=Path, help='Also write the result to this JSON file')
    args, ros_args = parser.parse_known_args()
    c = yaml.safe_load(open(args.config, encoding='utf-8'))
    calibration, t_ee_cam = load_calibration(args.calibration or c['calibration_file'], c, c['camera_frame'])
    print(f'Hand-eye: status {calibration["status"]}, typical error '
          f'{calibration["all_samples"]["translation_rms_m"] * 1000:.1f} mm / '
          f'{calibration["all_samples"]["rotation_rms_deg"]:.1f} deg')
    height, tolerance = c['block_size'][2], c.get('table_check_tolerance', 0.008)
    warnings = []
    if args.depth is not None:                                             # ---- offline
        if not (args.intrinsics and args.ee_pose and args.pixel):
            raise ValueError('Offline mode needs --pixel, --depth, --intrinsics and --ee-pose')
        t_base_cam = pose_matrix(args.ee_pose[:3], args.ee_pose[3:]) @ t_ee_cam
        intrinsics, centre_px, long_px, depth, spread = args.intrinsics, args.pixel, args.long_end, args.depth, 0.0
        long_depth = args.depth
    else:                                                                  # ---- live
        import rclpy
        rclpy.init(args=ros_args)
        node = rclpy.create_node('pixel_to_base')
        try:
            camera = Camera(node, c, t_ee_cam)
            camera.wait_ready()
            camera.spin(0.5)
            if args.pixel:
                bgr, depth_m, intrinsics, t_base_cam, warnings = camera.snapshot()
                centre_px, long_px = args.pixel, args.long_end
            else:
                picked = click_pixels(camera)
                if picked is None:
                    raise SystemExit('Cancelled.')
                centre_px, long_px, (bgr, depth_m, intrinsics, t_base_cam, warnings) = picked
            depth, spread = robust_depth(depth_m, *centre_px)
            long_depth = robust_depth(depth_m, *long_px)[0] if long_px else None
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
    result = brick_pose_from_pixels(t_base_cam, intrinsics, centre_px, depth, height, long_px, long_depth)
    if result['yaw'] is None and args.yaw is not None:
        result['yaw'], result['yaw_from'] = wrap_line_angle(math.radians(args.yaw)), '--yaw'
    for w in warnings:
        print(f'WARNING: {w}')
    print(report(c, args.config, result, depth, spread, tolerance))
    if args.save:
        args.save.write_text(json.dumps({
            'centre': result['centre'].tolist(), 'top': result['top'].tolist(),
            'yaw_rad': result['yaw'], 'depth_m': depth, 'pixel': list(centre_px),
            'table_check_mm': table_check(c, result['top'][2]) * 1000, 'warnings': warnings,
            'T_base_camera': np.asarray(t_base_cam).tolist()}, indent=2))
        print(f'Saved {args.save}')


if __name__ == '__main__':
    main()
