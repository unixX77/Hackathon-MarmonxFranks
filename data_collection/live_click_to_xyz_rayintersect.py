"""Click the live D435 RGB feed: camera-frame XYZ of a block's top face by ray-plane intersection.

1. Fit the table plane to the aligned depth image (RANSAC, the block is rejected as an outlier).
2. Shift that plane towards the camera by --block-height  -> the plane of the block's top face.
3. Intersect the clicked pixel's ray with it                -> XYZ in camera_color_optical_frame.

No robot pose needed; camera tilt is handled because the plane is measured, not assumed.
Also prints the camera's tilt from the table normal and the median-depth answer as a cross-check.
Use --live for repeated measurements at the selected pixel (not object tracking).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from capture_d435 import load_settings


def depth_to_points(depth, intr, scale, step=6, zmin=0.1, zmax=1.5):
    """Subsampled 3D points (camera frame, m) from an aligned depth image, pinhole model."""
    d = depth[::step, ::step].astype(float) * scale
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    m = (d > zmin) & (d < zmax)
    z = d[m]
    x = (u[m] - intr.ppx) / intr.fx * z
    y = (v[m] - intr.ppy) / intr.fy * z
    return np.c_[x, y, z]


def fit_plane(pts, iters=300, thresh=0.004, rng=np.random.default_rng(0)):
    """RANSAC + least-squares refine. Returns (n, c, inlier_ratio) with n.p + c = 0, n towards camera."""
    if len(pts) < 100:
        raise ValueError('not enough valid depth points to fit the table')
    best = None
    for _ in range(iters):
        a, b, c3 = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c3 - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(pts @ n - n @ a) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    P = pts[best]
    centroid = P.mean(0)
    n = np.linalg.svd(P - centroid, full_matrices=False)[2][-1]
    c = -n @ centroid
    if c < 0:  # orient the normal towards the camera (camera at origin is on the + side)
        n, c = -n, -c
    return n, c, best.mean()


def ray_plane_cam(u, v, intr, n, c, h):
    """Intersect the pixel ray with the plane shifted h metres towards the camera: n.p + c - h = 0."""
    r = np.array(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0))  # handles distortion
    denom = n @ r
    if abs(denom) < 1e-6:
        raise ValueError('ray parallel to the plane')
    t = (h - c) / denom
    if t <= 0:
        raise ValueError('plane is behind the camera')
    return t * r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--block-height', type=float, default=0.0096,
                   help='height of the clicked top face above the table, m (LEGO brick body 0.0096)')
    p.add_argument('--win', type=int, default=4, help='half window for the median-depth cross-check')
    p.add_argument('--live', action='store_true', help='Refit table and update the selected pixel continuously')
    p.add_argument('--update-interval', type=float, default=0.5,
                   help='Seconds between live plane fits and terminal updates (default: 0.5)')
    p.add_argument('--settings', type=Path, default=Path(__file__).resolve().parents[1] / 'd435_settings.json')
    p.add_argument('--serial')
    args = p.parse_args()
    if not np.isfinite(args.block_height) or args.block_height < 0:
        p.error('--block-height must be finite and nonnegative')
    if args.win < 0:
        p.error('--win must be nonnegative')
    if not np.isfinite(args.update_interval) or args.update_interval <= 0:
        p.error('--update-interval must be positive and finite')
    settings = args.settings.read_text()
    viewer = json.loads(settings).get('viewer', {})
    width = int(viewer.get('stream-width', 848))
    height = int(viewer.get('stream-height', 480))
    fps = int(viewer.get('stream-fps', 30))
    serial = load_settings(args.serial, settings)
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    title = 'Ray-plane XYZ | click: select | l: live/snapshot | q: quit'
    current = None
    selected = None
    live = args.live
    last_update = 0.0

    def measure(x, y):
        depth, intr, scale = current
        try:
            n, c, ratio = fit_plane(depth_to_points(depth, intr, scale))
            xyz = ray_plane_cam(x, y, intr, n, c, args.block_height)
        except ValueError as e:
            print(f'Pixel ({x}, {y}): {e}', flush=True)
            return None
        tilt = np.degrees(np.arccos(min(1.0, abs(n[2]))))
        msg = (f'Pixel ({x}, {y}): X={xyz[0]:.6f}, Y={xyz[1]:.6f}, Z={xyz[2]:.6f} m   (ray-plane)\n'
               f'    table: {c * 1000:.1f} mm from camera, camera tilt {tilt:.2f} deg, '
               f'{ratio * 100:.0f}% of depth points on the table')
        if ratio < 0.5:
            msg += '  WARNING: plane fit may have locked onto something other than the table'
        # cross-check with median depth at the click
        h, w = depth.shape
        patch = depth[max(0, y - args.win):min(h, y + args.win + 1),
                      max(0, x - args.win):min(w, x + args.win + 1)].astype(float) * scale
        patch = patch[patch > 0]
        if patch.size:
            xyz_d = np.array(rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(np.median(patch))))
            diff = (xyz_d - xyz) * 1000
            msg += (f'\n    depth check: X={xyz_d[0]:.6f}, Y={xyz_d[1]:.6f}, Z={xyz_d[2]:.6f} m  '
                    f'diff {diff.round(1)} mm')
        print(msg, flush=True)
        return xyz

    def click(event, x, y, flags, userdata):
        nonlocal selected, last_update
        if event != cv2.EVENT_LBUTTONDOWN or current is None:
            return
        selected = (x, y, measure(x, y))
        last_update = time.monotonic()

    started = False
    try:
        profile = pipe.start(config)
        started = True
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipe.wait_for_frames()
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, click)
        print('Camera color optical frame: X right, Y down, Z forward; metres.')
        print(f'Block top assumed {args.block_height * 1000:.1f} mm above the table. Click the block top-face centre.')
        print('Press l for live/snapshot mode. Live refits the plane at the selected fixed pixel; no object tracking.')
        while True:
            frames = align.process(pipe.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            view = np.asanyarray(color.get_data()).copy()
            intr = color.profile.as_video_stream_profile().get_intrinsics()  # COLOUR intrinsics
            current = (np.asanyarray(depth.get_data()).copy(), intr, scale)
            if selected:
                x, y, xyz = selected
                if live and time.monotonic() - last_update >= args.update_interval:
                    xyz = measure(x, y)
                    selected = (x, y, xyz)
                    last_update = time.monotonic()
                cv2.drawMarker(view, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                text = ('Live: ' if live else 'Snapshot: ') + (
                    ' '.join(f'{k}={v:.4f}' for k, v in zip('XYZ', xyz)) + ' m'
                    if xyz is not None else 'Plane measurement unavailable')
                cv2.putText(view, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                cv2.putText(view, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cv2.imshow(title, view)
            key = cv2.waitKey(10) & 0xff
            if key == ord('l'):
                live = not live
                last_update = 0.0
                print('Mode:', 'live (fixed pixel)' if live else 'snapshot', flush=True)
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if started:
            pipe.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
