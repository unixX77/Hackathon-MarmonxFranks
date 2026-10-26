"""Click the live D435 RGB feed to print camera-frame metric XYZ (camera_color_optical_frame).

Depth = median of valid aligned depth in a small window around the click (robust to noise/holes).
Use --live to update XYZ continuously at the selected pixel; l toggles live/snapshot.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from capture_d435 import load_settings
from click_to_xyz import pixel_to_xyz
from capture_rgbd import intrinsics_to_dict


def median_depth(depth, x, y, scale, half_win):
    """Median valid depth (m) in a (2*half_win+1)^2 window. Returns (z, n_valid, iqr_m)."""
    h, w = depth.shape
    patch = depth[max(0, y - half_win):min(h, y + half_win + 1),
                  max(0, x - half_win):min(w, x + half_win + 1)].astype(float) * scale
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0.0, 0, 0.0
    q25, q75 = np.percentile(vals, [25, 75])
    return float(np.median(vals)), int(vals.size), float(q75 - q25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--z', type=float, help='Fixed camera Z in metres (overrides depth; not recommended)')
    p.add_argument('--win', type=int, default=4, help='Half window for median depth in pixels (4 -> 9x9)')
    p.add_argument('--live', action='store_true', help='Continuously measure the last selected pixel')
    p.add_argument('--settings', type=Path, default=Path(__file__).resolve().parents[1] / 'd435_settings.json')
    p.add_argument('--serial')
    args = p.parse_args()
    if args.z is not None and (not np.isfinite(args.z) or args.z <= 0):
        p.error('--z must be positive and finite')
    if args.win < 0:
        p.error('--win must be >= 0')
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
    title = 'Median depth XYZ | click: select | l: live/snapshot | q: quit'
    current = None
    selected = None
    live = args.live
    last_print = 0.0

    def measure(x, y, announce=False):
        depth, intr, scale = current
        if args.z is not None:
            z, n, iqr = args.z, 0, 0.0
        else:
            z, n, iqr = median_depth(depth, x, y, scale, args.win)
        if z <= 0:
            if announce:
                print('No valid depth around that pixel. Choose another point.', flush=True)
            return None
        xyz = pixel_to_xyz(x, y, z, intr)
        msg = f'Pixel ({x}, {y}): X={xyz[0]:.6f}, Y={xyz[1]:.6f}, Z={xyz[2]:.6f} m'
        if args.z is None:
            msg += f'  [{n} depth px, spread {iqr * 1000:.1f} mm]'
            if iqr > 0.01:
                msg += '  WARNING: depth varies >10 mm here (block edge?), click the top-face centre'
        if announce:
            print(msg, flush=True)
        return xyz

    def click(event, x, y, flags, userdata):
        nonlocal selected, last_print
        if event != cv2.EVENT_LBUTTONDOWN or current is None:
            return
        selected = (x, y, measure(x, y, announce=True))
        last_print = time.monotonic()

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
        print('Click selects a pixel. Press l to toggle continuous measurements; no object tracking.')
        while True:
            frames = align.process(pipe.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            view = np.asanyarray(color.get_data()).copy()
            # Aligned depth and displayed RGB share the color pixel grid.
            current = (np.asanyarray(depth.get_data()).copy(),
                       intrinsics_to_dict(color.profile.as_video_stream_profile().get_intrinsics()), scale)
            if selected:
                x, y, xyz = selected
                if live:
                    announce = time.monotonic() - last_print >= 0.5
                    xyz = measure(x, y, announce=announce)
                    selected = (x, y, xyz)
                    if announce:
                        last_print = time.monotonic()
                if args.z is None:
                    cv2.rectangle(view, (x - args.win, y - args.win), (x + args.win, y + args.win), (0, 255, 255), 1)
                cv2.drawMarker(view, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                label = 'Live: ' if live else 'Snapshot: '
                text = label + (' '.join(f'{k}={v:.4f}' for k, v in zip('XYZ', xyz)) + ' m'
                                if xyz is not None else 'No valid depth')
                cv2.putText(view, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                cv2.putText(view, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cv2.imshow(title, view)
            key = cv2.waitKey(10) & 0xff
            if key == ord('l'):
                live = not live
                print('Mode:', 'live (fixed pixel)' if live else 'snapshot', flush=True)
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if started:
            pipe.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
