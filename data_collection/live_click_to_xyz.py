"""Click the live D435 RGB feed to print camera-frame metric XYZ."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from capture_d435 import load_settings
from click_to_xyz import pixel_to_xyz
from capture_rgbd import intrinsics_to_dict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--z', type=float, help='Fixed camera Z in metres; otherwise use aligned depth')
    p.add_argument('--settings', type=Path, default=Path(__file__).resolve().parents[1] / 'd435_settings.json')
    p.add_argument('--serial')
    args = p.parse_args()
    if args.z is not None and (not np.isfinite(args.z) or args.z <= 0):
        p.error('--z must be positive and finite')
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
    title = 'Live click XYZ | click: XYZ | q: quit'
    current = None
    selected = None

    def click(event, x, y, flags, userdata):
        nonlocal selected
        if event != cv2.EVENT_LBUTTONDOWN or current is None:
            return
        depth, intr, scale = current
        z = args.z if args.z is not None else float(depth[y, x]) * scale
        if z <= 0:
            print('Missing depth at that pixel. Choose another point.', flush=True)
            return
        xyz = pixel_to_xyz(x, y, z, intr)
        selected = (x, y, xyz)
        print(f'Pixel ({x}, {y}): X={xyz[0]:.6f}, Y={xyz[1]:.6f}, Z={xyz[2]:.6f} m', flush=True)

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
        print('Click measurements refer to that instant; markers show the last clicked pixel.')
        while True:
            frames = align.process(pipe.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            view = np.asanyarray(color.get_data()).copy()
            current = (np.asanyarray(depth.get_data()).copy(),
                       intrinsics_to_dict(depth.profile.as_video_stream_profile().get_intrinsics()), scale)
            if selected:
                x, y, xyz = selected
                cv2.drawMarker(view, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                text = 'Last click: ' + ' '.join(f'{k}={v:.4f}' for k, v in zip('XYZ', xyz)) + ' m'
                cv2.putText(view, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                cv2.putText(view, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cv2.imshow(title, view)
            if cv2.waitKey(10) & 0xff in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if started:
            pipe.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
