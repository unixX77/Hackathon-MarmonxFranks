"""Click RGB pixels to obtain metric XYZ in the camera color optical frame."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def pixel_to_xyz(u, v, z, intr):
    if not np.isfinite(z) or z <= 0:
        raise ValueError('Z must be positive and finite')
    if any(abs(c) > 1e-8 for c in intr.get('distortion_coeffs', [])):
        raise ValueError('This tool requires rectified/zero-distortion intrinsics')
    return [(u - intr['ppx']) * z / intr['fx'],
            (v - intr['ppy']) * z / intr['fy'], z]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path, help='Folder with rgb/, depth/, camera_info.json')
    p.add_argument('--frame', default='000000')
    p.add_argument('--z', type=float, help='Camera Z in metres; omit to use aligned depth at the click')
    p.add_argument('--out', type=Path, default=Path('data/clicked_points'))
    args = p.parse_args()
    info = json.loads((args.recording / 'camera_info.json').read_text())
    intr = info.get('aligned_depth_intrinsics', info.get('color_intrinsics'))
    if intr is None:
        p.error('Missing color intrinsics')
    rgb = cv2.imread(str(args.recording / 'rgb' / f'{args.frame}.png'))
    if rgb is None:
        p.error('RGB frame not found')
    if rgb.shape[:2] != (intr['height'], intr['width']):
        p.error('RGB size does not match calibration')
    # Validate projection support before opening the window.
    pixel_to_xyz(0, 0, args.z if args.z is not None else 1., intr)
    depth = None
    if args.z is None:
        if not info.get('depth_aligned_to_color'):
            p.error('Depth mode requires depth aligned to color')
        depth = cv2.imread(str(args.recording / 'depth' / f'{args.frame}.png'), -1)
        if depth is None or depth.shape != rgb.shape[:2] or depth.dtype != np.uint16:
            p.error('Missing aligned uint16 depth matching RGB size')
    points = []
    view = rgb.copy()
    title = 'Click XYZ | left: select | u: undo | s: save | q: quit'

    def redraw():
        nonlocal view
        view = rgb.copy()
        for i, point in enumerate(points, 1):
            u, v = point['pixel_uv']
            cv2.drawMarker(view, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
            text = f"#{i} X={point['xyz_m'][0]:.4f} Y={point['xyz_m'][1]:.4f} Z={point['xyz_m'][2]:.4f} m"
            x = max(0, min(u + 10, rgb.shape[1] - 480))
            y = max(20, v - 10)
            cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
            cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)

    def click(event, u, v, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        z = args.z if args.z is not None else float(depth[v, u]) * info['depth_scale_m']
        if z <= 0:
            print(f'Pixel ({u}, {v}): missing depth; choose another pixel.', flush=True)
            return
        xyz = pixel_to_xyz(u, v, z, intr)
        points.append({'pixel_uv': [u, v], 'xyz_m': xyz})
        print(f'Pixel ({u}, {v}): X={xyz[0]:.6f}, Y={xyz[1]:.6f}, Z={xyz[2]:.6f} metres', flush=True)
        redraw()

    def save():
        output = args.out / args.recording.name / args.frame
        output.mkdir(parents=True, exist_ok=True)
        (output / 'points.json').write_text(json.dumps({
            'recording': str(args.recording.resolve()), 'frame': args.frame,
            'frame_id': 'camera_color_optical_frame', 'units': 'metres',
            'coordinate_convention': 'x right, y down, z forward',
            'z_source': 'user supplied camera Z' if args.z is not None else 'aligned depth pixel',
            'points': points}, indent=2))
        if not cv2.imwrite(str(output / 'points.png'), view):
            raise RuntimeError('Could not save annotated image')
        print('Saved:', output, flush=True)

    print('Camera coordinates: X right, Y down, Z forward; metres.')
    print('Z is distance along the optical axis, not table height or range along the viewing ray.')
    print(f'Z source: {args.z} m' if args.z is not None else 'Z source: aligned depth at selected pixel')
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, click)
    try:
        while True:
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('u') and points:
                points.pop()
                redraw()
            if key == ord('s'):
                save()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
