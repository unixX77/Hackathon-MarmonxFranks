"""Compare median depth, ray-plane and SAM3 top-face XYZ on one frozen RGB-D frame."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

from capture_d435 import load_settings
from capture_rgbd import camera_info, intrinsics_to_dict
from live_click_to_xyz_mediandepth import median_depth
from live_click_to_xyz_rayintersect import depth_to_points, fit_plane, ray_plane_cam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'perception/sam3_detection'))
from top_layer_bricks import build_parser, process_frame


def sdk_intrinsics(info):
    intr = rs.intrinsics()
    intr.width, intr.height = info['width'], info['height']
    intr.fx, intr.fy = info['fx'], info['fy']
    intr.ppx, intr.ppy = info['ppx'], info['ppy']
    intr.coeffs = info.get('distortion_coeffs', [0.] * 5)
    intr.model = getattr(rs.distortion, info.get('distortion_model', 'none'))
    return intr


def compare(u, v, depth, intr, scale, block_height, win, plane, brick=None):
    # A matched brick is compared at its projected TOP-FACE centre, so all
    # three estimates refer to the same image location, rather than an arbitrary click.
    sam = np.asarray(brick['centre_cam_m']) if brick else None
    if sam is not None:
        u, v = rs.rs2_project_point_to_pixel(intr, sam.tolist())
    x, y = int(round(u)), int(round(v))
    print('\n' + '=' * 76)
    print('camera_color_optical_frame | metres | X right, Y down, Z forward')
    print(f'Comparison pixel: ({u:.2f}, {v:.2f})' +
          (f" | SAM3 brick #{brick['id']} TOP-FACE centre" if brick else ' | clicked point (no matched brick)'))
    values = {}
    if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
        z, count, iqr = median_depth(depth, x, y, scale, win)
        if z > 0:
            values['Median depth'] = np.asarray(rs.rs2_deproject_pixel_to_point(intr, [u, v], z))
            print(f'Depth support: {count} pixels, IQR={iqr*1000:.2f} mm')
    if plane is not None:
        n, c, ratio = plane
        try:
            values['Ray-plane'] = ray_plane_cam(u, v, intr, n, c, block_height)
        except ValueError as error:
            print('Ray-plane unavailable:', error)
        print(f'Table inliers: {ratio*100:.1f}% | assumed top height: {block_height*1000:.1f} mm')
    if sam is not None:
        values['SAM3 pipeline'] = sam
    print(f"{'Method':<20}{'X (m)':>13}{'Y (m)':>13}{'Z (m)':>13}")
    for name in ('Median depth', 'Ray-plane', 'SAM3 pipeline'):
        if name in values:
            print(f'{name:<20}' + ''.join(f'{value:13.6f}' for value in values[name]))
        else:
            print(f'{name:<20} unavailable')
    if sam is not None:
        for name in ('Median depth', 'Ray-plane'):
            if name in values:
                delta = (values[name] - sam) * 1000
                print(f'{name} - SAM3 (mm): {np.round(delta, 3).tolist()}')
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', type=Path, default=ROOT / 'd435_settings.json')
    p.add_argument('--serial')
    p.add_argument('--recording', type=Path, help='Use a saved session instead of live camera')
    p.add_argument('--frame', default='000000')
    p.add_argument('--pixel', nargs=2, type=int, metavar=('U', 'V'), help='Saved recording only: compare once without a window')
    p.add_argument('--block-height', type=float, default=.01, help='Top face height above table in metres')
    p.add_argument('--win', type=int, default=4)
    p.add_argument('--brick-size-mm', nargs=3, type=float, default=[30,15,10])
    p.add_argument('--threshold', type=float, default=.5)
    args = p.parse_args()
    if args.win < 0 or not np.isfinite(args.block_height) or args.block_height < 0:
        p.error('Window and block height must be nonnegative and finite')
    if args.pixel and not args.recording:
        p.error('--pixel requires --recording')
    import torch
    from transformers import Sam3Model, Sam3Processor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate sam3 and check nvidia-smi')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    processor = Sam3Processor.from_pretrained('facebook/sam3')
    model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=dtype).to('cuda').eval()
    pipeline_args = build_parser().parse_args([str(args.recording or Path('live_comparison'))])
    pipeline_args.brick_size_mm = args.brick_size_mm
    pipeline_args.block_height_mm = args.brick_size_mm[2]
    pipeline_args.threshold = args.threshold
    state = {'packet': None, 'result': None, 'paused': bool(args.recording), 'preview': None}
    title = 'Compare XYZ | click brick: freeze + compare | r: resume | q: quit'
    preview_title = 'SAM3 pipeline | same frozen comparison frame'

    def analyze():
        bgr, depth, calibration, intr = state['packet']
        scale = calibration['depth_scale_m']
        try:
            plane = fit_plane(depth_to_points(depth, intr, scale))
        except ValueError as error:
            print('Table-plane fit failed:', error, flush=True)
            plane = None
        try:
            images, metadata, bricks = process_frame(bgr, depth, calibration, model, processor,
                                                     pipeline_args, 'cuda', dtype)
            instance = images['final_instances.png']
            state['preview'] = images['preview.png']
            report = metadata['depth_report']
            print('\nSAM3 depth-stage diagnostics:', flush=True)
            for record in report['instances']:
                height = record.get('height_m')
                spread = record.get('height_spread_m')
                print(f"  mask #{record['id']}: "
                      f"height_mm={height * 1000 if height is not None else 'N/A'}, "
                      f"spread_mm={spread * 1000 if spread is not None else 'N/A'}, "
                      f"valid_pixels={record['valid_pixels']}, "
                      f"kept={record['kept']}, reason={record.get('reason', 'accepted')}", flush=True)
        except ValueError as error:
            print('SAM3 pipeline:', error, flush=True)
            instance, bricks = np.zeros(depth.shape, np.uint16), []
            state['preview'] = None
        state['result'] = (plane, instance, bricks)

    def select(u, v):
        bgr, depth, calibration, intr = state['packet']
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            print('Click outside image')
            return
        state['paused'] = True
        if state['result'] is None:
            print('Frozen frame: running SAM3 once...', flush=True)
            analyze()
        plane, instance, bricks = state['result']
        ident = int(instance[v,u])
        brick = next((b for b in bricks if b['id']==ident), None)
        if brick is None:
            candidates = [b for b in bricks if cv2.pointPolygonTest(
                np.array(b['footprint_px'],np.float32), (float(u),float(v)), False)>=0]
            if len(candidates)==1:
                brick = candidates[0]
        compare(u,v,depth,intr,calibration['depth_scale_m'],args.block_height,args.win,plane,brick)
        if not args.pixel and state['preview'] is not None:
            cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_title, 1440, 800)
            cv2.imshow(preview_title, state['preview'])
            cv2.waitKey(1)
        print('Coordinates refer to this frozen frame. Click again or press r to resume.',flush=True)

    pipe = rs.pipeline()
    started = False
    try:
        if args.recording:
            calibration=json.loads((args.recording/'camera_info.json').read_text())
            if not calibration.get('depth_aligned_to_color'):
                raise ValueError('Saved depth must be aligned to color')
            bgr=cv2.imread(str(args.recording/'rgb'/f'{args.frame}.png'))
            depth=cv2.imread(str(args.recording/'depth'/f'{args.frame}.png'),-1)
            intr=sdk_intrinsics(calibration['aligned_depth_intrinsics'])
            if bgr is None or depth is None or depth.dtype != np.uint16 or depth.shape!=bgr.shape[:2]:
                raise ValueError('Missing/mismatched RGB and aligned uint16 depth')
            state['packet']=(bgr,depth,calibration,intr)
            if args.pixel:
                select(*args.pixel)
                return
        else:
            settings=args.settings.read_text();viewer=json.loads(settings).get('viewer',{})
            serial=load_settings(args.serial,settings)
            config=rs.config();config.enable_device(serial)
            fps=int(viewer.get('stream-fps',30))
            config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,fps)
            config.enable_stream(rs.stream.depth,int(viewer.get('stream-width',848)),int(viewer.get('stream-height',480)),rs.format.z16,fps)
            profile=pipe.start(config);started=True
            align=rs.align(rs.stream.color)
            for _ in range(30): pipe.wait_for_frames()
            scale=profile.get_device().first_depth_sensor().get_depth_scale()
            calibration=camera_info(profile,scale,None)
        cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title,lambda event,x,y,flags,param: select(x,y)
                            if event==cv2.EVENT_LBUTTONDOWN and state['packet'] is not None else None)
        while True:
            if not state['paused']:
                frames=align.process(pipe.wait_for_frames())
                color,depth=frames.get_color_frame(),frames.get_depth_frame()
                if not color or not depth: continue
                intr=color.profile.as_video_stream_profile().get_intrinsics()
                calibration['aligned_depth_intrinsics']=intrinsics_to_dict(depth.profile.as_video_stream_profile().get_intrinsics())
                state['packet']=(np.asanyarray(color.get_data()).copy(),np.asanyarray(depth.get_data()).copy(),calibration,intr)
            view=state['packet'][0].copy()
            cv2.putText(view,'FROZEN - click to compare | r resume' if state['paused'] else 'LIVE - click a brick to freeze and compare',
                        (8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imshow(title,view)
            key=cv2.waitKey(10)&0xff
            if key in (ord('q'),27) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1: break
            if key==ord('r') and not args.recording:
                if state['preview'] is not None:
                    cv2.destroyWindow(preview_title)
                state['paused']=False;state['result']=None;state['preview']=None
    finally:
        if started: pipe.stop()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
