#!/usr/bin/env python3
"""Export matched LEGO RGB-D recordings and saved outputs for the static playground.

Example:
  python tools/export_playground.py --project /path/to/Hackathon-MarmonxFranks \
    --scene data3:20261024_091244 --scene data3:20261024_091640 \
    --scene data3:20261024_091956 --output playground

Requires numpy, Pillow, matplotlib and OpenCV (for the optional SAM 3 mask scene). The original SAM 3 masks were not saved
by top-plane-first-v2; the exporter does not substitute rectangles as masks.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import colormaps

DEFAULT_SCENES = ('data3:20261024_091244', 'data3:20261024_091640', 'data3:20261024_091956')
NAMES = {'20261024_091244': 'Dense top layer', '20261024_091640': 'Partially cleared stack',
         '20261024_091956': 'Sparse top layer'}

def load(path):
    return json.loads(path.read_text())

def finite_list(value, n):
    return isinstance(value, (list, tuple)) and len(value) == n and all(isinstance(x, (int, float)) and math.isfinite(x) for x in value)

def export_scene(root, out, dataset, recording, frame, url_prefix):
    recording_dir = root / 'data_collection/extracted_rgbd/extracted' / dataset / recording
    result_dir = root / 'perception/sam3_detection/results_rgbd_topplane_v2' / dataset / recording / f'{frame:06d}'
    quality_dir = root / 'classifier/results_rgbd_quality_v1' / dataset / recording / f'{frame:06d}'
    rgb_file, depth_file = recording_dir / 'rgb' / f'{frame:06d}.png', recording_dir / 'depth' / f'{frame:06d}.png'
    info_file, top_file = recording_dir / 'camera_info.json', result_dir / 'top_boxes.json'
    quality_file, pose_file = quality_dir / 'quality_boxes.json', quality_dir / 'camera_boxes.json'
    for f in (rgb_file, depth_file, info_file, top_file, quality_file, pose_file):
        if not f.is_file(): raise FileNotFoundError(f'Missing matched scene file: {f}')
    info, top, quality, pose = map(load, (info_file, top_file, quality_file, pose_file))
    for obj in (top, quality, pose):
        if (obj.get('dataset'), obj.get('recording'), obj.get('frame')) != (dataset, recording, frame):
            raise ValueError(f'Output frame does not match {dataset}/{recording}/{frame}')
    rgb = Image.open(rgb_file).convert('RGB'); depth = np.asarray(Image.open(depth_file))
    intr = info['aligned_depth_intrinsics']; color_intr = info['color_intrinsics']; width, height = rgb.size
    if any(not math.isclose(float(intr[k]), float(color_intr[k]), abs_tol=1e-6) for k in ('fx', 'fy', 'ppx', 'ppy')):
        raise ValueError('Color and aligned-depth intrinsics differ; cannot safely project')
    if not info.get('depth_aligned_to_color') or (depth.shape[1], depth.shape[0]) != rgb.size:
        raise ValueError('Depth is not aligned to the RGB frame')
    if (intr['width'], intr['height']) != rgb.size: raise ValueError('Intrinsics dimensions do not match RGB')
    qboxes = {x['id']: x for x in quality['boxes']}; pboxes = {x['id']: x for x in pose['bricks']}
    tboxes = {x['id']: x for x in top['boxes']}
    if len(qboxes) != len(quality['boxes']) or qboxes.keys() != pboxes.keys() or qboxes.keys() != tboxes.keys():
        raise ValueError('Brick IDs are not unique and matched across outputs')
    for key in qboxes:
        if not np.allclose(qboxes[key]['corners_xy'], tboxes[key]['corners_xy'], atol=0.01) or not np.allclose(qboxes[key]['corners_xy'], pboxes[key]['corners_pixel_xy'], atol=0.01):
            raise ValueError(f'Detection and classification geometry differs for brick {key}')
    dist = intr.get('distortion_coeffs')
    can_project = isinstance(dist, list) and bool(dist) and all(abs(float(x)) < 1e-10 for x in dist)
    def project(v):
        if not can_project or not finite_list(v, 3) or v[2] <= 0: return None
        return [round(intr['fx'] * v[0] / v[2] + intr['ppx'], 3),
                round(intr['fy'] * v[1] / v[2] + intr['ppy'], 3)]
    bricks=[]
    for key in sorted(qboxes):
        q, p = qboxes[key], pboxes[key]
        corners = q['corners_xy']; center = q['center_xy']
        if len(corners) != 4 or not all(finite_list(c, 2) for c in corners) or not finite_list(center, 2):
            raise ValueError(f'Invalid image geometry for brick {key}')
        points = p.get('box_corners_camera_m') or []
        projected = [project(x) for x in points]
        if len(projected) != 8 or any(x is None for x in projected): projected = None
        bricks.append({'id':key, 'corners':corners, 'center':center, 'quality':q.get('quality'),
                       'colour':q.get('colour'), 'confidence_category':q.get('confidence'),
                       'colour_distance':q.get('colour_distance'), 'runner_up':q.get('runner_up'),
                       'angle_deg':q.get('angle_deg'), 'top_coverage':q.get('top_coverage'),
                       'source':q.get('source'), 'top_face_center_camera_m':p.get('top_face_center_camera_m'),
                       'box_center_camera_m':p.get('box_center_camera_m'),
                       'box_corners_camera_m':points if len(points)==8 else None,
                       'rotation_matrix_camera':p.get('box_rotation_camera'),
                       'projected_box_corners':projected})
    scale = float(info['depth_scale_m']); meters = depth.astype(np.float32) * scale
    valid = np.isfinite(meters) & (depth > 0)
    if not valid.any(): raise ValueError('Depth frame has no valid pixels')
    low, high = map(float, np.percentile(meters[valid], [2, 98]))
    if high <= low: high = low + 0.001
    normalized = np.clip((meters - low) / (high - low), 0, 1)
    heat = (colormaps['viridis'](normalized)[..., :3] * 255).astype(np.uint8)
    heat[~valid] = 0
    scene_id = f'{dataset}-{recording}-{frame:06d}'
    dst = out / 'scenes' / scene_id; dst.mkdir(parents=True, exist_ok=True)
    rgb.save(dst / 'rgb.jpg', quality=91, optimize=True)
    rgb.copy().resize((240,180), Image.Resampling.LANCZOS).save(dst / 'thumb.jpg', quality=82, optimize=True)
    Image.fromarray(heat).save(dst / 'depth.png', optimize=True)
    # Pose visualization is a real saved image from the same recorded frame.
    pose_preview = root / 'classifier/viz_box' / dataset / recording / f'{frame:06d}' / 'pose_visualization.png'
    if pose_preview.is_file(): Image.open(pose_preview).convert('RGB').save(dst / 'pose.jpg', quality=88, optimize=True)
    obj={'id':scene_id, 'dataset':dataset, 'recording':recording, 'frame':frame, 'width':width, 'height':height,
         'intrinsics':{'fx':intr['fx'],'fy':intr['fy'],'cx':intr['ppx'],'cy':intr['ppy'],'distortion_coeffs':dist},
         'coordinate_convention':'camera_color_optical_frame: +x right, +y down, +z forward; metres',
         'depth':{'scale_m':scale,'display_min_m':round(low,4),'display_max_m':round(high,4),
                  'description':'Aligned single depth frame; colors are clipped at the 2nd and 98th valid-depth percentiles. Black pixels are invalid.'},
         'masks_available':False,'masks_reason':'The original SAM 3 instance masks were not saved with these top-plane-first-v2 outputs.',
         'projection_available':can_project,'bricks':bricks}
    (dst/'scene.json').write_text(json.dumps(obj,separators=(',',':'))+'\n')
    return {'id':scene_id,'name':NAMES.get(recording, f'{dataset} · {recording}'),
            'description':f'{len(bricks)} saved top-layer candidates',
            'thumbnail':f'{url_prefix}/scenes/{scene_id}/thumb.jpg',
            'rgb':f'{url_prefix}/scenes/{scene_id}/rgb.jpg',
            'depth':f'{url_prefix}/scenes/{scene_id}/depth.png',
            'metadata':f'{url_prefix}/scenes/{scene_id}/scene.json',
            'available':{'rgb':True,'depth':True,'segmentation':False,'detection':True,
                         'quality':True,'pose':can_project}}

def export_rgb_sam_scene(root, out, image_name, url_prefix):
    # This is an earlier standalone, RGB-only SAM 3 run. Its instance IDs are
    # not the top-plane-v2 brick IDs; never combine the two outputs.
    import cv2
    rgb_file = root / 'data 2/rgb' / f'{image_name}.png'
    source = root / 'perception/sam3_detection/results' / image_name
    instance_file, metadata_file = source / 'instance.png', source / 'instances.json'
    for path in (rgb_file, instance_file, metadata_file):
        if not path.is_file(): raise FileNotFoundError(f'Missing SAM 3 source: {path}')
    rgb = Image.open(rgb_file).convert('RGB'); labels = np.asarray(Image.open(instance_file))
    records = load(metadata_file)['instances']
    if labels.shape != (rgb.height, rgb.width): raise ValueError('SAM instance map does not match RGB dimensions')
    record_ids = {int(item['id']) for item in records}
    mask_ids = set(map(int, np.unique(labels))) - {0}
    if record_ids != mask_ids or len(records) != len(record_ids): raise ValueError('SAM mask IDs do not match metadata')
    preview = source / 'preview.png'
    if preview.is_file():
        preview_rgb = Image.open(preview).convert('RGB').crop((0,0,rgb.width,rgb.height))
        if not np.array_equal(np.asarray(rgb), np.asarray(preview_rgb)):
            raise ValueError('SAM preview and selected RGB frame do not match')
    scene_id = f'rgb-sam3-{image_name}'; dst = out / 'scenes' / scene_id
    (dst/'masks').mkdir(parents=True,exist_ok=True)
    rgb.save(dst/'rgb.jpg', quality=91, optimize=True)
    rgb.copy().resize((240,180), Image.Resampling.LANCZOS).save(dst/'thumb.jpg', quality=82, optimize=True)
    palette = [(37, 181, 245), (247, 94, 156), (255, 194, 73), (103, 220, 147), (169, 133, 255), (245, 128, 84)]
    bricks=[]
    for index, item in enumerate(records):
        key=int(item['id']); mask=(labels==key).astype(np.uint8)
        area=int(mask.sum())
        if area != int(item['area_px']): raise ValueError(f'SAM mask area mismatch for instance {key}')
        rgba=np.zeros((rgb.height,rgb.width,4),dtype=np.uint8)
        rgba[:,:,:3]=palette[index % len(palette)]; rgba[:,:,3]=mask*255
        Image.fromarray(rgba).save(dst/'masks'/f'{key}.png',optimize=True)
        contours,_ = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        parts=[]
        for contour in contours:
            points=contour.reshape(-1,2)
            if len(points)<3: continue
            parts.append('M '+' L '.join(f'{int(x)} {int(y)}' for x,y in points)+' Z')
        if not parts: raise ValueError(f'No usable SAM contour for instance {key}')
        bricks.append({'id':key,'center':item.get('centroid_xy'),'mask_area':area,
                       'sam_score':item.get('score'),'colour':item.get('colour'),
                       'mask_image':f'{url_prefix}/scenes/{scene_id}/masks/{key}.png',
                       'mask_path':' '.join(parts),'quality':None})
    obj={'id':scene_id,'dataset':'standalone RGB SAM 3','recording':image_name,
         'frame':None,'width':rgb.width,'height':rgb.height,'depth':None,
         'coordinate_convention':'RGB image coordinates; no camera-frame pose saved',
         'masks_available':True,'masks_reason':'', 'projection_available':False,
         'scene_note':'Earlier RGB-only standalone SAM 3 run; not the top-layer RGB-D pipeline.', 'bricks':bricks}
    (dst/'scene.json').write_text(json.dumps(obj,separators=(',',':'))+'\n')
    return {'id':scene_id,'name':'SAM 3 candidates · RGB only',
            'description':f'Earlier RGB-only run · {len(bricks)} masks',
            'thumbnail':f'{url_prefix}/scenes/{scene_id}/thumb.jpg',
            'rgb':f'{url_prefix}/scenes/{scene_id}/rgb.jpg','depth':None,
            'metadata':f'{url_prefix}/scenes/{scene_id}/scene.json',
            'available':{'rgb':True,'depth':False,'segmentation':True,'detection':False,'quality':False,'pose':False}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,default=Path('playground'))
    parser.add_argument('--scene',action='append',help='dataset:recording; may be repeated')
    parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--rgb-sam-scene', help='Optional standalone RGB SAM 3 image basename')
    args=parser.parse_args(); scenes=[]
    try:
        url_prefix = args.output.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        parser.error('--output must be inside the current website directory')
    for value in args.scene or DEFAULT_SCENES:
        dataset, recording = value.split(':',1)
        scenes.append(export_scene(args.project,args.output,dataset,recording,args.frame,url_prefix))
    if args.rgb_sam_scene:
        scenes.append(export_rgb_sam_scene(args.project,args.output,args.rgb_sam_scene,url_prefix))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'manifest.json').write_text(json.dumps({'version':1,'scenes':scenes},indent=2)+'\n')
    print(f'Exported {len(scenes)} matched scenes to {args.output}')
if __name__=='__main__': main()
