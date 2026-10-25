"""Detect rectangular boxes for the exposed top layer of a LEGO stack.

Each output contains ``comparison.png`` (RGB | fused result), ``fusion.json``,
``top_mask.png`` and ``height_map.png``.  Poses are in the color optical frame
(x right, y down, z forward); a robot extrinsic is intentionally not assumed.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segment_lego_sam3_single_prompt import (
    BrickGeometrySegmenter, BrickSegmenter, add_detection_args)


def load_camera_info(recording):
    info = json.loads((recording / 'camera_info.json').read_text())
    if not info.get('depth_aligned_to_color'):
        raise ValueError(f'{recording}: depth is not aligned to color')
    intr = info.get('aligned_depth_intrinsics', info['color_intrinsics'])
    return info, intr


def xyz_image(depth_raw, scale, intr):
    z = depth_raw.astype(np.float32) * float(scale)
    v, u = np.indices(z.shape, dtype=np.float32)
    x = (u - float(intr['ppx'])) * z / float(intr['fx'])
    y = (v - float(intr['ppy'])) * z / float(intr['fy'])
    return np.dstack((x, y, z))


def fit_plane_ransac(points, rng, iterations=300, threshold=0.004):
    """Fit dominant plane to Nx3 points and orient its normal toward camera."""
    if len(points) < 100:
        raise ValueError('not enough background depth points for a support plane')
    if len(points) > 30000:
        points = points[rng.choice(len(points), 30000, replace=False)]
    best = None
    for _ in range(iterations):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        # The support surface should be roughly facing the camera.
        if abs(n[2]) < 0.7:
            continue
        d = -float(n @ p[0])
        residual = np.abs(points @ n + d)
        inliers = residual < threshold
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = score, inliers
    if best is None or best[0] < 100:
        raise ValueError('could not fit a reliable support plane')
    q = points[best[1]]
    center = q.mean(axis=0)
    _, _, vh = np.linalg.svd(q - center, full_matrices=False)
    n = vh[-1]
    d = -float(n @ center)
    if d < 0:  # positive signed distance points from table toward camera
        n, d = -n, -d
    residual = np.abs(q @ n + d)
    return n.astype(np.float32), float(d), int(len(q)), float(np.sqrt(np.mean(residual ** 2)))


def support_plane(xyz, valid, instance, args):
    occupied = instance > 0
    occupied = cv2.dilate(occupied.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    h, w = valid.shape
    border = np.zeros_like(valid)
    border[: max(20, h // 6)] = True
    border[-max(20, h // 6):] = True
    border[:, : max(20, w // 6)] = True
    border[:, -max(20, w // 6):] = True
    candidates = xyz[valid & ~occupied & border]
    if len(candidates) < 500:
        candidates = xyz[valid & ~occupied]
    return fit_plane_ransac(
        candidates, np.random.default_rng(args.seed), args.plane_iterations,
        args.plane_threshold_mm / 1000.0)


def normalize_yaw(rect):
    w, h = rect['size_wh']
    yaw = float(rect['angle_deg'])
    if w < h:
        yaw += 90.0
    return ((yaw + 90.0) % 180.0) - 90.0


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def surface_geometry(component, xyz, support_normal):
    """Return plane and metric rectangle diagnostics for one top-face component."""
    pts = xyz[component]
    center = np.median(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = vh[-1]
    if normal @ support_normal < 0:
        normal = -normal
    residual = np.abs((pts - center) @ normal)
    plane_rmse = float(np.sqrt(np.mean(residual ** 2)))
    parallel_deg = float(np.degrees(np.arccos(np.clip(normal @ support_normal, -1, 1))))

    # Principal extents in the fitted surface plane, robust to a few edge pixels.
    axes = vh[:2]
    uv = (pts - center) @ axes.T
    extents = np.percentile(uv, 98, axis=0) - np.percentile(uv, 2, axis=0)
    length, width = sorted((float(extents[0]), float(extents[1])), reverse=True)
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    component_rectangularity = float(component.sum() / max(rect[1][0] * rect[1][1], 1))
    return center, plane_rmse, parallel_deg, length, width, component_rectangularity


def cluster_supported_layer(records, args):
    """Select the highest height cluster with enough planar surface support."""
    if not records:
        return [], []
    gap = args.height_cluster_gap_mm / 1000.0
    clusters = []
    for record in sorted(records, key=lambda r: r['height_m']):
        if not clusters or record['height_m'] - clusters[-1][-1]['height_m'] > gap:
            clusters.append([record])
        else:
            clusters[-1].append(record)
    typical_area = float(np.median([r['top_surface_pixels'] for r in records]))
    minimum_support = args.min_layer_support_bricks * typical_area
    supported = [c for c in clusters if sum(r['top_surface_pixels'] for r in c) >= minimum_support]
    selected = max(supported or clusters,
                   key=lambda c: np.average([r['height_m'] for r in c],
                                            weights=[r['top_surface_pixels'] for r in c]))
    diagnostics = [{
        'height_m': round(float(np.average([r['height_m'] for r in c],
                                          weights=[r['top_surface_pixels'] for r in c])), 5),
        'instances': len(c),
        'surface_pixels': int(sum(r['top_surface_pixels'] for r in c)),
        'supported': c in supported,
        'selected': c is selected,
    } for c in clusters]
    return selected, diagnostics


def analyse_instances(rgb, depth_raw, instance, records, info, intr, args):
    xyz = xyz_image(depth_raw, info['depth_scale_m'], intr)
    valid = (depth_raw > int(info.get('depth_invalid_value', 0))) & np.isfinite(xyz[..., 2])
    n, d, plane_inliers, plane_rmse = support_plane(xyz, valid, instance, args)
    heights = xyz @ n + d
    erosion = np.ones((args.erode_px, args.erode_px), np.uint8)
    analysed = []

    for record in records:
        mask = instance == record['id']
        inner = cv2.erode(mask.astype(np.uint8), erosion) > 0
        sample = inner & valid
        if sample.sum() < args.min_depth_pixels:
            sample = mask & valid
        if sample.sum() < args.min_depth_pixels:
            analysed.append({**record, 'top_layer': False, 'depth_status': 'insufficient'})
            continue
        hs = heights[sample]
        # A robust centre prevents a few stud/neighbour pixels from promoting a
        # lower brick to the top layer.
        top_h = float(np.median(hs))
        band = sample & (np.abs(heights - top_h) <= args.surface_band_mm / 1000.0)
        component = largest_component(band)
        if component.sum() < args.min_depth_pixels:
            analysed.append({**record, 'top_layer': False,
                             'depth_status': 'fragmented_surface'})
            continue
        coverage = float(component.sum() / max(sample.sum(), 1))
        median_abs_dev = float(np.median(np.abs(hs - top_h)))
        pos, surface_rmse, parallel_deg, length_m, width_m, surface_rect = \
            surface_geometry(component, xyz, n)
        metric_shape = (args.min_brick_length_mm <= length_m * 1000 <= args.max_brick_length_mm and
                        args.min_brick_width_mm <= width_m * 1000 <= args.max_brick_width_mm)
        planar = (coverage >= args.min_surface_coverage and
                  surface_rmse * 1000 <= args.max_surface_rmse_mm and
                  parallel_deg <= args.max_surface_tilt_deg)
        rectangular = surface_rect >= args.top_min_rect
        status = 'ok' if planar and rectangular and metric_shape else (
            'nonplanar_surface' if not planar else
            'irregular_rectangle' if not rectangular else 'wrong_metric_size')
        analysed.append({
            **record, 'top_layer': False, 'depth_status': status,
            'height_m': round(top_h, 5),
            'position_camera_m': np.round(pos, 5).tolist(),
            'yaw_camera_deg': round(normalize_yaw(record['min_area_rect']), 2),
            'valid_depth_fraction': round(float((mask & valid).sum() / max(mask.sum(), 1)), 3),
            'top_surface_pixels': int(component.sum()),
            'top_surface_coverage': round(coverage, 3),
            'surface_mad_m': round(median_abs_dev, 6),
            'surface_rmse_m': round(surface_rmse, 6),
            'surface_parallel_deg': round(parallel_deg, 2),
            'surface_rectangularity': round(surface_rect, 3),
            'metric_size_mm': [round(length_m * 1000, 2), round(width_m * 1000, 2)],
        })

    usable = [r for r in analysed if r.get('depth_status') == 'ok']
    if usable:
        selected, layer_clusters = cluster_supported_layer(usable, args)
        selected_ids = {r['id'] for r in selected}
        selected_height = float(np.average(
            [r['height_m'] for r in selected],
            weights=[r['top_surface_pixels'] for r in selected]))
        for record in usable:
            record['top_layer'] = record['id'] in selected_ids
            record['height_below_selected_m'] = round(selected_height - record['height_m'], 5)
    else:
        layer_clusters = []
    plane = {
        'normal_camera': np.round(n, 7).tolist(), 'd': round(d, 7),
        'inliers': plane_inliers, 'rmse_m': round(plane_rmse, 6),
        'layer_clusters': layer_clusters,
    }
    return analysed, heights, valid, plane


def analyse_instances_baseline(rgb, depth_raw, instance, records, info, intr, args):
    """Frozen original percentile + global-maximum baseline."""
    xyz = xyz_image(depth_raw, info['depth_scale_m'], intr)
    valid = (depth_raw > int(info.get('depth_invalid_value', 0))) & np.isfinite(xyz[..., 2])
    n, d, plane_inliers, plane_rmse = support_plane(xyz, valid, instance, args)
    heights = xyz @ n + d
    erosion = np.ones((args.erode_px, args.erode_px), np.uint8)
    analysed = []
    for record in records:
        mask = instance == record['id']
        inner = cv2.erode(mask.astype(np.uint8), erosion) > 0
        sample = inner & valid
        if sample.sum() < args.min_depth_pixels:
            sample = mask & valid
        if sample.sum() < args.min_depth_pixels:
            analysed.append({**record, 'top_layer': False, 'depth_status': 'insufficient'})
            continue
        hs = heights[sample]
        top_h = float(np.percentile(hs, args.surface_percentile))
        band = sample & (np.abs(heights - top_h) <= args.baseline_surface_band_mm / 1000.0)
        if band.sum() < args.min_depth_pixels:
            band = sample
        pos = np.median(xyz[band], axis=0)
        analysed.append({
            **record, 'top_layer': False, 'depth_status': 'ok',
            'height_m': round(top_h, 5),
            'position_camera_m': np.round(pos, 5).tolist(),
            'yaw_camera_deg': round(normalize_yaw(record['min_area_rect']), 2),
            'valid_depth_fraction': round(float((mask & valid).sum() / max(mask.sum(), 1)), 3),
            'top_surface_pixels': int(band.sum()),
        })
    usable = [r for r in analysed if r.get('depth_status') == 'ok']
    if usable:
        highest = max(r['height_m'] for r in usable)
        tolerance = args.baseline_top_tolerance_mm / 1000.0
        for record in usable:
            record['top_layer'] = record['height_m'] >= highest - tolerance
            record['height_below_highest_m'] = round(highest - record['height_m'], 5)
    plane = {'normal_camera': np.round(n, 7).tolist(), 'd': round(d, 7),
             'inliers': plane_inliers, 'rmse_m': round(plane_rmse, 6)}
    return analysed, heights, valid, plane


def put_label(image, text, xy, colour):
    x, y = map(int, xy)
    cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, colour, 1, cv2.LINE_AA)


def render(rgb, records, baseline=False):
    """Draw only oriented bounding boxes for top-layer bricks."""
    result = rgb.copy()
    for record in records:
        if record.get('top_layer'):
            colour = ((20, 220, 20) if not baseline or record['quality'] == 'good'
                      else (255, 35, 35))
            rect = record['min_area_rect']
            box = cv2.boxPoints((tuple(rect['center_xy']), tuple(rect['size_wh']),
                                float(rect['angle_deg'])))
            cv2.polylines(result, [np.rint(box).astype(np.int32)], True, colour, 2,
                          cv2.LINE_AA)
            suffix = '' if not baseline else f" {'G' if record['quality'] == 'good' else 'B'}"
            put_label(result, f"#{record['id']}{suffix}", record['centroid_xy'], colour)

    count = sum(r.get('top_layer') for r in records)
    if baseline:
        good = sum(r.get('top_layer') and r['quality'] == 'good' for r in records)
        title = f'TOP LAYER: {count} bricks | GOOD {good} | BAD {count - good}'
    else:
        title = f'TOP LAYER: {count} bricks'
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    put_label(result, title, (8, 21), (255, 255, 255))
    comparison = np.hstack((rgb, result))
    return comparison, result


def discover_recordings(root, datasets):
    recordings = []
    for name in datasets:
        dataset = root / name
        if not dataset.is_dir():
            raise FileNotFoundError(dataset)
        recordings.extend((name, p) for p in sorted(dataset.iterdir()) if p.is_dir())
    return recordings


def temporal_median_depth(recording, frame, count):
    """Median of evenly sampled aligned frames; zeros do not vote."""
    paths = sorted((recording / 'depth').glob('*.png'))
    if not paths:
        raise FileNotFoundError(recording / 'depth')
    if count <= 1:
        chosen = [recording / 'depth' / f'{frame:06d}.png']
    else:
        indices = np.linspace(0, len(paths) - 1, min(count, len(paths))).round().astype(int)
        chosen = [paths[i] for i in sorted(set(indices))]
    stack = np.stack([cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in chosen])
    masked = np.ma.masked_equal(stack, 0)
    median = np.ma.median(masked, axis=0).filled(0).astype(np.uint16)
    return median, chosen


def main(strategy='improved'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='folder containing data3 and data4')
    parser.add_argument('--out', type=Path,
                        default=Path('perception/sam3_detection') /
                        ('results_rgbd_baseline_reproduced' if strategy == 'baseline'
                         else 'results_rgbd_improved_v1'))
    parser.add_argument('--datasets', nargs='+', default=['data3', 'data4'])
    parser.add_argument('--frame', type=int, default=0, help='frame index per recording')
    parser.add_argument('--max-recordings', type=int, default=0, help='0 processes all recordings')
    parser.add_argument('--plane-iterations', type=int, default=300)
    parser.add_argument('--plane-threshold-mm', type=float, default=4.0)
    parser.add_argument('--temporal-frames', type=int, default=9)
    parser.add_argument('--surface-band-mm', type=float, default=2.0)
    parser.add_argument('--min-surface-coverage', type=float, default=0.45)
    parser.add_argument('--height-cluster-gap-mm', type=float, default=3.0)
    parser.add_argument('--min-layer-support-bricks', type=float, default=0.75)
    parser.add_argument('--top-min-rect', type=float, default=0.72)
    parser.add_argument('--max-surface-rmse-mm', type=float, default=1.6)
    parser.add_argument('--max-surface-tilt-deg', type=float, default=15.0)
    parser.add_argument('--min-brick-length-mm', type=float, default=23.0)
    parser.add_argument('--max-brick-length-mm', type=float, default=43.0)
    parser.add_argument('--min-brick-width-mm', type=float, default=10.0)
    parser.add_argument('--max-brick-width-mm', type=float, default=24.0)
    parser.add_argument('--surface-percentile', type=float, default=70.0)
    parser.add_argument('--baseline-surface-band-mm', type=float, default=3.5)
    parser.add_argument('--baseline-top-tolerance-mm', type=float, default=5.0)
    parser.add_argument('--erode-px', type=int, default=7)
    parser.add_argument('--min-depth-pixels', type=int, default=40)
    parser.add_argument('--seed', type=int, default=7)
    add_detection_args(parser)
    args = parser.parse_args()

    recordings = discover_recordings(args.root, args.datasets)
    if args.max_recordings:
        recordings = recordings[:args.max_recordings]
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = strategy == 'baseline'
    segmenter = BrickSegmenter(args) if baseline else BrickGeometrySegmenter(args)
    summaries = []

    for index, (dataset, recording) in enumerate(recordings, 1):
        stem = f'{args.frame:06d}.png'
        rgb_path, depth_path = recording / 'rgb' / stem, recording / 'depth' / stem
        if not rgb_path.exists() or not depth_path.exists():
            print(f'[{index}/{len(recordings)}] skip missing pair {recording.name}/{stem}')
            continue
        print(f'[{index}/{len(recordings)}] {dataset}/{recording.name}/{stem}', flush=True)
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        if baseline:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_sources = [depth_path]
        else:
            depth, depth_sources = temporal_median_depth(
                recording, args.frame, args.temporal_frames)
        if depth is None or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
            raise ValueError(f'invalid aligned depth: {depth_path}')
        info, intr = load_camera_info(recording)
        if baseline:
            _, instance, records, rejected, offset = segmenter(rgb)
            fused, heights, valid, plane = analyse_instances_baseline(
                rgb, depth, instance, records, info, intr, args)
        else:
            instance, records, rejected = segmenter(rgb)
            fused, heights, valid, plane = analyse_instances(
                rgb, depth, instance, records, info, intr, args)
        comparison, result = render(rgb, fused, baseline)

        output = args.out / dataset / recording.name / f'{args.frame:06d}'
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(comparison).save(output / 'comparison.png')
        Image.fromarray(result).save(output / 'result.png')
        payload = {
            'dataset': dataset, 'recording': recording.name, 'frame': args.frame,
            'rgb': str(rgb_path), 'depth': str(depth_path),
            'coordinate_frame': 'color optical frame: x right, y down, z forward; metres',
            'strategy': strategy, 'support_plane': plane,
            'depth_sources': [str(p) for p in depth_sources],
            'instances': fused, 'sam_rejected': rejected,
        }
        if baseline:
            payload['colour_offset_lab'] = np.round(offset, 2).tolist()
        (output / 'fusion.json').write_text(json.dumps(payload, indent=2))
        top = [r for r in fused if r.get('top_layer')]
        (output / 'top_boxes.json').write_text(json.dumps({
            'dataset': dataset, 'recording': recording.name, 'frame': args.frame,
            'count': len(top),
            'boxes': [{
                'id': r['id'], 'box_xyxy': r['box_xyxy'],
                'oriented_box': r['min_area_rect'], 'score': r['score'],
                'height_m': r['height_m'],
                **({'top_surface_coverage': r['top_surface_coverage']} if not baseline else {}),
            } for r in top],
        }, indent=2))
        summaries.append({
            'image': f'image{len(summaries) + 1}',
            'dataset': dataset, 'recording': recording.name, 'frame': args.frame,
            'sam_instances': len(fused), 'top_layer_bricks': len(top),
            'comparison': str(output / 'comparison.png'),
        })

    (args.out / 'summary.json').write_text(json.dumps({'results': summaries}, indent=2))
    with (args.out / 'summary.csv').open('w', newline='') as stream:
        fields = ['image', 'dataset', 'recording', 'frame', 'top_layer_bricks',
                  'sam_instances', 'comparison']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f'Wrote {len(summaries)} results to {args.out}')


if __name__ == '__main__':
    main('improved')
