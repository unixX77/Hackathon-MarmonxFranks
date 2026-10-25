"""Top-plane-first LEGO detection with SAM-seeded lattice reconstruction.

Unlike the baseline, SAM instances are evidence rather than a hard gate.  The
algorithm finds the globally supported upper depth plane, learns the projected
2x4-brick dimensions and lattice from reliable masks, then tests every lattice
cell against the top-plane pixels.  This recovers missed instances and splits
SAM masks spanning multiple bricks.  It intentionally does no quality/colour
classification.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segment_lego_sam3_single_prompt import BrickGeometrySegmenter, add_detection_args
from segment_lego_rgbd import (discover_recordings, load_camera_info, normalize_yaw,
                               support_plane, temporal_median_depth, xyz_image)


def rectangle_mask(shape, center, size, angle):
    mask = np.zeros(shape, np.uint8)
    box = cv2.boxPoints((tuple(center), tuple(size), float(angle)))
    cv2.fillConvexPoly(mask, np.rint(box).astype(np.int32), 1)
    return mask.astype(bool), box


def supported_height_candidates(heights, valid, roi, args):
    values = heights[valid & roi & (heights > args.min_object_height_mm / 1000.0)]
    if len(values) < args.min_top_plane_pixels:
        raise ValueError('not enough elevated depth pixels in the stack ROI')
    lo, hi = float(values.min()), float(values.max())
    step = args.height_bin_mm / 1000.0
    edges = np.arange(lo, hi + 2 * step, step)
    hist, edges = np.histogram(values, edges)
    smooth = np.convolve(hist, np.array([1, 2, 3, 2, 1]), mode='same')
    centres = (edges[:-1] + edges[1:]) / 2
    half_band = args.top_plane_band_mm / 1000.0
    candidates = []
    for index in np.argsort(-centres):
        centre = centres[index]
        support = int(np.count_nonzero(np.abs(values - centre) <= half_band))
        left = smooth[index - 1] if index else -1
        right = smooth[index + 1] if index + 1 < len(smooth) else -1
        if smooth[index] >= left and smooth[index] >= right and support >= args.min_top_plane_pixels:
            selected = values[np.abs(values - centre) <= half_band]
            candidates.append((float(np.median(selected)), int(len(selected))))
    if not candidates:
        centre = centres[int(np.argmax(smooth))]
        selected = values[np.abs(values - centre) <= half_band]
        candidates.append((float(np.median(selected)), int(len(selected))))
    return candidates


def axial_mean_degrees(angles):
    radians = np.deg2rad(np.asarray(angles) * 2.0)
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(),
                                      np.cos(radians).mean())) / 2.0)


def angle_distance(a, b):
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def lattice_phase(values, pitch):
    """Phase that best snaps observed coordinates to a regular pitch."""
    values = np.asarray(values, np.float32)
    candidates = np.linspace(0, pitch, 200, endpoint=False)
    errors = []
    for phase in candidates:
        delta = np.abs(((values - phase + pitch / 2) % pitch) - pitch / 2)
        errors.append(np.median(delta))
    return float(candidates[int(np.argmin(errors))])


def projected_bounds(mask, axis):
    y, x = np.nonzero(mask)
    points = np.column_stack((x, y))
    values = points @ axis
    return float(values.min()), float(values.max())


def reconstruct_top_boxes(rgb, depth, instance, records, info, intr, args,
                          size_calibration=None):
    xyz = xyz_image(depth, info['depth_scale_m'], intr)
    valid = (depth > int(info.get('depth_invalid_value', 0))) & np.isfinite(xyz[..., 2])
    normal, offset, plane_inliers, plane_rmse = support_plane(xyz, valid, instance, args)
    heights = xyz @ normal + offset

    occupied = instance > 0
    if not occupied.any():
        return [], {'error': 'no SAM instances'}
    hull_points = np.column_stack(np.nonzero(occupied))[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(hull_points)
    roi = np.zeros_like(occupied, np.uint8)
    cv2.fillConvexPoly(roi, hull, 1)
    roi = cv2.dilate(roi, np.ones((9, 9), np.uint8)) > 0

    geometric, plane_evidence = [], []
    for record in records:
        mask = instance == record['id']
        rect = record['min_area_rect']
        width, height = map(float, rect['size_wh'])
        long_px, short_px = max(width, height), min(width, height)
        ratio = long_px / max(short_px, 1)
        common = {**record, 'long_px': long_px, 'short_px': short_px,
                  'long_angle_deg': normalize_yaw(rect), 'mask': mask}
        if args.seed_min_aspect <= ratio <= args.seed_max_aspect:
            geometric.append(common)
        if (record['rectangularity'] >= args.merged_seed_min_rect and
                args.seed_min_aspect <= ratio <= args.merged_seed_max_aspect):
            plane_evidence.append(common)
    if len(geometric) < args.min_lattice_seeds:
        raise ValueError(f'only {len(geometric)} geometric lattice seeds')

    top = top_height = top_support = None
    top_seed_count = 0
    for candidate_height, candidate_support in supported_height_candidates(
            heights, valid, roi, args):
        candidate = valid & roi & (np.abs(heights - candidate_height) <=
                                   args.top_plane_band_mm / 1000.0)
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((args.close_px, args.close_px), np.uint8)) > 0
        count = 0
        for seed in plane_evidence:
            inner = cv2.erode(seed['mask'].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if (inner & candidate).sum() / max(inner.sum(), 1) >= args.seed_min_coverage:
                count += 1
        if count >= args.min_top_plane_seeds:
            top, top_height, top_support, top_seed_count = (
                candidate, candidate_height, candidate_support, count)
            break
    if top is None:
        raise ValueError('no supported height peak overlaps a LEGO-shaped seed')

    # Use only seeds on the selected physical plane.  Successive LEGO layers
    # alternate by 90 degrees; mixing lower-layer masks can cancel or flip the
    # orientation precisely when only a few upper bricks remain.
    evidence = []
    for seed in plane_evidence:
        inner = cv2.erode(seed['mask'].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        coverage = float((inner & top).sum() / max(inner.sum(), 1))
        if coverage >= args.seed_min_coverage:
            evidence.append({**seed, 'top_coverage': coverage})
    angle = axial_mean_degrees([x['long_angle_deg'] for x in evidence])
    target_size = size_calibration
    oriented = [x for x in geometric
                if angle_distance(x['long_angle_deg'], angle) <= args.orientation_tolerance_deg]
    if target_size:
        target_long, target_short = target_size
        oriented = [x for x in oriented
                    if abs(x['long_px'] - target_long) / target_long <= args.calibration_size_tolerance
                    and abs(x['short_px'] - target_short) / target_short <= args.calibration_size_tolerance]
    if not oriented:
        raise ValueError('no normal-size seed agrees with the selected top-plane orientation')
    if target_size:
        brick_long, brick_short = map(float, target_size)
    else:
        brick_long = float(np.median([x['long_px'] for x in oriented]))
        brick_short = float(np.median([x['short_px'] for x in oriented]))
    radians = np.deg2rad(angle)
    along = np.array([np.cos(radians), np.sin(radians)])
    across = np.array([-np.sin(radians), np.cos(radians)])
    centres = np.array([x['centroid_xy'] for x in oriented], np.float32)

    # In this interlocked stack, centres occupy a square LEGO lattice whose
    # pitch is one projected brick length in both directions.
    pitch = brick_long
    along_values, across_values = centres @ along, centres @ across
    phase_along = lattice_phase(along_values, pitch)
    phase_across = lattice_phase(across_values, pitch)
    amin, amax = projected_bounds(top, along)
    cmin, cmax = projected_bounds(top, across)
    ia0 = int(np.floor((amin - phase_along) / pitch)) - 1
    ia1 = int(np.ceil((amax - phase_along) / pitch)) + 1
    ic0 = int(np.floor((cmin - phase_across) / pitch)) - 1
    ic1 = int(np.ceil((cmax - phase_across) / pitch)) + 1

    boxes = []
    for ia in range(ia0, ia1 + 1):
        for ic in range(ic0, ic1 + 1):
            pa = phase_along + ia * pitch
            pc = phase_across + ic * pitch
            center = along * pa + across * pc
            if not (0 <= center[0] < rgb.shape[1] and 0 <= center[1] < rgb.shape[0]):
                continue
            # cv2's angle describes the first size axis.  The long axis is angle.
            rect_mask, corners = rectangle_mask(top.shape, center, (brick_long, brick_short), angle)
            area = int(rect_mask.sum())
            coverage = float((rect_mask & top).sum() / max(area, 1))
            sam_overlap = float((rect_mask & occupied).sum() / max(area, 1))
            if coverage < args.candidate_min_coverage:
                continue
            pixels = rect_mask & top
            points = xyz[pixels]
            position = np.median(points, axis=0) if len(points) else np.zeros(3)
            boxes.append({
                'id': len(boxes) + 1,
                'center_xy': np.round(center, 1).tolist(),
                'size_wh': [round(brick_long, 1), round(brick_short, 1)],
                'angle_deg': round(angle, 2),
                'corners_xy': np.round(corners, 1).tolist(),
                'top_coverage': round(coverage, 3),
                'sam_overlap': round(sam_overlap, 3),
                'position_camera_m': np.round(position, 5).tolist(),
                'source': ('sam+lattice' if sam_overlap >= args.sam_overlap_threshold
                           else 'depth+lattice'),
            })

    diagnostics = {
        'support_plane': {'normal_camera': np.round(normal, 7).tolist(),
                          'd': round(offset, 7), 'inliers': plane_inliers,
                          'rmse_m': round(plane_rmse, 6)},
        'top_height_m': round(top_height, 6),
        'top_plane_pixels': top_support,
        'lattice': {'brick_long_px': round(brick_long, 2),
                    'brick_short_px': round(brick_short, 2),
                    'pitch_px': round(pitch, 2), 'angle_deg': round(angle, 2),
                    'geometric_seeds': len(geometric),
                    'oriented_seeds': len(oriented),
                    'top_plane_seeds': top_seed_count},
    }
    return boxes, diagnostics


def render(rgb, boxes):
    result = rgb.copy()
    for box in boxes:
        corners = np.rint(box['corners_xy']).astype(np.int32)
        colour = (20, 220, 20) if box['source'] == 'sam+lattice' else (20, 220, 255)
        cv2.polylines(result, [corners], True, colour, 2, cv2.LINE_AA)
        x, y = map(int, box['center_xy'])
        cv2.putText(result, f"#{box['id']}", (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, f"#{box['id']}", (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, colour, 1, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(result, f'TOP PLANE: {len(boxes)} bricks', (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return result, np.hstack((rgb, result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results_rgbd_topplane_v2'))
    parser.add_argument('--datasets', nargs='+', default=['data3', 'data4'])
    parser.add_argument('--recordings', nargs='+', help='optional recording names to process')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--max-recordings', type=int, default=0)
    parser.add_argument('--temporal-frames', type=int, default=9)
    parser.add_argument('--plane-iterations', type=int, default=300)
    parser.add_argument('--plane-threshold-mm', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--min-object-height-mm', type=float, default=20.0)
    parser.add_argument('--height-bin-mm', type=float, default=1.0)
    parser.add_argument('--top-plane-band-mm', type=float, default=2.5)
    parser.add_argument('--min-top-plane-pixels', type=int, default=500)
    parser.add_argument('--close-px', type=int, default=5)
    parser.add_argument('--seed-min-coverage', type=float, default=.55)
    parser.add_argument('--seed-min-aspect', type=float, default=1.55)
    parser.add_argument('--seed-max-aspect', type=float, default=2.5)
    parser.add_argument('--merged-seed-max-aspect', type=float, default=8.0)
    parser.add_argument('--merged-seed-min-rect', type=float, default=.75)
    parser.add_argument('--orientation-tolerance-deg', type=float, default=15.0)
    parser.add_argument('--calibration-size-tolerance', type=float, default=.3)
    parser.add_argument('--brick-long-px', type=float)
    parser.add_argument('--brick-short-px', type=float)
    parser.add_argument('--min-lattice-seeds', type=int, default=4)
    parser.add_argument('--min-top-plane-seeds', type=int, default=1)
    parser.add_argument('--candidate-min-coverage', type=float, default=.55)
    parser.add_argument('--sam-overlap-threshold', type=float, default=.35)
    add_detection_args(parser)
    args = parser.parse_args()

    recordings = discover_recordings(args.root, args.datasets)
    if args.recordings:
        wanted = set(args.recordings)
        recordings = [(dataset, recording) for dataset, recording in recordings
                      if recording.name in wanted]
    if args.max_recordings:
        recordings = recordings[:args.max_recordings]
    args.out.mkdir(parents=True, exist_ok=True)
    segmenter = BrickGeometrySegmenter(args)
    calibrations = {}
    if args.brick_long_px and args.brick_short_px:
        calibrations = {dataset: (args.brick_long_px, args.brick_short_px)
                        for dataset in args.datasets}
    summary = []
    for index, (dataset, recording) in enumerate(recordings, 1):
        rgb_path = recording / 'rgb' / f'{args.frame:06d}.png'
        if not rgb_path.exists():
            continue
        print(f'[{index}/{len(recordings)}] {dataset}/{recording.name}', flush=True)
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        depth, depth_sources = temporal_median_depth(recording, args.frame, args.temporal_frames)
        info, intr = load_camera_info(recording)
        instance, records, rejected = segmenter(rgb)
        boxes, diagnostics = reconstruct_top_boxes(
            rgb, depth, instance, records, info, intr, args,
            calibrations.get(dataset))
        if dataset not in calibrations:
            lattice = diagnostics['lattice']
            calibrations[dataset] = (lattice['brick_long_px'], lattice['brick_short_px'])
        result, comparison = render(rgb, boxes)
        output = args.out / dataset / recording.name / f'{args.frame:06d}'
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(output / 'result.png')
        Image.fromarray(comparison).save(output / 'comparison.png')
        payload = {'strategy': 'top-plane-first-v2', 'dataset': dataset,
                   'recording': recording.name, 'frame': args.frame,
                   'rgb': str(rgb_path),
                   'depth_sources': [str(p) for p in depth_sources],
                   **diagnostics, 'boxes': boxes, 'sam_rejected': rejected}
        (output / 'top_boxes.json').write_text(json.dumps(payload, indent=2))
        summary.append({'dataset': dataset, 'recording': recording.name,
                        'frame': args.frame, 'sam_instances': len(records),
                        'top_layer_bricks': len(boxes),
                        'comparison': str(output / 'comparison.png')})
    (args.out / 'summary.json').write_text(json.dumps({'results': summary}, indent=2))
    with (args.out / 'summary.csv').open('w', newline='') as stream:
        fields = ['dataset', 'recording', 'frame', 'sam_instances',
                  'top_layer_bricks', 'comparison']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f'Wrote {len(summary)} results to {args.out}')


if __name__ == '__main__':
    main()
