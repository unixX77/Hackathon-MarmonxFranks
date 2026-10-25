"""SAM3 -> aligned depth -> table-relative height -> top-layer mask (offline demo)."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def point_cloud(depth, intrinsics, scale):
    if any(abs(x) > 1e-8 for x in intrinsics.get('distortion_coeffs', [])):
        raise ValueError('This simple demo requires rectified/zero-distortion intrinsics')
    v, u = np.indices(depth.shape)
    z = depth.astype(np.float64) * scale
    return np.stack([(u - intrinsics['ppx']) * z / intrinsics['fx'],
                     (v - intrinsics['ppy']) * z / intrinsics['fy'], z], axis=-1)


def table_plane(points, threshold, iterations=500):
    """Dominant background plane, with normal facing toward the camera."""
    if len(points) < 100:
        raise ValueError('Too little valid background for table fitting')
    rng = np.random.default_rng(42)
    points = points[rng.choice(len(points), min(8000, len(points)), replace=False)]
    best = np.zeros(len(points), bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        inside = np.abs(points @ normal - a @ normal) < threshold
        if inside.sum() > best.sum():
            best = inside
    if best.mean() < 0.3 or best.sum() < 100:
        raise ValueError('No convincing table plane; use --table-roi around visible table')
    selected = points[best]
    center = np.median(selected, axis=0)
    _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
    normal = vt[-1]
    if normal @ center > 0:
        normal = -normal
    offset = -float(normal @ center)
    residual = np.abs(selected @ normal + offset)
    return normal, offset, float(np.median(residual)), float(best.mean())


def upper_surface(values, block_height, min_pixels):
    """Highest substantial local height mode, excluding small high outlier groups."""
    step = block_height / 10
    half_band = block_height / 4
    edges = np.arange(values.min() - step, values.max() + 2 * step, step)
    hist, edges = np.histogram(values, edges)
    smooth = np.convolve(hist, [1, 2, 1], mode='full')[1:-1]
    peaks = [i for i, count in enumerate(smooth)
             if count > 0 and count >= (smooth[i-1] if i else 0)
             and count >= (smooth[i+1] if i+1 < len(smooth) else 0)]
    groups = []
    for i in peaks:
        center = (edges[i] + edges[i+1]) / 2
        selected = values[np.abs(values - center) <= half_band]
        groups.append(selected)
    largest = max((len(group) for group in groups), default=0)
    required = max(min_pixels, int(np.ceil(len(values) * 0.25)), int(np.ceil(largest * 0.6)))
    supported = [group for group in groups if len(group) >= required]
    if not supported:
        return None
    return max(supported, key=lambda group: np.median(group))


def filter_top(instance, depth, calibration, args):
    intr = calibration['aligned_depth_intrinsics']
    if depth.shape != (intr['height'], intr['width']):
        raise ValueError('Depth dimensions differ from aligned_depth_intrinsics')
    points = point_cloud(depth, intr, calibration['depth_scale_m'])
    valid = (depth > 0) & np.isfinite(points).all(axis=-1)
    foreground = instance > 0
    exclusion = cv2.dilate(foreground.astype('uint8'), np.ones((9, 9), 'uint8')) > 0
    background = valid & ~exclusion
    if args.table_roi:
        x, y, w, h = args.table_roi
        roi = np.zeros(depth.shape, bool)
        roi[y:y+h, x:x+w] = True
        background &= roi
    normal, offset, residual, ratio = table_plane(points[background], args.plane_threshold_mm / 1000)
    heights = points @ normal + offset
    records = []
    for ident in np.unique(instance[foreground]):
        mask = instance == ident
        inner = cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8'),
                          iterations=args.erode) > 0 if args.erode else mask
        samples = inner & valid
        record = {'id': int(ident), 'valid_pixels': int(samples.sum()),
                  'valid_fraction': float(samples.sum() / max(int(inner.sum()), 1)), 'kept': False}
        if samples.sum() < args.min_depth_pixels or record['valid_fraction'] < args.min_valid_fraction:
            record['reason'] = 'insufficient valid interior depth'
        else:
            values = heights[samples]
            record['height_spread_m'] = float(np.percentile(values, 90) - np.percentile(values, 10))
            record['interior_median_height_m'] = float(np.median(values))
            surface = upper_surface(values, args.block_height_mm / 1000, args.min_depth_pixels)
            if surface is None:
                record['reason'] = 'no sufficiently supported upper surface'
            else:
                record['height_m'] = float(np.median(surface))
                record['surface_pixels'] = int(len(surface))
                record['surface_fraction'] = float(len(surface) / len(values))
                record['surface_spread_m'] = float(np.percentile(surface, 90) - np.percentile(surface, 10))
                record['approx_layer'] = int(np.rint(record['height_m'] / (args.block_height_mm / 1000)))
        records.append(record)
    tolerance = args.block_height_mm / 2000
    candidates = []
    for r in records:
        if r.get('height_m', 0) >= tolerance:
            candidates.append(r)
    # Find the highest narrow group supported by multiple instances. A lone
    # noisy maximum must not move the cutoff above a well-supported layer.
    minimum = min(getattr(args, 'top_min_support', 2), len(candidates))
    groups = [[r for r in candidates if low['height_m'] <= r['height_m'] <= low['height_m'] + tolerance]
              for low in candidates]
    supported = [g for g in groups if len(g) >= minimum]
    group = max(supported, key=lambda g: np.median([r['height_m'] for r in g]), default=[])
    highest = float(np.median([r['height_m'] for r in group])) if group else None
    for r in records:
        r['kept'] = highest is not None and r in candidates and abs(highest - r['height_m']) <= tolerance
        if not r['kept'] and 'reason' not in r:
            r['reason'] = 'outside supported top layer'
    kept = np.isin(instance, [r['id'] for r in records if r['kept']])
    report = {'block_height_mm': args.block_height_mm, 'top_layer_tolerance_m': tolerance,
              'top_height_m': highest, 'table_normal': normal.tolist(), 'table_offset_m': offset,
              'table_median_residual_m': residual, 'table_inlier_fraction': ratio,
              'instances': records, 'top_cluster_support': len(group),
              'top_min_support': getattr(args, 'top_min_support', 2),
              'method': 'supported upper-surface mode per mask; highest supported instance cluster',
              'limitations': 'Experimental: assumes dominant background is table; studs/occlusions/noise can bias height.'}
    return kept, report


def semantic_overlay(bgr, instance, records, ids):
    """Instances in `ids` as separate patches: filled with their colour class, with a white
    border just inside each brick so touching bricks of the same colour stay apart."""
    from segment_lego_sam3_single_prompt import DISPLAY
    out = bgr.copy()
    kernel = np.ones((3, 3), 'uint8')
    for r in records:
        if r['id'] not in ids:
            continue
        m = instance == r['id']
        out[m] = (0.3 * bgr[m] + 0.7 * DISPLAY[r['class_id']][::-1]).astype('uint8')
        border = m & ~(cv2.erode(m.astype('uint8'), kernel) > 0)
        out[border] = (255, 255, 255)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path, help='Extracted recording folder containing rgb/, depth/, camera_info.json')
    p.add_argument('--frame', default='000000')
    p.add_argument('--out', type=Path, default=Path('data/top_layer'))
    p.add_argument('--instance-mask', type=Path, help='Optional existing integer instance PNG; skip SAM3')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--top-min-support', type=int, default=2, choices=range(1, 100),
                   help='minimum masks supporting the top layer; use 1 for a single top brick')
    p.add_argument('--block-height-mm', type=float, default=10)
    p.add_argument('--table-roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    p.add_argument('--plane-threshold-mm', type=float, default=3)
    p.add_argument('--erode', type=int, default=2, help='3x3 erosion iterations')
    p.add_argument('--min-depth-pixels', type=int, default=30)
    p.add_argument('--min-valid-fraction', type=float, default=0.3)
    p.add_argument('--model', default='facebook/sam3')
    p.add_argument('--prompt', default='lego brick')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--mask-threshold', type=float, default=0.5)
    p.add_argument('--max-overlap', type=float, default=0.5)
    p.add_argument('--min-area', type=int, default=100)
    p.add_argument('--min-rect', type=float, default=0.0)
    p.add_argument('--min-rel-area', type=float, default=0.0)
    p.add_argument('--max-colour-dist', type=float, default=float('inf'))
    args = p.parse_args()
    if args.block_height_mm <= 0 or args.plane_threshold_mm <= 0 or args.erode < 0:
        p.error('Heights/thresholds must be positive and erosion nonnegative')
    bgr = cv2.imread(str(args.recording / 'rgb' / f'{args.frame}.png'))
    depth = cv2.imread(str(args.recording / 'depth' / f'{args.frame}.png'), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        p.error('Missing RGB or aligned depth frame')
    if depth.dtype != np.uint16 or depth.shape != bgr.shape[:2]:
        p.error('Expected aligned uint16 depth matching RGB dimensions')
    calibration = json.loads((args.recording / 'camera_info.json').read_text())
    semantic = None
    sam_records = []
    if args.instance_mask:
        instance = cv2.imread(str(args.instance_mask), cv2.IMREAD_UNCHANGED)
    else:
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
            from segment_lego_sam3_single_prompt import segment
        except ImportError as e:
            p.error(f'{e}. Run in your sam3 environment, or supply --instance-mask.')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        processor = Sam3Processor.from_pretrained(args.model)
        model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
        semantic, instance, sam_records, rejected, offset = segment(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), model, processor, args, device, dtype)
    if instance is None or instance.ndim != 2 or instance.shape != depth.shape:
        p.error('Instance mask must be a single-channel ID image matching RGB')
    if not np.any(instance):
        p.error('No foreground instances found; adjust SAM3 prompt/threshold')
    keep, report = filter_top(instance, depth, calibration, args)
    report.update({'recording': str(args.recording), 'frame': args.frame, 'sam3_instances': sam_records})
    output = args.out / args.recording.name / args.frame
    output.mkdir(parents=True, exist_ok=True)
    if semantic is not None:
        kept_ids = {r['id'] for r in report['instances'] if r['kept']}
        overlay = semantic_overlay(bgr, instance, sam_records, kept_ids)
        everything = semantic_overlay(bgr, instance, sam_records, {r['id'] for r in sam_records})
    else:
        overlay = bgr.copy()
        overlay[keep] = (0.45 * bgr[keep] + 0.55 * np.array([0, 255, 0])).astype('uint8')
        contours, _ = cv2.findContours(keep.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        everything = cv2.bitwise_and(bgr, bgr, mask=(instance > 0).astype('uint8') * 255)
    preview = np.hstack([bgr, everything, overlay])
    outputs = {'instances.png': instance.astype('uint16'), 'final_mask.png': keep.astype('uint8') * 255,
               'final_instances.png': np.where(keep, instance, 0).astype('uint16'),
               'rgb_masked.png': np.where(keep[..., None], bgr, 0),
               'depth_masked.png': np.where(keep, depth, 0).astype('uint16'),
               'overlay.png': overlay, 'preview.png': preview}
    if semantic is not None:
        outputs['semantic.png'] = semantic
        outputs['final_semantic.png'] = np.where(keep, semantic, 0).astype('uint8')
    for name, image in outputs.items():
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError(f'Cannot write {output / name}')
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(f"Kept {sum(r['kept'] for r in report['instances'])}/{len(report['instances'])} instances -> {output}")
    if not args.no_show and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print(f'No display; open {output / "preview.png"} instead (use --no-show to skip this check)')
    elif not args.no_show:
        cv2.namedWindow('RGB | SAM3 bricks | top layer', cv2.WINDOW_NORMAL)
        cv2.imshow('RGB | SAM3 bricks | top layer', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
