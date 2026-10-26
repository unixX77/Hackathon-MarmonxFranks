"""SAM3 -> depth top layer -> brick-size prior (offline test).

Same pipeline as top_layer_depth.py (SAM3 bricks, table plane from depth, highest
layer kept), plus a prior on the brick size: every brick is L x W x H
(default 30 x 15 x 10 mm). Each top-layer mask is measured in millimetres on the
table plane and then
  - kept as one brick when it is about L x W,
  - split into n bricks when it is about W wide and n x L long (touching bricks
    that SAM3 merged into one mask),
  - rejected as partial when it is about W wide but clearly shorter than L
    (a brick mostly hidden by the layer above), or clearly narrower than W
    (a sliver, e.g. the side face of an edge brick),
  - rejected as irregular when it fits none of these.
Width may reach --max-width-factor x W (default 1.5): an isolated raised brick's
mask includes its side walls. The top layer is the highest height supported by
--top-min-support masks; a lone mask one layer higher is still accepted when its
depth shows one clean, flat L x W brick (--no-lone-top-brick disables this).
Every brick gets a fitted L x W footprint: centre in the camera frame, yaw on the
table and layer index, which is what a grasp planner needs.

Each final brick is then classified by colour (good / bad follows from the colour):
  - white balance from the table pixels found by the depth plane fit, in linear RGB,
  - colour measured inside the fitted footprint and the brick's own mask, away from
    the edges, without shadows (darkest 15%) and highlights (brightest 10%),
  - nearest reference colour (PALETTE in segment_lego_sam3_single_prompt.py) after a
    bounded Lab offset estimated jointly over the bricks of the frame (lightness only
    when there are few bricks), with a confidence from distance and margin.

Outputs in <out>/<recording>/<frame>/:
  final_instances.png  uint16 brick ids after splitting (0 = background)
  final_semantic.png   uint8 colour class per pixel (see segment_lego_sam3_single_prompt.py)
  overlay.png          final bricks: colour class fill, green outline = good, red = bad,
                       yellow dot = fitted brick centre
  preview.png          all stages: input | height above table | all SAM3 bricks /
                       depth top layer | size prior with footprints | colour and good / bad
  bricks.json          per brick: id, status, colour, quality, colour confidence, measured size, fitted pose,
                       3D box (centre, size, rotation, 8 corners) in the camera frame
  poses.csv            one row per brick: box centre, top-face centre, yaw, size and the
                       8 box corners (top face 0-3, bottom face 4-7) in the camera frame (m)
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from top_layer_depth import filter_top, point_cloud, semantic_overlay, table_plane

LAB_WEIGHTS = np.array([0.45, 1.0, 1.0])     # lightness varies with exposure more than chroma
OFFSET_LIMIT = np.array([60.0, 18.0, 18.0])  # max per-frame Lab offset (L, a, b)
MIN_BRICKS_FOR_CHROMA_OFFSET = 5             # with fewer bricks only lightness is corrected


def table_frame(normal):
    """Orthonormal in-plane axes e1, e2 for a table normal."""
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def mask_on_plane(mask, K_inv, normal, offset, height, e1, e2):
    """Intersect the camera rays of the mask pixels with the plane `height` above the table.
    Returns pixel coordinates and their in-plane coordinates in mm."""
    v, u = np.nonzero(mask)
    rays = K_inv @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)
    points = rays * ((height - offset) / (normal @ rays))
    return u, v, np.stack([e1 @ points, e2 @ points], 1) * 1000


def to_camera(q_mm, normal, offset, height, e1, e2):
    """In-plane mm coordinates at `height` above the table -> 3D points in the camera frame (m)."""
    q = np.atleast_2d(np.asarray(q_mm, np.float64)) / 1000
    return q[:, :1] * e1 + q[:, 1:] * e2 + (height - offset) * normal


def to_pixels(q_mm, K, normal, offset, height, e1, e2):
    """In-plane mm coordinates at `height` above the table -> image pixels."""
    p = (K @ to_camera(q_mm, normal, offset, height, e1, e2).T).T
    return p[:, :2] / p[:, 2:]


def geometry_mask_filter(depth, calibration, args):
    """Validate raw SAM masks in mm before score-based overlap suppression."""
    points = point_cloud(depth, calibration['aligned_depth_intrinsics'], calibration['depth_scale_m'])
    K_inv = np.linalg.inv(np.asarray(calibration['aligned_depth_intrinsics']['K'], float))

    def select(masks, scores):
        foreground = masks.any(axis=0)
        excluded = cv2.dilate(foreground.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        background = (depth > 0) & ~excluded
        if args.table_roi:
            x, y, w, h = args.table_roi
            roi = np.zeros(depth.shape, bool)
            roi[y:y+h, x:x+w] = True
            background &= roi
        normal, offset, _, _ = table_plane(points[background], args.plane_threshold_mm / 1000)
        e1, e2 = table_frame(normal)
        heights = points @ normal + offset
        accepted, rejected = [], []
        for i, mask in enumerate(masks):
            valid = mask & (depth > 0)
            reason = None
            if valid.sum() < args.min_depth_pixels:
                reason = 'insufficient depth for geometry'
            else:
                height = float(np.median(heights[valid]))
                _, _, q = mask_on_plane(mask, K_inv, normal, offset, height, e1, e2)
                _, (w, h), _ = cv2.minAreaRect(q.astype(np.float32))
                # Long, one-brick-wide runs can still be split by the size prior. The limit is
                # halfway to two bricks wide: a raised brick's mask includes its side walls.
                if min(w, h) > args.brick_size_mm[1] * args.max_width_factor:
                    reason = 'too wide: assembly or multi-row mask'
            if reason:
                rejected.append({'score': float(scores[i]), 'reason': reason,
                                 'area_px': int(mask.sum())})
            else:
                accepted.append(i)
        return np.asarray(accepted, dtype=int), rejected
    return select


def apply_size_prior(instance, kept_records, sam_records, calibration, report, args):
    """Measure, split and fit every kept top-layer mask; return the brick id image and brick records."""
    L, W, H = args.brick_size_mm
    tol = args.size_tol
    K = np.array(calibration['aligned_depth_intrinsics']['K'], np.float64)
    K_inv = np.linalg.inv(K)
    normal, offset = np.array(report['table_normal']), report['table_offset_m']
    e1, e2 = table_frame(normal)
    colours = {r['id']: r for r in sam_records}

    bricks_img = np.zeros(instance.shape, np.uint16)
    bricks = []
    for r in kept_records:
        h = r['height_m']
        u, v, q = mask_on_plane(instance == r['id'], K_inv, normal, offset, h, e1, e2)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(q.astype(np.float32))
        length, width = max(rw, rh), min(rw, rh)
        long_deg = angle if rw >= rh else angle + 90  # direction of the long side
        axis = np.array([np.cos(np.radians(long_deg)), np.sin(np.radians(long_deg))])
        n = int(np.rint(length / L))

        width_ok = (1 - tol) * W <= width <= args.max_width_factor * W
        if width_ok and n >= 1 and abs(length - n * L) <= tol * L:
            status = 'brick' if n == 1 else 'split'
        elif (width_ok and length < (1 - tol) * L) or width < (1 - tol) * W:
            status, n = 'partial', 0
        else:
            status, n = 'irregular', 0

        base = {'source_instance': r['id'], 'status': status, 'height_m': round(h, 4),
                'approx_layer': r.get('approx_layer'), 'measured_mm': [round(length, 1), round(width, 1)]}
        if r['id'] in colours:
            c = colours[r['id']]
            base.update(colour=c['colour'], class_id=c['class_id'], quality=c['quality'])
        if n == 0:
            bricks.append({**base, 'id': None})
            continue

        # cut along the long axis into n equal parts (n = 1 keeps the mask whole)
        t = (q - [cx, cy]) @ axis
        part = np.clip(((t + length / 2) / (length / n)).astype(int), 0, n - 1)
        for k in range(n):
            sel = part == k
            if not sel.any():
                continue
            ident = len([b for b in bricks if b['id']]) + 1
            bricks_img[v[sel], u[sel]] = ident
            centre_q = np.array([cx, cy]) + axis * (-length / 2 + (k + 0.5) * length / n)
            # fitted footprint: L x W for bricks, the measured rectangle for irregular masks
            fl, fw = (L, W) if status != 'irregular' else (length, width)
            corners = [centre_q + sx * axis * fl / 2 + sy * np.array([-axis[1], axis[0]]) * fw / 2
                       for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            # 3D box in the camera frame: top face at the measured height, bottom one brick height lower
            bottom = h - H / 1000
            top_corners = to_camera(corners, normal, offset, h, e1, e2)
            bottom_corners = to_camera(corners, normal, offset, bottom, e1, e2)
            long_cam = axis[0] * e1 + axis[1] * e2
            rotation = np.stack([long_cam, np.cross(normal, long_cam), normal], 1)  # columns: long, short, up
            bricks.append({**base, 'id': ident,
                           'centre_cam_m': np.round(to_camera(centre_q, normal, offset, h, e1, e2)[0], 4).tolist(),
                           'box_centre_cam_m': np.round(
                               to_camera(centre_q, normal, offset, (h + bottom) / 2, e1, e2)[0], 4).tolist(),
                           'box_size_m': [round(fl / 1000, 4), round(fw / 1000, 4), round(H / 1000, 4)],
                           'box_rotation_cam': np.round(rotation, 4).tolist(),
                           'box_corners_cam_m': np.round(np.vstack([top_corners, bottom_corners]), 4).tolist(),
                           'yaw_deg_table': round(float(long_deg) % 180, 1),
                           'footprint_px': np.round(to_pixels(corners, K, normal, offset, h, e1, e2), 1).tolist()})
    return bricks_img, bricks


def write_poses_csv(path, bricks):
    """One row per brick: box centre, top-face centre, yaw and the 8 box corners, in the camera frame (m).
    Corners 0-3 are the top face, 4-7 the bottom face below them, in the same order."""
    header = ['id', 'status', 'colour', 'quality', 'cx', 'cy', 'cz', 'top_cx', 'top_cy', 'top_cz',
              'yaw_deg_table', 'length_m', 'width_m', 'height_m']
    header += [f'c{i}_{a}' for i in range(8) for a in 'xyz']
    rows = []
    for b in bricks:
        rows.append([b['id'], b['status'], b.get('colour', ''), b.get('quality', ''),
                     *b['box_centre_cam_m'], *b['centre_cam_m'], b['yaw_deg_table'], *b['box_size_m'],
                     *np.ravel(b['box_corners_cam_m'])])
    path.write_text('\n'.join(','.join(str(v) for v in row) for row in [header] + rows) + '\n')


def height_and_table(depth, calibration, report, instance, args):
    """Height above the table per pixel (m, NaN without depth) and the table pixels."""
    points = point_cloud(depth, calibration['aligned_depth_intrinsics'], calibration['depth_scale_m'])
    height = points @ np.array(report['table_normal']) + report['table_offset_m']
    height[depth == 0] = np.nan
    near_bricks = cv2.dilate((instance > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    table = (depth > 0) & (np.abs(np.nan_to_num(height, nan=1.0)) < args.plane_threshold_mm / 1000) & ~near_bricks
    return height, table


def white_balance(bgr, table, max_gain):
    """White-balanced image: per-channel gains in linear RGB that make the table grey (brightness kept)."""
    c = bgr.astype(np.float64) / 255
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    if table.sum() < 500:
        return bgr, np.ones(3)
    ref = np.median(linear[table], axis=0)
    gains = np.clip(ref.mean() / np.maximum(ref, 1e-6), 1 / max_gain, max_gain)
    out = np.clip(linear * gains, 0, 1)
    out = np.where(out <= 0.0031308, 12.92 * out, 1.055 * out ** (1 / 2.4) - 0.055)
    return np.round(out * 255).astype(np.uint8), gains


def classify_bricks(bgr, bricks_img, bricks, table, args):
    """Colour class, good / bad and confidence for every final brick (adds fields to `bricks`)."""
    from segment_lego_sam3_single_prompt import NAMES, PALETTE
    balanced, gains = white_balance(bgr, table, args.max_wb_gain)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB).astype(np.float64)
    refs = np.array([ref for n in NAMES for ref in PALETTE[n][0]], np.float64)
    ref_class = [n for n in NAMES for _ in PALETTE[n][0]]

    final = [b for b in bricks if b['id']]
    feats = []
    for b in final:
        box = np.zeros(bricks_img.shape, np.uint8)
        cv2.fillConvexPoly(box, np.round(np.array(b['footprint_px'])).astype(np.int32), 1)
        region = (box > 0) & (bricks_img == b['id'])
        side = min(np.linalg.norm(np.diff(np.array(b['footprint_px'])[:3], axis=0), axis=1))
        k = 2 * max(1, int(round(side * args.colour_erosion))) + 1
        inner = cv2.erode(region.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        px = lab[inner if inner.sum() >= 30 else (region if region.any() else bricks_img == b['id'])]
        lo, hi = np.percentile(px[:, 0], [15, 90])
        px = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
        feats.append(np.median(px, axis=0))
    feats = np.array(feats).reshape(-1, 3)

    # bounded offset, estimated jointly over all bricks of the frame
    limit = OFFSET_LIMIT if len(feats) >= MIN_BRICKS_FOR_CHROMA_OFFSET else OFFSET_LIMIT * [1, 0, 0]
    offset = np.zeros(3)
    for _ in range(6):
        if not len(feats):
            break
        d = np.linalg.norm(((feats - offset)[:, None] - refs[None]) * LAB_WEIGHTS, axis=2)
        offset = np.clip(np.median(feats - refs[d.argmin(1)], axis=0), -limit, limit)

    for b, f in zip(final, feats):
        dist = np.linalg.norm((refs - (f - offset)) * LAB_WEIGHTS, axis=1)
        per_class = {n: float(min(dist[i] for i, c in enumerate(ref_class) if c == n)) for n in NAMES}
        colour, runner_up = sorted(per_class, key=per_class.get)[:2]
        margin = per_class[runner_up] - per_class[colour]
        confident = per_class[colour] <= args.max_colour_dist_conf and margin >= args.min_colour_margin
        b.update(colour=colour, class_id=NAMES.index(colour) + 1, quality=PALETTE[colour][2],
                 colour_confidence='high' if confident else 'low',
                 colour_distance=round(per_class[colour], 2), colour_margin=round(margin, 2),
                 colour_runner_up=runner_up, lab=np.round(f, 1).tolist())
    return {'wb_gains_linear_rgb': np.round(gains, 4).tolist(), 'lab_offset': np.round(offset, 2).tolist(),
            'table_pixels': int(table.sum())}


def instance_view(bgr, instance):
    """Every instance in its own colour with a white boundary."""
    out = bgr.copy()
    kernel = np.ones((3, 3), np.uint8)
    for ident in np.unique(instance[instance > 0]):
        m = instance == ident
        hue = np.uint8([[[(int(ident) * 47) % 180, 200, 255]]])
        out[m] = (0.35 * bgr[m] + 0.65 * cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0, 0]).astype(np.uint8)
        out[m & ~(cv2.erode(m.astype(np.uint8), kernel) > 0)] = 255
    return out


def height_view(height, top_height):
    """Height above the table as a colour map (0 .. top layer + 1 cm), black without depth."""
    top = (top_height or 0.05) + 0.01
    v = np.clip(np.nan_to_num(height, nan=0) / top, 0, 1)
    out = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[np.isnan(height)] = 0
    return out


def final_view(bgr, bricks_img, bricks):
    """Colour-class patches, green outline = good, red = bad, yellow dot = fitted brick centre."""
    shown = [b for b in bricks if b['id'] and 'class_id' in b]
    out = semantic_overlay(bgr, bricks_img, shown, {b['id'] for b in shown})
    for b in shown:
        contours, _ = cv2.findContours((bricks_img == b['id']).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (60, 210, 60) if b['quality'] == 'good' else (40, 40, 240), 2)
        cx, cy = np.mean(b['footprint_px'], axis=0)
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 230, 255), -1, cv2.LINE_AA)
    return out


def titled(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_frame(bgr, depth, calibration, model, processor, args, device, dtype):
    """Shared offline/live analysis; returns images, metadata and final poses."""
    from segment_lego_sam3_single_prompt import segment
    semantic, instance, sam_records, rejected, _ = segment(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), model, processor, args, device, dtype,
        mask_filter=geometry_mask_filter(depth, calibration, args))
    if not np.any(instance):
        raise ValueError('No foreground instances found; adjust SAM3 prompt/threshold')

    keep, report = filter_top(instance, depth, calibration, args)
    kept = [r for r in report['instances'] if r['kept']]
    bricks_img, bricks = apply_size_prior(instance, kept, sam_records, calibration, report, args)

    height, table = height_and_table(depth, calibration, report, instance, args)
    colour_info = classify_bricks(bgr, bricks_img, bricks, table, args)
    final = [b for b in bricks if b['id']]

    final_semantic = np.zeros(bricks_img.shape, np.uint8)
    for b in final:
        final_semantic[bricks_img == b['id']] = b['class_id']
    overlay = final_view(bgr, bricks_img, bricks)

    # all stages, left to right and top to bottom
    table_view = bgr.copy()
    table_view[table] = (0.5 * table_view[table] + [0, 90, 0]).astype(np.uint8)
    kept_ids = np.isin(instance, [r['id'] for r in kept])
    prior = instance_view(bgr, bricks_img)
    for b in final:
        cv2.polylines(prior, [np.round(np.array(b['footprint_px'])).astype(np.int32)], True,
                      (0, 255, 255) if b['status'] != 'irregular' else (255, 0, 255), 1, cv2.LINE_AA)
    n_good = sum(b['quality'] == 'good' for b in final)
    counts = {s: sum(b['status'] == s for b in bricks) for s in ('brick', 'split', 'partial', 'irregular')}
    stages = [
        titled(table_view, '1 input (green: table used for plane + white balance)'),
        titled(height_view(height, report['top_height_m']), '2 height above table (depth)'),
        titled(instance_view(bgr, instance), f'3 SAM3 "{args.prompt}": {len(sam_records)} masks'),
        titled(instance_view(bgr, np.where(kept_ids, instance, 0)), f'4 depth top layer: {len(kept)} masks'),
        titled(prior, f'5 size prior: {len(final)} bricks ({counts["split"]} split, {counts["partial"] + counts["irregular"]} rejected)'),
        titled(overlay, f'6 colour: {n_good} good, {len(final) - n_good} bad'),
    ]
    preview = np.vstack([np.hstack(stages[:3]), np.hstack(stages[3:])])

    images = {'final_instances.png': bricks_img, 'final_semantic.png': final_semantic,
              'overlay.png': overlay, 'preview.png': preview}
    metadata = {
        'recording': str(args.recording), 'frame': args.frame, 'brick_size_mm': args.brick_size_mm,
        'size_tol': args.size_tol, 'top_height_m': report['top_height_m'],
        'table_normal': report['table_normal'], 'table_offset_m': report['table_offset_m'],
        'colour': colour_info, 'status_counts': counts, 'sam_rejected': rejected, 'depth_report': report,
        'good': n_good, 'bad': len(final) - n_good,
        'low_confidence_colour': sum(b['colour_confidence'] == 'low' for b in final),
        'bricks': bricks}
    return images, metadata, final


def save_result(output, images, metadata, final):
    output.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError(f'Cannot write {output / name}')
    (output / 'bricks.json').write_text(json.dumps(metadata, indent=2))
    write_poses_csv(output / 'poses.csv', final)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('recording', type=Path, help='Extracted recording folder containing rgb/, depth/, camera_info.json')
    p.add_argument('--frame', default='000000')
    p.add_argument('--out', type=Path, default=Path('data/top_layer_bricks'))
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--brick-size-mm', nargs=3, type=float, default=[30, 15, 10], metavar=('L', 'W', 'H'),
                   help='brick length, width, height (H is also the layer height)')
    p.add_argument('--top-min-support', type=int, default=2, choices=range(1, 100),
                   help='minimum masks supporting the top layer; use 1 for a single top brick')
    p.add_argument('--lone-top-brick', action=argparse.BooleanOptionalAction, default=True,
                   help='accept a single clean brick one layer above the supported top (last brick of a layer)')
    p.add_argument('--size-tol', type=float, default=0.3, help='relative tolerance on length and width')
    p.add_argument('--max-width-factor', type=float, default=1.5,
                   help='masks wider than this x brick width are multi-row assemblies')
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
    p.add_argument('--max-wb-gain', type=float, default=1.25, help='limit on the white-balance gain per channel')
    p.add_argument('--colour-erosion', type=float, default=0.12,
                   help='fraction of the short brick side ignored at the edges when measuring colour')
    p.add_argument('--max-colour-dist-conf', type=float, default=35.0,
                   help='colour distance above which the colour is low-confidence')
    p.add_argument('--min-colour-margin', type=float, default=4.0,
                   help='distance margin to the second-best colour below which the colour is low-confidence')
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.block_height_mm = args.brick_size_mm[2]  # used by filter_top

    bgr = cv2.imread(str(args.recording / 'rgb' / f'{args.frame}.png'))
    depth = cv2.imread(str(args.recording / 'depth' / f'{args.frame}.png'), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        p.error('Missing RGB or aligned depth frame')
    if depth.dtype != np.uint16 or depth.shape != bgr.shape[:2]:
        p.error('Expected aligned uint16 depth matching RGB dimensions')
    calibration = json.loads((args.recording / 'camera_info.json').read_text())

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
        from segment_lego_sam3_single_prompt import segment
    except ImportError as e:
        p.error(f'{e}. Run in your sam3 environment.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    images, metadata, final = process_frame(bgr, depth, calibration, model, processor, args, device, dtype)
    output = args.out / args.recording.name / args.frame
    save_result(output, images, metadata, final)
    preview = images['preview.png']
    print(f'{len(final)} bricks -> {output}')

    if not args.no_show and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print(f'No display; open {output / "preview.png"} instead (use --no-show to skip this check)')
    elif not args.no_show:
        title = 'top-layer bricks: all stages'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
