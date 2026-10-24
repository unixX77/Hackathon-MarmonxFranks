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
  - kept but flagged irregular when it fits none of these.
Every brick gets a fitted L x W footprint: centre in the camera frame, yaw on the
table and layer index, which is what a grasp planner needs.

Outputs in <out>/<recording>/<frame>/:
  final_instances.png  uint16 brick ids after splitting (0 = background)
  final_semantic.png   uint8 colour class per pixel (see segment_lego_sam3_single_prompt.py)
  overlay.png          top-layer bricks as coloured patches with the fitted footprints
  preview.png          input | all SAM3 bricks | top layer from depth | bricks after the size prior
  bricks.json          per brick: id, status, colour, quality, measured size, fitted pose
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from top_layer_depth import filter_top, semantic_overlay


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


def to_pixels(q_mm, K, normal, offset, height, e1, e2):
    """In-plane mm coordinates at `height` above the table -> image pixels."""
    q = np.asarray(q_mm, np.float64) / 1000
    points = q[:, :1] * e1 + q[:, 1:] * e2 + (height - offset) * normal
    p = (K @ points.T).T
    return p[:, :2] / p[:, 2:]


def apply_size_prior(instance, kept_records, sam_records, calibration, report, args):
    """Measure, split and fit every kept top-layer mask; return the brick id image and brick records."""
    L, W, _ = args.brick_size_mm
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

        width_ok = abs(width - W) <= tol * W
        if width_ok and n >= 1 and abs(length - n * L) <= tol * L:
            status = 'brick' if n == 1 else 'split'
        elif (width_ok and length < (1 - tol) * L) or width < (1 - tol) * W:
            status, n = 'partial', 0
        else:
            status, n = 'irregular', 1

        base = {'source_instance': r['id'], 'status': status, 'height_m': round(h, 4),
                'approx_layer': r.get('approx_layer'), 'measured_mm': [round(length, 1), round(width, 1)]}
        if r['id'] in colours:
            c = colours[r['id']]
            base.update(colour=c['colour'], class_id=c['class_id'], quality=c['quality'])
        if status == 'partial':
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
            centre_cam = (centre_q[0] * e1 + centre_q[1] * e2) / 1000 + (h - offset) * normal
            bricks.append({**base, 'id': ident,
                           'centre_cam_m': np.round(centre_cam, 4).tolist(),
                           'yaw_deg_table': round(float(long_deg) % 180, 1),
                           'footprint_px': np.round(to_pixels(corners, K, normal, offset, h, e1, e2), 1).tolist()})
    return bricks_img, bricks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('recording', type=Path, help='Extracted recording folder containing rgb/, depth/, camera_info.json')
    p.add_argument('--frame', default='000000')
    p.add_argument('--out', type=Path, default=Path('data/top_layer_bricks'))
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--brick-size-mm', nargs=3, type=float, default=[30, 15, 10], metavar=('L', 'W', 'H'),
                   help='brick length, width, height (H is also the layer height)')
    p.add_argument('--size-tol', type=float, default=0.3, help='relative tolerance on length and width')
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
    semantic, instance, sam_records, _, _ = segment(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), model, processor, args, device, dtype)
    if not np.any(instance):
        p.error('No foreground instances found; adjust SAM3 prompt/threshold')

    keep, report = filter_top(instance, depth, calibration, args)
    kept = [r for r in report['instances'] if r['kept']]
    bricks_img, bricks = apply_size_prior(instance, kept, sam_records, calibration, report, args)

    # colour class per brick pixel, from the SAM3 instance it came from
    final_semantic = np.where(bricks_img > 0, semantic, 0).astype('uint8')
    brick_records = [{'id': b['id'], 'class_id': b['class_id']} for b in bricks if b['id'] and 'class_id' in b]
    overlay = semantic_overlay(bgr, bricks_img, brick_records, {b['id'] for b in brick_records})
    for b in bricks:
        if b['id']:
            cv2.polylines(overlay, [np.round(np.array(b['footprint_px'])).astype(np.int32)], True,
                          (0, 255, 255) if b['status'] != 'irregular' else (255, 0, 255), 1, cv2.LINE_AA)
    all_bricks = semantic_overlay(bgr, instance, sam_records, {r['id'] for r in sam_records})
    depth_top = semantic_overlay(bgr, instance, sam_records, {r['id'] for r in kept})
    preview = np.vstack([np.hstack([bgr, all_bricks]), np.hstack([depth_top, overlay])])

    output = args.out / args.recording.name / args.frame
    output.mkdir(parents=True, exist_ok=True)
    for name, image in {'final_instances.png': bricks_img, 'final_semantic.png': final_semantic,
                        'overlay.png': overlay, 'preview.png': preview}.items():
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError(f'Cannot write {output / name}')
    counts = {s: sum(b['status'] == s for b in bricks) for s in ('brick', 'split', 'partial', 'irregular')}
    (output / 'bricks.json').write_text(json.dumps({
        'recording': str(args.recording), 'frame': args.frame, 'brick_size_mm': args.brick_size_mm,
        'size_tol': args.size_tol, 'top_height_m': report['top_height_m'],
        'table_normal': report['table_normal'], 'table_offset_m': report['table_offset_m'],
        'status_counts': counts, 'bricks': bricks}, indent=2))
    n_final = sum(1 for b in bricks if b['id'])
    print(f'{len(kept)} top-layer masks -> {n_final} bricks {counts} -> {output}')

    if not args.no_show and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print(f'No display; open {output / "preview.png"} instead (use --no-show to skip this check)')
    elif not args.no_show:
        title = 'RGB | SAM3 bricks / depth top layer | size prior'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
