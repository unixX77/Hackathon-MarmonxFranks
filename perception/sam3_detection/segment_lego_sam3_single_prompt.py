"""SAM 3 instance + colour-semantic segmentation of LEGO bricks (single text prompt).

One text prompt (default "lego brick") gives the instances. Each instance is then
filtered (overlap, size, rectangularity) and given a colour class by nearest-palette
matching of its robust Lab colour, with a per-image global colour offset estimated
jointly over all bricks (absorbs white-balance / exposure drift).

For every image `<stem>.png` the output folder gets `<stem>/` containing:
  semantic.png     uint8 class image, 0 = background, k = PALETTE index + 1
  instance.png     uint16 instance image, 0 = background, 1..N = bricks
  instances.json   per brick: id, colour, class_id, quality, score, box, area, centroid,
                   min-area rect, Lab colour, colour distance; plus rejected detections
  preview.png      input | instances (id, score) | semantic colours (colour name)
The output root also gets summary.json with per-image counts per colour.

Run in the `sam3` conda env (needs Hugging Face access to facebook/sam3).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# name: (reference Lab colours in OpenCV 8-bit units, display RGB, quality).
# RECALIBRATE these with this script's brick_lab (see the 'lab' field in instances.json),
# under fixed camera white balance / exposure.
PALETTE = {
    'red':           ([(86, 174, 167)], (230, 30, 30), 'good'),
    'dark_red':      ([(60, 151, 133)], (120, 10, 30), 'good'),
    'dark_orange':   ([(111, 139, 172)], (230, 140, 20), 'good'),
    'medium_nougat': ([(136, 131, 141)], (225, 190, 150), 'bad'),
    'reddish_brown': ([(57, 136, 142), (66, 135, 131)], (110, 60, 30), 'bad'),
}
NAMES = list(PALETTE)
QUALITY = {n: PALETTE[n][2] for n in NAMES}
REF_LAB = np.array([ref for n in NAMES for ref in PALETTE[n][0]], dtype=np.float32)
REF_CLASS = np.array([k for k, n in enumerate(NAMES) for _ in PALETTE[n][0]])
DISPLAY = np.array([(0, 0, 0)] + [PALETTE[n][1] for n in NAMES], dtype=np.uint8)
L_WEIGHT = 0.5  # brightness varies with exposure more than chroma does


# ---------------------------------------------------------------- detection

def run_sam3(model, processor, image, prompt, threshold, mask_threshold, device, dtype):
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    r = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=mask_threshold,
        target_sizes=[image.size[::-1]])[0]
    scores = r['scores'].float().cpu().numpy()
    if not len(scores):  # empty results are not resized to the image
        return np.zeros((0, image.height, image.width), bool), scores
    return r['masks'].cpu().numpy().astype(bool), scores


def dedupe(masks, scores, max_overlap):
    """Keep masks by descending score; drop any mask mostly covered by already kept ones."""
    keep, covered = [], np.zeros(masks.shape[1:], bool)
    for i in np.argsort(-scores):
        area = masks[i].sum()
        if area and np.logical_and(masks[i], covered).sum() / area <= max_overlap:
            keep.append(i)
            covered |= masks[i]
    return keep


def rect_stats(mask):
    """Rectangularity (mask area / min-area-rect area) and the min-area rect itself."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    return float(mask.sum() / max(w * h, 1.0)), {'center_xy': [round(cx, 1), round(cy, 1)],
                                                  'size_wh': [round(w, 1), round(h, 1)],
                                                  'angle_deg': round(ang, 1)}


# ---------------------------------------------------------------- colour

def brick_lab(lab, mask):
    """Robust brick colour: eroded interior, without shadows (dark 10%) and specular (bright 10%)."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    px = lab[inner if inner.sum() >= 50 else mask]
    lo, hi = np.percentile(px[:, 0], [10, 90])
    px = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
    return np.median(px, axis=0)


def classify_all(feats, iters=5):
    """Nearest palette colour for all bricks of one image, with a jointly estimated global
    Lab offset (robust median), so a colour cast shifts the bricks back onto the palette."""
    if not len(feats):
        return np.zeros(0, int), np.zeros(0), np.zeros(3)
    w = np.array([L_WEIGHT, 1, 1], np.float32)
    offset = np.zeros(3, np.float32)
    for _ in range(iters):
        d = np.linalg.norm(((feats - offset)[:, None] - REF_LAB[None]) * w, axis=2)
        j = d.argmin(1)
        offset = np.median(feats - REF_LAB[j], axis=0)
    d = np.linalg.norm(((feats - offset)[:, None] - REF_LAB[None]) * w, axis=2)
    j = d.argmin(1)
    return REF_CLASS[j], d[np.arange(len(j)), j], offset


# ---------------------------------------------------------------- drawing

def colour(i):
    hsv = np.uint8([[[(i * 47) % 180, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def put_text(img, text, org, scale):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 0, 0), int(3 * scale))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), max(1, int(scale)))


def preview(rgb, instance, records):
    scale = max(rgb.shape[0] / 480, 1.0)
    inst, sem = rgb.copy(), rgb.copy()
    for r in records:
        m = instance == r['id']
        inst[m] = (0.4 * rgb[m] + 0.6 * np.array(colour(r['id']))).astype(np.uint8)
        sem[m] = (0.25 * rgb[m] + 0.75 * DISPLAY[r['class_id']]).astype(np.uint8)
    for r in records:
        contours, _ = cv2.findContours((instance == r['id']).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for img in (inst, sem):
            cv2.drawContours(img, contours, -1, (255, 255, 255), max(1, int(2 * scale)))
        x, y = map(int, r['centroid_xy'])
        put_text(inst, f"#{r['id']} {r['score']:.2f}", (x - int(20 * scale), y), scale)
        put_text(sem, r['colour'], (x - int(30 * scale), y), scale * 0.8)
    put_text(inst, f'{len(records)} bricks', (int(10 * scale), int(30 * scale)), scale * 1.5)
    return np.hstack([rgb, inst, sem])


# ---------------------------------------------------------------- reusable interface

def add_detection_args(parser):
    """SAM 3 and filtering options shared by batch and live scripts."""
    parser.add_argument('--prompt', default='lego brick')
    parser.add_argument('--model', default='facebook/sam3')
    parser.add_argument('--threshold', type=float, default=0.5, help='instance score threshold')
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--max-overlap', type=float, default=0.5,
                        help='drop an instance if this fraction is already covered by better ones')
    parser.add_argument('--min-area', type=int, default=300, help='drop instances smaller than this (px)')
    parser.add_argument('--min-rect', type=float, default=0.7,
                        help='drop instances whose area / min-area-rect area is below this')
    parser.add_argument('--min-rel-area', type=float, default=0.4,
                        help='drop instances smaller than this fraction of the median brick area')
    parser.add_argument('--max-colour-dist', type=float, default=35.0,
                        help='drop instances further than this from every palette colour (after offset)')


class BrickSegmenter:
    """Load SAM 3 once and apply the single-prompt brick pipeline to RGB arrays."""

    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        self.processor = Sam3Processor.from_pretrained(args.model)
        self.model = Sam3Model.from_pretrained(
            args.model, torch_dtype=self.dtype).to(self.device).eval()

    def __call__(self, rgb):
        return segment(rgb, self.model, self.processor, self.args, self.device, self.dtype)


class BrickGeometrySegmenter(BrickSegmenter):
    """SAM brick instances without colour or good/bad classification."""

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        masks, scores = run_sam3(
            self.model, self.processor, Image.fromarray(rgb), self.args.prompt,
            self.args.threshold, self.args.mask_threshold, self.device, self.dtype)
        rejected, candidates = [], []
        for i in (dedupe(masks, scores, self.args.max_overlap) if len(scores) else []):
            area = int(masks[i].sum())
            if area < self.args.min_area:
                rejected.append({'score': float(scores[i]), 'reason': 'small',
                                 'area_px': area})
                continue
            rectness, rect = rect_stats(masks[i])
            if rectness < self.args.min_rect:
                rejected.append({'score': float(scores[i]), 'reason': 'not rectangular',
                                 'rectangularity': round(rectness, 2), 'area_px': area})
                continue
            candidates.append((i, rectness, rect))

        instance = np.zeros((h, w), np.uint16)
        records = []
        for i, rectness, rect in candidates:
            mask = masks[i] & (instance == 0)
            if not mask.any():
                continue
            inst_id = len(records) + 1
            instance[mask] = inst_id
            ys, xs = np.nonzero(mask)
            records.append({
                'id': inst_id, 'score': round(float(scores[i]), 3),
                'area_px': int(mask.sum()),
                'box_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                'centroid_xy': [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                'min_area_rect': rect, 'rectangularity': round(rectness, 2),
                'touches_border': bool(xs.min() == 0 or ys.min() == 0 or
                                       xs.max() == w - 1 or ys.max() == h - 1),
            })
        return instance, records, rejected


def save_result(out_dir, image_path, rgb, semantic, instance, records, rejected,
                colour_offset, args):
    """Write the standard result artifacts and return one summary entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(semantic).save(out_dir / 'semantic.png')
    Image.fromarray(instance).save(out_dir / 'instance.png')
    Image.fromarray(preview(rgb, instance, records)).save(out_dir / 'preview.png')
    (out_dir / 'instances.json').write_text(json.dumps({
        'image': str(image_path), 'prompt': args.prompt, 'threshold': args.threshold,
        'classes': {i + 1: name for i, name in enumerate(NAMES)}, 'quality': QUALITY,
        'colour_offset_lab': np.round(colour_offset, 1).tolist(),
        'instances': records, 'rejected': rejected,
    }, indent=2))
    per_colour = {}
    for record in records:
        per_colour[record['colour']] = per_colour.get(record['colour'], 0) + 1
    good = sum(record['quality'] == 'good' for record in records)
    return {
        'image': Path(image_path).name, 'bricks': len(records), 'good': good,
        'bad': len(records) - good, 'per_colour': per_colour,
        'rejected': len(rejected),
        'colour_offset_lab': np.round(colour_offset, 1).tolist(),
    }


def write_summary(out, summary):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps({
        'classes': {i + 1: name for i, name in enumerate(NAMES)}, 'quality': QUALITY,
        'per_image': summary,
    }, indent=2))


# ---------------------------------------------------------------- main

def segment(rgb, model, processor, args, device, dtype, mask_filter=None):
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    masks, scores = run_sam3(model, processor, Image.fromarray(rgb), args.prompt,
                             args.threshold, args.mask_threshold, device, dtype)
    rejected = []

    # Geometry must reject assembly masks before they can suppress contained bricks.
    if mask_filter is not None and len(scores):
        accepted, geometry_rejected = mask_filter(masks, scores)
        masks, scores = masks[accepted], scores[accepted]
        rejected.extend(geometry_rejected)

    # 1. overlap + absolute size + rectangularity
    cand = []
    for i in (dedupe(masks, scores, args.max_overlap) if len(scores) else []):
        area = int(masks[i].sum())
        if area < args.min_area:
            rejected.append({'score': float(scores[i]), 'reason': 'small', 'area_px': area})
            continue
        rectness, rect = rect_stats(masks[i])
        if rectness < args.min_rect:
            rejected.append({'score': float(scores[i]), 'reason': 'not rectangular',
                             'rectangularity': round(rectness, 2), 'area_px': area})
            continue
        cand.append((i, area, rectness, rect))

    # 2. relative size: slivers of lower-layer bricks seen through gaps
    if cand:
        med = np.median([c[1] for c in cand])
        kept = []
        for c in cand:
            if c[1] < args.min_rel_area * med:
                rejected.append({'score': float(scores[c[0]]), 'reason': 'sliver',
                                 'area_rel': round(c[1] / med, 2)})
            else:
                kept.append(c)
        cand = kept

    # 3. colour, jointly over the image
    feats = np.array([brick_lab(lab, masks[c[0]]) for c in cand], np.float32).reshape(-1, 3)
    classes, dists, offset = classify_all(feats)

    semantic = np.zeros((h, w), np.uint8)
    instance = np.zeros((h, w), np.uint16)
    records = []
    for (i, area, rectness, rect), k, dist, f in zip(cand, classes, dists, feats):
        if dist > args.max_colour_dist:
            rejected.append({'score': float(scores[i]), 'reason': 'colour', 'lab': f.round(1).tolist(),
                             'nearest': NAMES[k], 'colour_dist': round(float(dist), 1)})
            continue
        m = masks[i] & (instance == 0)
        if not m.any():
            continue
        inst_id = len(records) + 1
        instance[m] = inst_id
        semantic[m] = k + 1
        ys, xs = np.nonzero(m)
        records.append({
            'id': inst_id, 'colour': NAMES[k], 'class_id': int(k) + 1, 'quality': QUALITY[NAMES[k]],
            'score': round(float(scores[i]), 3), 'area_px': int(m.sum()),
            'box_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            'centroid_xy': [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            'min_area_rect': rect, 'rectangularity': round(rectness, 2),
            'touches_border': bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1),
            'lab': f.round(1).tolist(), 'colour_dist': round(float(dist), 1)})
    return semantic, instance, records, rejected, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='a PNG image or a folder of PNGs')
    parser.add_argument('--out', type=Path, default=Path('data/predictions/sam3_semantic'))
    add_detection_args(parser)
    args = parser.parse_args()

    segmenter = BrickSegmenter(args)

    paths = sorted(args.input.glob('*.png')) if args.input.is_dir() else [args.input]
    print(f'[sam3] {len(paths)} images, prompt={args.prompt!r}, device={segmenter.device}')
    summary = []
    for n, path in enumerate(paths, 1):
        rgb = np.array(Image.open(path).convert('RGB'))
        semantic, instance, records, rejected, offset = segmenter(rgb)
        entry = save_result(args.out / path.stem, path, rgb, semantic, instance,
                            records, rejected, offset, args)
        summary.append(entry)
        print(f'[sam3] {n}/{len(paths)} {path.name}: {entry["bricks"]} bricks '
              f'({entry["good"]} good, {entry["bad"]} bad) {entry["per_colour"]}, '
              f'rejected {entry["rejected"]}, colour offset {entry["colour_offset_lab"]}')

    write_summary(args.out, summary)
    print(f'[sam3] wrote {args.out}')


if __name__ == '__main__':
    main()
