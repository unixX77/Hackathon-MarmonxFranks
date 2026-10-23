"""SAM 3 instance + colour-semantic segmentation of LEGO bricks in RGB images.

SAM 3 is prompted with one text prompt per LEGO colour plus a generic "lego brick"
prompt and returns one mask per brick, including bricks that touch each other.
Detections from all prompts are merged, then each instance is given a colour class
by nearest-centroid matching of its robust Lab colour against PALETTE (more reliable
than the prompt alone for close shades such as red / dark red / reddish brown).
Every colour is also labelled good or bad (see PALETTE).

For every image `<stem>.png` the output folder gets `<stem>/` containing:
  semantic.png     uint8 class image, 0 = background, k = PALETTE index + 1
  instance.png     uint16 instance image, 0 = background, 1..N = bricks
  instances.json   per brick: id, colour, colour_index (n-th brick of that colour),
                   score, box, area, centroid, touches_border, Lab colour, colour distance
  preview.png      input | semantic colours, outlines green = good / red = bad,
                   with a legend listing the good and bad bricks of that image
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

# The five 2x4 bricks in the set (LEGO element id in the comment).
# name: (reference Lab colours in OpenCV 8-bit units, display RGB, quality, SAM 3 prompt).
# Lab references are calibrated on data/rgb; reddish brown looks either brown or
# grey-brown depending on where it lies under the lamp, so it has two references.
PALETTE = {
    'red':           ([(86, 174, 167)], (230, 30, 30), 'good', 'red lego brick'),                     # 300121
    'dark_red':      ([(60, 151, 133)], (120, 10, 30), 'good', 'dark red lego brick'),                # 6117418
    'dark_orange':   ([(111, 139, 172)], (230, 140, 20), 'good', 'orange lego brick'),                # 6487946
    'medium_nougat': ([(136, 131, 141)], (225, 190, 150), 'bad', 'beige lego brick'),                 # 6135191
    'reddish_brown': ([(57, 136, 142), (66, 135, 131)], (110, 60, 30), 'bad', 'brown lego brick'),    # 4211201
}
NAMES = list(PALETTE)
QUALITY = {n: PALETTE[n][2] for n in NAMES}
PROMPTS = [PALETTE[n][3] for n in NAMES] + ['lego brick']
REF_LAB = np.array([ref for n in NAMES for ref in PALETTE[n][0]], dtype=np.float32)
REF_CLASS = np.array([k for k, n in enumerate(NAMES) for _ in PALETTE[n][0]])
DISPLAY = np.array([(0, 0, 0)] + [PALETTE[n][1] for n in NAMES], dtype=np.uint8)
QUALITY_RGB = {'good': (40, 200, 60), 'bad': (235, 40, 40)}
L_WEIGHT = 0.5  # brightness varies with exposure more than chroma does


def run_sam3(model, processor, image, prompts, threshold, mask_threshold, device, dtype):
    """Run every text prompt on the image; return all detections and the prompt that found each."""
    inputs = processor(images=[image] * len(prompts), text=prompts, return_tensors='pt').to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=mask_threshold,
        target_sizes=[image.size[::-1]] * len(prompts))
    masks, scores, boxes, found_by = [], [], [], []
    for prompt, r in zip(prompts, results):
        if len(r['scores']) == 0:  # empty results are not resized to the image
            continue
        masks.append(r['masks'].cpu().numpy().astype(bool))
        scores.append(r['scores'].float().cpu().numpy())
        boxes.append(r['boxes'].float().cpu().numpy().reshape(-1, 4))
        found_by += [prompt] * len(scores[-1])
    if not found_by:
        h, w = image.size[::-1]
        return np.zeros((0, h, w), bool), np.zeros(0), np.zeros((0, 4)), []
    return np.concatenate(masks), np.concatenate(scores), np.concatenate(boxes), found_by


def dedupe(masks, scores, max_overlap):
    """Keep masks by descending score; drop any mask mostly covered by already kept ones
    (removes SAM 3's occasional 'whole row of bricks' mask on top of the single bricks)."""
    keep, covered = [], np.zeros(masks.shape[1:], bool)
    for i in np.argsort(-scores):
        area = masks[i].sum()
        if area and np.logical_and(masks[i], covered).sum() / area <= max_overlap:
            keep.append(i)
            covered |= masks[i]
    return keep


def brick_lab(lab, mask):
    """Robust brick colour: eroded interior, without specular highlights, shadows and the
    least saturated half of the pixels."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    px = lab[inner if inner.sum() >= 50 else mask]
    lo, hi = np.percentile(px[:, 0], [10, 95])
    px = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
    chroma = np.hypot(px[:, 1] - 128, px[:, 2] - 128)
    return np.median(px[chroma >= np.median(chroma)], axis=0)


def classify(lab_colour):
    d = (REF_LAB - lab_colour) * [L_WEIGHT, 1, 1]
    dist = np.linalg.norm(d, axis=1)
    j = int(np.argmin(dist))
    return int(REF_CLASS[j]), float(dist[j])


def put_text(img, text, org, colour=(255, 255, 255), scale=0.45):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_legend(img, records):
    """Good / bad list of this image's bricks in a translucent box, in the corner with fewest bricks."""
    rows = []
    for q in ('good', 'bad'):
        items = [r for r in records if r['quality'] == q]
        rows.append((f'{q.upper()} ({len(items)})', QUALITY_RGB[q], None))
        rows += [(f"#{r['id']} {r['colour']}#{r['colour_index']}", (255, 255, 255), DISPLAY[r['class_id']])
                 for r in items]
    line, pad = 18, 8
    bw = pad * 2 + 22 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for t, _, _ in rows)
    bh = pad * 2 + line * len(rows)
    h, w = img.shape[:2]
    corners = [(5, 5), (w - bw - 5, 5), (5, h - bh - 5), (w - bw - 5, h - bh - 5)]
    fg = np.zeros((h, w), np.int64)
    for r in records:
        x1, y1, x2, y2 = map(int, r['box_xyxy'])
        fg[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] += 1
    x0, y0 = min(corners, key=lambda c: fg[max(c[1], 0):c[1] + bh, max(c[0], 0):c[0] + bw].sum())
    x0, y0 = max(x0, 0), max(y0, 0)
    box = img[y0:y0 + bh, x0:x0 + bw]
    box[:] = (0.35 * box).astype(np.uint8)
    for i, (text, colour, swatch) in enumerate(rows):
        y = y0 + pad + line * i + 13
        if swatch is None:
            put_text(img, text, (x0 + pad, y), colour, 0.5)
        else:
            cv2.rectangle(img, (x0 + pad + 4, y - 10), (x0 + pad + 16, y + 2), tuple(int(c) for c in swatch), -1)
            put_text(img, text, (x0 + pad + 22, y))


def preview(rgb, semantic, instance, records):
    sem = rgb.copy()
    fg = semantic > 0
    sem[fg] = (0.25 * rgb[fg] + 0.75 * DISPLAY[semantic[fg]]).astype(np.uint8)
    for r in records:
        m = (instance == r['id']).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sem, contours, -1, QUALITY_RGB[r['quality']], 2)
        x, y = map(int, r['centroid_xy'])
        put_text(sem, f"#{r['id']} {r['colour']}#{r['colour_index']}", (x - 50, y))
    draw_legend(sem, records)
    return np.hstack([rgb, sem])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path, nargs='?', default=Path('data/rgb'))
    parser.add_argument('--out', type=Path, default=Path('data/predictions/sam3_goodbad'))
    parser.add_argument('--prompts', nargs='+', default=PROMPTS,
                        help='text prompts; detections of all prompts are merged')
    parser.add_argument('--model', default='facebook/sam3')
    parser.add_argument('--threshold', type=float, default=0.6, help='instance score threshold')
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--min-area', type=int, default=300, help='drop instances smaller than this (px)')
    parser.add_argument('--max-overlap', type=float, default=0.5,
                        help='drop an instance if this fraction of it is already covered by better ones')
    parser.add_argument('--max-colour-dist', type=float, default=35.0,
                        help='instances further than this from every palette colour are dropped as non-LEGO')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    paths = sorted(args.root.glob('*.png'))
    print(f'[sam3] {len(paths)} images, prompts={args.prompts}, device={device}')
    summary = []
    for n, path in enumerate(paths, 1):
        image = Image.open(path).convert('RGB')
        rgb = np.array(image)
        h, w = rgb.shape[:2]
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        masks, scores, boxes, found_by = run_sam3(model, processor, image, args.prompts,
                                                  args.threshold, args.mask_threshold, device, dtype)
        semantic = np.zeros((h, w), np.uint8)
        instance = np.zeros((h, w), np.uint16)
        records, per_colour, rejected = [], {}, []
        for i in dedupe(masks, scores, args.max_overlap) if len(scores) else []:
            m = masks[i]
            if m.sum() < args.min_area:
                continue
            colour_lab = brick_lab(lab, m)
            k, dist = classify(colour_lab)
            if dist > args.max_colour_dist:
                rejected.append({'score': float(scores[i]), 'prompt': found_by[i], 'lab': colour_lab.round(1).tolist(),
                                 'nearest': NAMES[k], 'colour_dist': round(dist, 1)})
                continue
            m = m & (instance == 0)
            inst_id = len(records) + 1
            instance[m] = inst_id
            semantic[m] = k + 1
            per_colour[NAMES[k]] = per_colour.get(NAMES[k], 0) + 1
            ys, xs = np.nonzero(m)
            records.append({
                'id': inst_id, 'colour': NAMES[k], 'class_id': k + 1, 'colour_index': per_colour[NAMES[k]],
                'quality': QUALITY[NAMES[k]], 'prompt': found_by[i],
                'score': round(float(scores[i]), 3), 'area_px': int(m.sum()),
                'box_xyxy': [round(float(v), 1) for v in boxes[i]],
                'centroid_xy': [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                'touches_border': bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1),
                'lab': colour_lab.round(1).tolist(), 'colour_dist': round(dist, 1)})

        out_dir = args.out / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(semantic).save(out_dir / 'semantic.png')
        Image.fromarray(instance).save(out_dir / 'instance.png')
        Image.fromarray(preview(rgb, semantic, instance, records)).save(out_dir / 'preview.png')
        (out_dir / 'instances.json').write_text(json.dumps({
            'image': str(path), 'prompts': args.prompts, 'threshold': args.threshold,
            'classes': {i + 1: name for i, name in enumerate(NAMES)}, 'quality': QUALITY,
            'instances': records, 'rejected': rejected}, indent=2))
        n_good = sum(r['quality'] == 'good' for r in records)
        summary.append({'image': path.name, 'bricks': len(records), 'good': n_good,
                        'bad': len(records) - n_good, 'per_colour': per_colour})
        print(f'[sam3] {n}/{len(paths)} {path.name}: {len(records)} bricks '
              f'({n_good} good, {len(records) - n_good} bad) {per_colour}'
              + (f', rejected {len(rejected)}' if rejected else ''))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps({
        'classes': {i + 1: name for i, name in enumerate(NAMES)}, 'quality': QUALITY,
        'per_image': summary}, indent=2))
    print(f'[sam3] wrote {args.out}')


if __name__ == '__main__':
    main()
