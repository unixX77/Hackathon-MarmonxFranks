"""Classify already-detected top-layer LEGO boxes as good or bad.

This module is intentionally independent of SAM and RGB-D segmentation.  It
consumes ``top_boxes.json`` files produced by ``segment_lego_rgbd_topplane.py``
and uses robust colour measurements inside each oriented box.  The two bad
colours are medium nougat and reddish brown; all other configured colours are
good.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# OpenCV 8-bit Lab references measured previously with the project camera.
# They are deliberately kept here rather than imported from the segmenter so
# classification can run without torch, transformers, SAM, or a GPU.
COLOUR_PROTOTYPES = {
    'red':           ([(86, 174, 167)], 'good'),
    'dark_red':      ([(60, 151, 133)], 'good'),
    'dark_orange':   ([(111, 139, 172)], 'good'),
    'medium_nougat': ([(136, 131, 141)], 'bad'),
    'reddish_brown': ([(57, 136, 142), (66, 135, 131)], 'bad'),
}
BAD_COLOURS = {'medium_nougat', 'reddish_brown'}
CLASS_NAMES = list(COLOUR_PROTOTYPES)
LAB_WEIGHTS = np.array([0.45, 1.0, 1.0], np.float32)


def polygon_mask(shape, corners):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1)
    return mask


def estimate_white_balance(rgb, boxes):
    """Estimate conservative RGB gains from neutral table pixels outside boxes."""
    occupied = np.zeros(rgb.shape[:2], np.uint8)
    for box in boxes:
        occupied |= polygon_mask(rgb.shape, np.asarray(box['corners_xy']))
    occupied = cv2.dilate(occupied, np.ones((31, 31), np.uint8)) > 0
    pixels = rgb[~occupied].astype(np.float32)
    if not len(pixels):
        return np.ones(3, np.float32), [0.0, 0.0, 0.0]
    brightness = pixels.mean(axis=1)
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    neutral = pixels[(brightness >= 70) & (brightness <= 245) & (chroma <= 35)]
    if len(neutral) < 500:
        neutral = pixels[(brightness >= 70) & (brightness <= 245)]
    if not len(neutral):
        return np.ones(3, np.float32), [0.0, 0.0, 0.0]
    table_rgb = np.median(neutral, axis=0)
    target = float(table_rgb.mean())
    gains = np.clip(target / np.maximum(table_rgb, 1), 0.75, 1.25)
    return gains.astype(np.float32), np.round(table_rgb, 2).tolist()


def corrected_lab(rgb, gains, colour_space="cielab"):
    corrected = np.clip(rgb.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    if colour_space == 'cielab':
        return corrected, cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)
    if colour_space != 'oklab':
        raise ValueError(f'Unknown colour space: {colour_space}')
    # sRGB -> linear RGB -> Oklab (Björn Ottosson's published matrices).
    srgb = corrected.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    lms = linear @ np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]]).T
    oklab = np.cbrt(lms) @ np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]]).T
    # Shared measurement/features use L on 0..255 and neutral a,b at 128.
    # Keep float precision; these encoded coordinates are not CIELAB values.
    return corrected, oklab * 255.0 + [0.0, 128.0, 128.0]


def measure_box(lab, box, erosion_fraction, min_pixels):
    """Measure robust Lab colour away from box edges, shadows, and highlights."""
    mask = polygon_mask(lab.shape, np.asarray(box['corners_xy']))
    short_side = max(3.0, min(map(float, box['size_wh'])))
    erosion = max(1, int(round(short_side * erosion_fraction)))
    kernel_size = erosion * 2 + 1
    inner = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8)) > 0
    if inner.sum() < min_pixels:
        inner = mask > 0
    pixels = lab[inner].astype(np.float32)
    if not len(pixels):
        raise ValueError(f"box {box.get('id')} contains no image pixels")
    low, high = np.percentile(pixels[:, 0], [15, 90])
    reliable = pixels[(pixels[:, 0] >= low) & (pixels[:, 0] <= high)]
    if len(reliable) < min_pixels:
        reliable = pixels
    median = np.median(reliable, axis=0)
    q25, q75 = np.percentile(reliable, [25, 75], axis=0)
    return median, q25, q75, int(len(reliable))


def prototype_arrays():
    refs, classes = [], []
    for name, (values, _) in COLOUR_PROTOTYPES.items():
        for value in values:
            refs.append(value)
            classes.append(name)
    return np.asarray(refs, np.float32), classes


def estimate_scene_offset(features, references, iterations=6):
    """Estimate bounded Lab drift jointly across all bricks in one frame."""
    offset = np.zeros(3, np.float32)
    for _ in range(iterations):
        distances = np.linalg.norm(
            ((features - offset)[:, None] - references[None]) * LAB_WEIGHTS,
            axis=2)
        nearest = distances.argmin(axis=1)
        offset = np.median(features - references[nearest], axis=0)
        # Brightness changes substantially in the stack camera, while large
        # chroma shifts are more likely to erase genuine class differences.
        offset = np.clip(offset, [-60, -18, -18], [60, 18, 18])
    return offset


def classify_features(features, max_distance, min_margin):
    references, reference_classes = prototype_arrays()
    offset = estimate_scene_offset(features, references)
    corrected = features - offset
    predictions = []
    for feature, adjusted in zip(features, corrected):
        per_class = {}
        for name in CLASS_NAMES:
            indices = [i for i, value in enumerate(reference_classes) if value == name]
            distance = np.linalg.norm(
                (references[indices] - adjusted) * LAB_WEIGHTS, axis=1).min()
            per_class[name] = float(distance)
        ordered = sorted(per_class, key=per_class.get)
        colour, runner_up = ordered[:2]
        distance = per_class[colour]
        margin = per_class[runner_up] - distance
        quality = 'bad' if colour in BAD_COLOURS else 'good'
        confident = distance <= max_distance and margin >= min_margin
        predictions.append({
            'colour': colour,
            'quality': quality,
            'confidence': 'high' if confident else 'low',
            'colour_distance': round(distance, 3),
            'distance_margin': round(margin, 3),
            'runner_up': runner_up,
            'lab_raw': np.round(feature, 2).tolist(),
            'lab_corrected': np.round(adjusted, 2).tolist(),
            'class_distances': {key: round(value, 3)
                                for key, value in per_class.items()},
        })
    return predictions, np.round(offset, 3).tolist()


def render(rgb, boxes):
    result = rgb.copy()
    for box in boxes:
        good = box['quality'] == 'good'
        colour = (20, 220, 20) if good else (240, 25, 25)
        corners = np.rint(box['corners_xy']).astype(np.int32)
        cv2.polylines(result, [corners], True, colour, 1, cv2.LINE_AA)
        x, y = map(int, box['center_xy'])
        label = f"#{box['id']}"
        cv2.putText(result, label, (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, colour, 1, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    title = f'TOP LAYER: {len(boxes)} bricks'
    cv2.putText(result, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (255, 255, 255), 1, cv2.LINE_AA)
    return result, np.hstack((rgb, result))


def classify_result(top_boxes_path, output_root, args):
    payload = json.loads(top_boxes_path.read_text())
    rgb_path = Path(payload['rgb'])
    if not rgb_path.exists():
        raise FileNotFoundError(f'RGB source not found: {rgb_path}')
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    boxes = payload['boxes']
    gains, table_rgb = estimate_white_balance(rgb, boxes)
    model_path = getattr(args, 'colour_model', None)
    model = None
    colour_space = 'cielab'
    if model_path:
        from colour_model import ColourModel
        model = ColourModel.load(model_path)
        colour_space = model.payload.get('colour_space', 'cielab')
    corrected_rgb, lab = corrected_lab(rgb, gains, colour_space)
    features, measurements = [], []
    for box in boxes:
        median, q25, q75, pixels = measure_box(
            lab, box, args.erosion_fraction, args.min_pixels)
        features.append(median)
        measurements.append({'lab_q25': np.round(q25, 2).tolist(),
                             'lab_q75': np.round(q75, 2).tolist(),
                             'reliable_pixels': pixels})
    model_path = getattr(args, 'colour_model', None)
    if features and model_path:
        # Learned model: table-relative features replace the per-frame Lab offset.
        from colour_model import ColourModel, colour_features, table_lightness
        table_l = table_lightness(lab, boxes)
        predictions, scene_transform = model.classify(
            [colour_features(median, table_l) for median in features], features)
        offset = [0.0, 0.0, 0.0]
    elif features:
        predictions, offset = classify_features(
            np.asarray(features), args.max_distance, args.min_margin)
        scene_transform = None
    else:
        predictions, offset, scene_transform = [], [0.0, 0.0, 0.0], None
    quality_boxes = []
    for box, measurement, prediction in zip(boxes, measurements, predictions):
        quality_boxes.append({**box, **prediction, **measurement, 'colour_space': colour_space})
    result, comparison = render(rgb, quality_boxes)

    dataset = payload.get('dataset', 'dataset')
    recording = payload.get('recording', top_boxes_path.parents[1].name)
    frame = int(payload.get('frame', 0))
    destination = output_root / dataset / recording / f'{frame:06d}'
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(destination / 'quality_result.png')
    Image.fromarray(comparison).save(destination / 'quality_comparison.png')
    output_payload = {
        'strategy': 'learned-colour-model-v1' if model_path else 'robust-lab-prototype-v1',
        'colour_model': str(model_path) if model_path else None,
        'colour_space': colour_space,
        'coordinate_encoding': 'L*255, a*255+128, b*255+128' if colour_space == 'oklab' else 'OpenCV uint8 Lab',
        'segmentation_source': str(top_boxes_path),
        'rgb': str(rgb_path),
        'dataset': dataset,
        'recording': recording,
        'frame': frame,
        'white_balance': {'table_rgb': table_rgb,
                          'rgb_gains': np.round(gains, 5).tolist()},
        'scene_lab_offset': offset,
        'scene_feature_transform': scene_transform,
        'good': sum(box['quality'] == 'good' for box in quality_boxes),
        'bad': sum(box['quality'] == 'bad' for box in quality_boxes),
        'low_confidence': sum(box['confidence'] == 'low' for box in quality_boxes),
        'boxes': quality_boxes,
    }
    (destination / 'quality_boxes.json').write_text(
        json.dumps(output_payload, indent=2))
    return output_payload, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segmentation_results', type=Path,
                        help='top-plane result folder containing top_boxes.json files')
    parser.add_argument('--out', type=Path, default=Path('results_rgbd_quality_v1'))
    parser.add_argument('--erosion-fraction', type=float, default=.12)
    parser.add_argument('--min-pixels', type=int, default=80)
    parser.add_argument('--max-distance', type=float, default=35.0)
    parser.add_argument('--min-margin', type=float, default=4.0)
    parser.add_argument('--colour-model', type=Path,
                        help='JSON from colour_model.py; omit to use the fixed Lab prototypes')
    args = parser.parse_args()

    inputs = sorted(args.segmentation_results.rglob('top_boxes.json'))
    if not inputs:
        raise SystemExit(f'No top_boxes.json files found under {args.segmentation_results}')
    summary = []
    for index, path in enumerate(inputs, 1):
        print(f'[{index}/{len(inputs)}] {path}', flush=True)
        result, destination = classify_result(path, args.out, args)
        summary.append({
            'dataset': result['dataset'], 'recording': result['recording'],
            'frame': result['frame'], 'bricks': len(result['boxes']),
            'good': result['good'], 'bad': result['bad'],
            'low_confidence': result['low_confidence'],
            'comparison': str(destination / 'quality_comparison.png'),
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps({'results': summary}, indent=2))
    with (args.out / 'summary.csv').open('w', newline='') as stream:
        fields = ['dataset', 'recording', 'frame', 'bricks', 'good', 'bad',
                  'low_confidence', 'comparison']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    print(f'Wrote {len(summary)} classified results to {args.out}')


if __name__ == '__main__':
    main()
