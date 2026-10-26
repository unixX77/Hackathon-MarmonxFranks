"""Learned brick-colour model trained on the extracted single-brick dataset.

Features are chosen to transfer from isolated bricks on the table to bricks in
the stack, where auto exposure and auto white balance differ:

  hue        Lab hue angle of the brick (degrees), nearly exposure invariant
  L_rel      brick lightness / table lightness
  chroma_rel brick Lab chroma / table lightness

Colours are measured with the same white balance and box measurement as
``classify_top_brick_quality.py``.  The model is linear discriminant analysis
with equal class priors, stored as JSON so the pipeline needs only numpy.

Frames with many bricks (a stack) also get a bounded scene correction: a hue
shift and L/chroma scales chosen so the frame's bricks best fit the colour
clusters.  This absorbs per-capture auto-exposure/white-balance drift.  Frames
with few bricks are left uncorrected because the fit would be unreliable.

Train and evaluate:
  python classifier/colour_model.py --dataset data/brick_dataset_extracted \
      --out classifier/colour_model.json
"""

import argparse
import csv
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

import classify_top_brick_quality as base


FEATURES = ['hue', 'L_rel', 'chroma_rel']
DEFAULT_THRESHOLD = 0.7
SCENE_MIN_BOXES = 10
SCENE_HUE_SHIFTS = np.arange(-10, 10.5, 1.0)
SCENE_SCALES = np.exp(np.linspace(np.log(0.6), np.log(1.6), 25))


def table_lightness(lab, boxes):
    """Median L of table pixels well away from every box."""
    occupied = np.zeros(lab.shape[:2], np.uint8)
    for box in boxes:
        occupied |= base.polygon_mask(lab.shape, np.asarray(box['corners_xy']))
    occupied = cv2.dilate(occupied, np.ones((31, 31), np.uint8)) > 0
    values = lab[..., 0][~occupied]
    return float(np.median(values)) if len(values) else 255.0


def colour_features(median_lab, table_l):
    a, b = float(median_lab[1]) - 128, float(median_lab[2]) - 128
    table_l = max(table_l, 1.0)
    return [float(np.degrees(np.arctan2(b, a))), float(median_lab[0]) / table_l,
            float(np.hypot(a, b)) / table_l]


class ColourModel:
    def __init__(self, payload):
        self.payload = payload
        self.classes = payload['classes']
        self.mean = np.asarray(payload['feature_mean'])
        self.scale = np.asarray(payload['feature_scale'])
        self.class_means = np.asarray(payload['class_means'])
        self.precision = np.linalg.inv(np.asarray(payload['covariance']))
        self.threshold = float(payload['threshold'])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def fit(cls, features, labels, threshold=DEFAULT_THRESHOLD, shrinkage=0.05):
        features, labels = np.asarray(features, float), np.asarray(labels)
        classes = [name for name in base.CLASS_NAMES if name in set(labels)]
        mean, scale = features.mean(axis=0), features.std(axis=0) + 1e-9
        z = (features - mean) / scale
        class_means = np.array([z[labels == name].mean(axis=0) for name in classes])
        residuals = np.concatenate([z[labels == name] - m
                                    for name, m in zip(classes, class_means)])
        covariance = np.cov(residuals.T)
        covariance = ((1 - shrinkage) * covariance +
                      shrinkage * np.eye(len(covariance)) * np.trace(covariance) / len(covariance))
        return cls({'features': FEATURES, 'classes': classes,
                    'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
                    'class_means': class_means.tolist(), 'covariance': covariance.tolist(),
                    'threshold': threshold})

    def distances(self, features):
        z = (np.asarray(features, float) - self.mean) / self.scale
        delta = z[:, None] - self.class_means[None]
        return np.sqrt(np.einsum('nki,ij,nkj->nk', delta, self.precision, delta))

    def probabilities(self, features):
        logits = -0.5 * self.distances(features) ** 2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, features):
        return [self.classes[i] for i in self.probabilities(features).argmax(axis=1)]

    def scene_transform(self, features):
        """Hue shift and L/chroma scales maximizing the frame's cluster likelihood."""
        features = np.asarray(features, float)
        identity = {'hue_shift_deg': 0.0, 'L_scale': 1.0, 'chroma_scale': 1.0, 'applied': False}
        if len(features) < SCENE_MIN_BOXES:
            return identity
        best, best_score = None, -np.inf
        for shift in SCENE_HUE_SHIFTS:
            shifted = features + [shift, 0.0, 0.0]
            # All L/chroma scale pairs for this shift at once: (scales, scales, boxes, 3).
            grid = shifted[None, None] * np.stack(np.broadcast_arrays(
                1.0, SCENE_SCALES[:, None, None], SCENE_SCALES[None, :, None]), axis=-1)
            z = (grid - self.mean) / self.scale
            delta = z[..., None, :] - self.class_means
            d2 = np.einsum('...ki,ij,...kj->...k', delta, self.precision, delta)
            score = np.log(np.exp(-0.5 * d2).sum(axis=-1) + 1e-300).sum(axis=-1)
            i, j = np.unravel_index(np.argmax(score), score.shape)
            if score[i, j] > best_score:
                best_score, best = score[i, j], (shift, SCENE_SCALES[i], SCENE_SCALES[j])
        return {'hue_shift_deg': float(best[0]), 'L_scale': round(float(best[1]), 4),
                'chroma_scale': round(float(best[2]), 4), 'applied': True}

    @staticmethod
    def apply_transform(features, transform):
        return (np.asarray(features, float) * [1.0, transform['L_scale'], transform['chroma_scale']]
                + [transform['hue_shift_deg'], 0.0, 0.0])

    def classify(self, features, median_labs):
        """Predictions in the standalone classifier's per-box schema, plus the scene transform."""
        transform = self.scene_transform(features)
        adjusted = self.apply_transform(features, transform)
        distances, probabilities = self.distances(adjusted), self.probabilities(adjusted)
        predictions = []
        for feature, lab, d, p in zip(adjusted, median_labs, distances, probabilities):
            first, second = np.argsort(-p)[:2]
            colour = self.classes[first]
            predictions.append({
                'colour': colour,
                'quality': 'bad' if colour in base.BAD_COLOURS else 'good',
                'confidence': 'high' if p[first] >= self.threshold else 'low',
                'probability': round(float(p[first]), 3),
                'colour_distance': round(float(d[first]), 3),
                'distance_margin': round(float(d[second] - d[first]), 3),
                'runner_up': self.classes[second],
                'lab_raw': np.round(lab, 2).tolist(),
                'lab_corrected': np.round(lab, 2).tolist(),
                'features': dict(zip(FEATURES, np.round(feature, 4).tolist())),
                'class_probabilities': {name: round(float(v), 4)
                                        for name, v in zip(self.classes, p)},
                'class_distances': {name: round(float(v), 3)
                                    for name, v in zip(self.classes, d)},
            })
        return predictions, transform


# ---------------------------------------------------------------- training

def load_dataset(root, erosion_fraction, min_pixels):
    lighting_path = root / 'lighting_suggestions.csv'
    lighting = ({row['sample_id']: row['lighting'] for row in csv.DictReader(lighting_path.open())}
                if lighting_path.exists() else {})
    samples = []
    for entry in json.loads((root / 'manifest.json').read_text()):
        if not entry.get('mask') or not (root / entry['mask']).exists():
            continue
        bgr = cv2.imread(entry['source'])
        mask = cv2.imread(str(root / entry['mask']), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None or not mask.any():
            continue
        points = np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = {'id': entry['id'], 'corners_xy': cv2.boxPoints(rect).tolist(),
               'size_wh': list(rect[1])}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gains, _ = base.estimate_white_balance(rgb, [box])
        _, lab = base.corrected_lab(rgb, gains)
        median, *_ = base.measure_box(lab, box, erosion_fraction, min_pixels)
        samples.append({'id': entry['id'], 'colour': entry['color'],
                        'lighting': lighting.get(entry['id'], 'unknown'),
                        'features': colour_features(median, table_lightness(lab, [box]))})
    return samples


def confusion(labels, predictions, classes):
    counts = Counter(zip(labels, predictions))
    width = max(map(len, classes)) + 2
    lines = ['true \\ predicted'.ljust(width) + ''.join(c[:12].rjust(14) for c in classes)]
    for true in classes:
        lines.append(true.ljust(width) + ''.join(str(counts[(true, p)]).rjust(14) for p in classes))
    return '\n'.join(lines)


def cross_validate(features, labels, splits, threshold):
    predictions = np.empty(len(labels), object)
    confidence = np.zeros(len(labels))
    for train, test in splits:
        model = ColourModel.fit(features[train], labels[train], threshold)
        p = model.probabilities(features[test])
        predictions[test] = [model.classes[i] for i in p.argmax(axis=1)]
        confidence[test] = p.max(axis=1)
    return predictions, confidence


def summarize(name, labels, predictions, confidence, threshold):
    bad = np.isin(labels, list(base.BAD_COLOURS))
    predicted_bad = np.isin(predictions.astype(str), list(base.BAD_COLOURS))
    sure = confidence >= threshold
    result = {'colour_accuracy': float(np.mean(predictions == labels)),
              'quality_accuracy': float(np.mean(predicted_bad == bad)),
              'confident_fraction': float(sure.mean()),
              'confident_colour_accuracy': float(np.mean(predictions[sure] == labels[sure]))
              if sure.any() else None}
    print(f"\n## {name}\ncolour {result['colour_accuracy']:.1%}, good/bad "
          f"{result['quality_accuracy']:.1%}; confident (p>={threshold}) "
          f"{result['confident_fraction']:.0%} of bricks at "
          f"{result['confident_colour_accuracy'] or 0:.1%} colour accuracy")
    print(confusion(labels, predictions, [c for c in base.CLASS_NAMES if c in set(labels)]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=Path('data/brick_dataset_extracted'))
    parser.add_argument('--out', type=Path, default=Path('classifier/colour_model.json'))
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='minimum class probability for a high-confidence prediction')
    parser.add_argument('--erosion-fraction', type=float, default=.12)
    parser.add_argument('--min-pixels', type=int, default=80)
    args = parser.parse_args()
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    samples = load_dataset(args.dataset.resolve(), args.erosion_fraction, args.min_pixels)
    if not samples:
        raise SystemExit(f'No usable samples in {args.dataset}')
    features = np.array([s['features'] for s in samples])
    labels = np.array([s['colour'] for s in samples])
    print(f'{len(samples)} samples: {dict(Counter(labels))}')

    lighting = np.array([s['lighting'] for s in samples])
    evaluations = {}
    for name, splits in [
            ('random 5-fold (optimistic: near-duplicate frames on both sides)',
             StratifiedKFold(5, shuffle=True, random_state=0).split(features, labels)),
            ('one lighting condition held out (pessimistic: lighting tied to colour)',
             LeaveOneGroupOut().split(features, labels, lighting))]:
        predictions, confidence = cross_validate(features, labels, list(splits), args.threshold)
        evaluations[name] = summarize(name, labels, predictions, confidence, args.threshold)

    model = ColourModel.fit(features, labels, args.threshold)
    model.payload.update({
        'trained_on': str(args.dataset.resolve()), 'samples': len(samples),
        'class_counts': dict(Counter(labels.tolist())),
        'measurement': {'erosion_fraction': args.erosion_fraction, 'min_pixels': args.min_pixels},
        'evaluation': evaluations,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model.payload, indent=2))
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
