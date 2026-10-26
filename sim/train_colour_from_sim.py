"""Train the brick colour model on simulated scenes and test it on real labelled bricks.

Each visible top-layer brick of every simulated frame is measured exactly as the pipeline
measures a detected brick: its top-face footprint (from ground truth) becomes the box,
white balance comes from the plate around the boxes, and the same table-relative colour
features are computed. Optional camera augmentation (exposure, white-balance drift, gamma)
imitates the real D435's automatic adjustments.

Real test sets (never used for training):
  - the extracted single-brick dataset (data/brick_dataset_extracted), and
  - any real capture listed with --real-capture <bricks.json> <labels>, where labels are
    "id:colour" pairs, e.g. --real-capture data/top_layer_bricks/20261026_124136/000000/bricks.json
    1:red 2:reddish_brown 3:red 4:medium_nougat

Example:
  python sim/train_colour_from_sim.py data/sim/v1 --out classifier/colour_model_sim.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'classifier'))
import classify_top_brick_quality as base  # noqa: E402
from colour_model import ColourModel, colour_features, load_dataset, table_lightness  # noqa: E402

BRICK_L, BRICK_W = 0.0318, 0.0158


def project(camera_info, position, rotation, point):
    intr = camera_info['aligned_depth_intrinsics']
    q = np.asarray(rotation).T @ (np.asarray(point) - position)
    x, y, z = q[0], -q[1], -q[2]
    return np.array([intr['fx'] * x / z + intr['ppx'], intr['fy'] * y / z + intr['ppy']])


def augment(rgb, rng):
    """Camera-like variation: exposure gain, white-balance drift, gamma."""
    linear = (rgb.astype(np.float32) / 255.0) ** 2.2
    linear *= rng.uniform(0.6, 1.5) * rng.normal(1.0, 0.04, 3)
    return np.clip(255 * linear ** (1 / (2.2 * rng.uniform(0.9, 1.1))), 0, 255).astype(np.uint8)


def sim_samples(root, colour_space, augment_copies, min_visible, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for scene in sorted(Path(root).glob('scene_*')):
        camera_info = json.loads((scene / 'camera_info.json').read_text())
        for label_path in sorted((scene / 'labels').glob('*.json')):
            label = json.loads(label_path.read_text())
            bgr = cv2.imread(str(scene / 'rgb' / f'{label_path.stem}.png'))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            position = np.array(label['camera_position_world_m'])
            rotation = np.array(label['camera_rotation_world'])
            boxes, colours = [], []
            for brick in label['bricks']:
                if not brick['top_layer'] or brick['visible_px'] < min_visible:
                    continue
                centre = np.array(brick['centre_world_m'])
                yaw = math.radians(brick['yaw_world_deg'])
                axis, perp = np.array([math.cos(yaw), math.sin(yaw), 0]), np.array([-math.sin(yaw), math.cos(yaw), 0])
                corners = [project(camera_info, position, rotation, centre + sx * axis * BRICK_L / 2 + sy * perp * BRICK_W / 2)
                           for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
                corners = np.array(corners)
                h, w = rgb.shape[:2]
                if corners.min() < 0 or corners[:, 0].max() >= w or corners[:, 1].max() >= h:
                    continue
                boxes.append({'id': brick['id'], 'corners_xy': corners.tolist(),
                              'size_wh': [float(np.linalg.norm(corners[0] - corners[1])),
                                          float(np.linalg.norm(corners[1] - corners[2]))]})
                colours.append(brick['colour'])
            if not boxes:
                continue
            for copy in range(1 + augment_copies):
                image = rgb if copy == 0 else augment(rgb, rng)
                gains, _ = base.estimate_white_balance(image, boxes)
                _, lab = base.corrected_lab(image, gains, colour_space)
                table_l = table_lightness(lab, boxes)
                for box, colour in zip(boxes, colours):
                    median, *_ = base.measure_box(lab, box, 0.12, 80)
                    samples.append({'scene': scene.name, 'colour': colour,
                                    'features': colour_features(median, table_l)})
    return samples


def real_capture_samples(bricks_json, labels, colour_space):
    """Real capture with known colours: footprints from the pipeline, labels from the user."""
    result = json.loads(Path(bricks_json).read_text())
    truth = dict(label.split(':') for label in labels)
    rgb_path = Path(result['recording']) / 'rgb' / f"{int(result['frame']):06d}.png"
    if not rgb_path.is_absolute():
        rgb_path = ROOT / rgb_path
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    boxes = []
    for brick in result['bricks']:
        if brick.get('id') and str(brick['id']) in truth:
            corners = np.asarray(brick['footprint_px'])
            boxes.append({'id': brick['id'], 'corners_xy': corners.tolist(),
                          'size_wh': [float(np.linalg.norm(corners[0] - corners[1])),
                                      float(np.linalg.norm(corners[1] - corners[2]))]})
    gains, _ = base.estimate_white_balance(rgb, boxes)
    _, lab = base.corrected_lab(rgb, gains, colour_space)
    table_l = table_lightness(lab, boxes)
    return [{'colour': truth[str(box['id'])], 'id': box['id'],
             'features': colour_features(base.measure_box(lab, box, 0.12, 80)[0], table_l)} for box in boxes]


def report(name, model, samples):
    if not samples:
        return None
    labels = np.array([s['colour'] for s in samples])
    predicted = np.array(model.predict(np.array([s['features'] for s in samples])))
    accuracy = float(np.mean(predicted == labels))
    bad = np.isin(labels, list(base.BAD_COLOURS))
    quality = float(np.mean(np.isin(predicted, list(base.BAD_COLOURS)) == bad))
    print(f'  {name}: colour {accuracy:.0%}, good/bad {quality:.0%} ({len(samples)} bricks)')
    for (t, p), n in sorted(Counter(zip(labels, predicted)).items()):
        if t != p:
            print(f'      {t} -> {p}: {n}')
    return {'colour_accuracy': accuracy, 'quality_accuracy': quality, 'bricks': len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sim_root', type=Path)
    parser.add_argument('--out', type=Path, help='save the sim-trained model here')
    parser.add_argument('--colour-space', choices=['cielab', 'oklab'], default='cielab')
    parser.add_argument('--augment-copies', type=int, default=2)
    parser.add_argument('--min-visible-px', type=int, default=2500)
    parser.add_argument('--real-dataset', type=Path, default=ROOT / 'data/brick_dataset_extracted')
    parser.add_argument('--real-capture', nargs='+', action='append', default=[],
                        metavar=('BRICKS_JSON', 'ID:COLOUR'))
    parser.add_argument('--compare-model', type=Path, default=ROOT / 'classifier/colour_model.json',
                        help='existing model to score on the same real test sets')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    samples = sim_samples(args.sim_root, args.colour_space, args.augment_copies, args.min_visible_px, args.seed)
    print(f'sim samples: {len(samples)} {dict(Counter(s["colour"] for s in samples))}')
    scenes = sorted({s['scene'] for s in samples})
    held = set(scenes[::5])                                   # every 5th scene held out
    train = [s for s in samples if s['scene'] not in held]
    model = ColourModel.fit(np.array([s['features'] for s in train]), np.array([s['colour'] for s in train]))
    model.payload.update({'colour_space': args.colour_space, 'trained_on': f'simulation {args.sim_root}',
                          'samples': len(train), 'augment_copies': args.augment_copies})

    real = load_dataset(args.real_dataset.resolve(), 0.12, 80, args.colour_space) if args.real_dataset.exists() else []
    captures = [s for spec in args.real_capture for s in real_capture_samples(spec[0], spec[1:], args.colour_space)]
    print('\nsim-trained model')
    evaluation = {'sim_held_out_scenes': report('simulation, held-out scenes', model,
                                                [s for s in samples if s['scene'] in held]),
                  'real_dataset': report('REAL single-brick dataset', model, real),
                  'real_captures': report('REAL labelled captures', model, captures)}
    model.payload['evaluation'] = evaluation
    if args.compare_model and args.compare_model.exists():
        existing = ColourModel.load(args.compare_model)
        if existing.payload.get('colour_space', 'cielab') == args.colour_space:
            print(f'\nexisting model {args.compare_model.name} (trained on the real dataset itself)')
            report('REAL single-brick dataset (its own training data)', existing, real)
            report('REAL labelled captures', existing, captures)
    if args.out:
        args.out.write_text(json.dumps(model.payload, indent=2))
        print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
