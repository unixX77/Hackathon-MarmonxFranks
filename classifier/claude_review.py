"""Claude Code review of a final pipeline result: fix wrong colours and drop wrong masks.

Runs the `claude` CLI headless (your Claude Code login, no API key) on images written to
<final>/claude_review/: the raw frame, an overview with every final brick numbered, and
close-ups. Classifier labels are not shown, so Claude judges each brick blind. For each
brick it returns a colour, a confidence and whether the mask is a real top-layer brick.

Correction rule (geometry of kept bricks is never changed):
  colour   Claude disagrees with medium/high confidence -> brick takes Claude's colour
           (class_id and good/bad follow); the classifier label is kept under `claude_review`.
  mask     Claude says not_a_brick / not_top_layer / merged with high confidence -> the brick
           is dropped (id set to None, mask removed from the reviewed instance image).
Bricks Claude sees but the pipeline missed are only reported, never added.

Writes into the final result folder (classifier files are left untouched):
  bricks_reviewed.json, poses_reviewed.csv, overlay_reviewed.png,
  final_instances_reviewed.png, final_semantic_reviewed.png
  preview.png                 gains a 4th column: 7 what Claude changed, 8 corrected result
  claude_review/review.json   Claude's raw answer and run metadata

Run on an existing result:
  python classifier/claude_review.py <run>/03_final/capture/<recording>/<frame>
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'classifier'))
sys.path.insert(0, str(ROOT / 'perception/sam3_detection'))
from claude_second_opinion import BAD_COLOURS, COLOURS, SYSTEM, close_up, label, outline, overview

MASK_PROBLEMS = ['not_a_brick', 'not_top_layer', 'merged']
TILE = 200
TILES_PER_SHEET = 20

RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'bricks': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {
                'id': {'type': 'integer'},
                'colour': {'type': 'string', 'enum': COLOURS + ['unsure']},
                'colour_confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
                'mask': {'type': 'string', 'enum': ['ok'] + MASK_PROBLEMS},
                'mask_confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
                'note': {'type': 'string'},
            },
            'required': ['id', 'colour', 'colour_confidence', 'mask', 'mask_confidence', 'note'],
            'additionalProperties': False}},
        'missed': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['bricks', 'missed'],
    'additionalProperties': False,
}

PROMPT = """{system}

You are checking the output of a perception pipeline that finds the top layer of a LEGO \
brick stack and classifies each brick's colour. Use the Read tool on every image in this \
directory before answering:
- rgb.png: the raw camera frame
- overview.png: crop of the frame; every brick the pipeline kept is outlined in white and numbered
- {sheets}: close-up of each numbered brick, each tile labelled with its number

For each of the {count} numbered bricks (ids {ids}) give:
- colour: one of the five colours, or "unsure"
- mask: "ok" if the outline covers exactly one brick on the top layer; "merged" if it covers \
two or more bricks; "not_top_layer" if the brick is clearly lower than the bricks around it; \
"not_a_brick" if it is not a brick at all
- a confidence for each, and a note under 15 words
Under "missed", describe any top-layer brick visible in rgb.png that has no number \
(where it is and its colour); leave it empty if there are none."""


def contact_sheets(bgr, bricks):
    """Close-ups tiled 5 wide, each labelled with its brick id."""
    sheets = []
    for start in range(0, len(bricks), TILES_PER_SHEET):
        chunk = bricks[start:start + TILES_PER_SHEET]
        rows = -(-len(chunk) // 5)
        sheet = np.zeros((rows * TILE, 5 * TILE, 3), np.uint8)
        for k, b in enumerate(chunk):
            crop = close_up(bgr, b)
            scale = TILE / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            y, x = (k // 5) * TILE, (k % 5) * TILE
            sheet[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            label(sheet, str(b['id']), (x + 6, y + 22), 0.7)
        sheets.append(sheet)
    return sheets


def ask_claude(folder, bricks, model, effort, timeout):
    sheets = sorted(p.name for p in folder.glob('closeups_*.png'))
    prompt = PROMPT.format(system=SYSTEM, sheets=', '.join(sheets), count=len(bricks),
                           ids=', '.join(str(b['id']) for b in bricks))
    claude = shutil.which('claude')
    if claude is None:
        raise RuntimeError('claude CLI not found on PATH')
    command = [claude, '-p', prompt, '--output-format', 'json',
               '--json-schema', json.dumps(RESPONSE_SCHEMA), '--tools', 'Read', '--allowedTools', 'Read',
               '--model', model, '--effort', effort, '--no-session-persistence']
    run = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=timeout)
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f'claude exited {run.returncode}: {(run.stderr or run.stdout).strip()[:500]}')
    if result.get('is_error') or not result.get('structured_output'):
        raise RuntimeError(f"claude review failed: {result.get('subtype')} {str(result.get('result'))[:500]}")
    meta = {k: result.get(k) for k in ('session_id', 'total_cost_usd', 'duration_ms', 'num_turns')}
    meta['model'] = ', '.join(result.get('modelUsage', {})) or model
    return result['structured_output'], meta


def apply(result, instance, answers):
    """Return reviewed copies of the result and instance mask, plus the list of changes."""
    from segment_lego_sam3_single_prompt import NAMES
    reviewed = copy.deepcopy(result)
    reviewed_instance = instance.copy()
    by_id = {a['id']: a for a in answers['bricks']}
    changes = []
    for brick in reviewed['bricks']:
        ident = brick.get('id')
        if not ident:
            continue
        answer = by_id.get(ident)
        if answer is None:
            brick['claude_review'] = {'action': 'unchecked'}
            continue
        entry = {'action': 'kept', 'claude_colour': answer['colour'],
                 'colour_confidence': answer['colour_confidence'], 'mask': answer['mask'],
                 'mask_confidence': answer['mask_confidence'], 'note': answer['note']}
        if answer['mask'] in MASK_PROBLEMS and answer['mask_confidence'] == 'high':
            entry.update(action='removed', removed_id=ident)
            reviewed_instance[reviewed_instance == ident] = 0
            brick['id'] = None
            changes.append({'id': ident, 'action': 'removed', 'reason': answer['mask'],
                            'footprint_px': brick['footprint_px'], 'note': answer['note']})
        elif (answer['colour'] not in ('unsure', brick['colour'])
              and answer['colour_confidence'] in ('high', 'medium')):
            entry.update(action='recoloured', classifier_colour=brick['colour'],
                         classifier_quality=brick['quality'])
            changes.append({'id': ident, 'action': 'recoloured', 'from': brick['colour'],
                            'to': answer['colour'], 'footprint_px': brick['footprint_px'],
                            'note': answer['note']})
            brick.update(colour=answer['colour'], class_id=NAMES.index(answer['colour']) + 1,
                         quality='bad' if answer['colour'] in BAD_COLOURS else 'good',
                         colour_confidence=answer['colour_confidence'])
        brick['claude_review'] = entry
    final = [b for b in reviewed['bricks'] if b.get('id')]
    reviewed['good'] = sum(b['quality'] == 'good' for b in final)
    reviewed['bad'] = len(final) - reviewed['good']
    reviewed['claude_review_missed'] = answers['missed']
    return reviewed, reviewed_instance, changes


def changes_view(bgr, bricks, changes):
    """Raw frame: unchanged bricks thin white, recoloured yellow with old->new, removed magenta."""
    out = bgr.copy()
    changed = {c['id'] for c in changes}
    for b in bricks:
        if b['id'] not in changed:
            cv2.polylines(out, [np.rint(b['footprint_px']).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    for c in changes:
        corners = np.asarray(c['footprint_px'])
        x, y = corners.min(axis=0).astype(int)
        if c['action'] == 'removed':
            outline(out, corners, (255, 0, 255))
            cv2.line(out, tuple(np.rint(corners[0]).astype(int)), tuple(np.rint(corners[2]).astype(int)),
                     (255, 0, 255), 2, cv2.LINE_AA)
            label(out, f"#{c['id']} removed ({c['reason']})", (x, max(y - 6, 36)), 0.45)
        else:
            outline(out, corners, (0, 230, 255))
            label(out, f"#{c['id']} {c['from']}->{c['to']}", (x, max(y - 6, 36)), 0.45)
    return out


def review(final_folder, model='opus', effort='medium', timeout=900):
    """Review a final result folder; returns the reviewed bricks.json path, or None."""
    from top_layer_bricks import final_view, titled, write_poses_csv
    final_folder = Path(final_folder)
    result = json.loads((final_folder / 'bricks.json').read_text())
    rgb_path = Path(result['recording']) / 'rgb' / f"{int(result['frame']):06d}.png"
    bgr = cv2.imread(str(rgb_path))
    instance = cv2.imread(str(final_folder / 'final_instances.png'), -1)
    preview = cv2.imread(str(final_folder / 'preview.png'))
    if bgr is None or instance is None or preview is None:
        raise FileNotFoundError(f'Missing RGB ({rgb_path}), final_instances.png or preview.png')
    bricks = [b for b in result['bricks'] if b.get('id')]
    if not bricks:
        print('Claude review: no bricks to review.')
        return None

    folder = final_folder / 'claude_review'
    folder.mkdir(exist_ok=True)
    cv2.imwrite(str(folder / 'rgb.png'), bgr)
    cv2.imwrite(str(folder / 'overview.png'), overview(bgr, bricks))
    for k, sheet in enumerate(contact_sheets(bgr, bricks), 1):
        cv2.imwrite(str(folder / f'closeups_{k}.png'), sheet)
    print(f'Claude review: checking {len(bricks)} bricks with claude -p ({model}, effort {effort})...', flush=True)
    answers, meta = ask_claude(folder, bricks, model, effort, timeout)
    (folder / 'review.json').write_text(json.dumps({**meta, **answers}, indent=2))

    reviewed, reviewed_instance, changes = apply(result, instance, answers)
    reviewed['claude_review'] = {**meta, 'changes': len(changes)}
    final = [b for b in reviewed['bricks'] if b.get('id')]
    semantic = np.zeros(reviewed_instance.shape, np.uint8)
    for b in final:
        semantic[reviewed_instance == b['id']] = b['class_id']
    overlay = final_view(bgr, reviewed_instance, reviewed['bricks'])

    h, w = bgr.shape[:2]
    if preview.shape[:2] == (2 * h, 4 * w):  # reviewed before: replace the old 4th column
        preview = preview[:, :3 * w]
    if preview.shape[:2] != (2 * h, 3 * w):
        raise ValueError('Expected the six-panel preview')
    recoloured = sum(c['action'] == 'recoloured' for c in changes)
    column = np.vstack([
        titled(changes_view(bgr, bricks, changes), f'7 Claude changes: {recoloured} recoloured, '
                                                   f'{len(changes) - recoloured} removed, {len(answers["missed"])} missed'),
        titled(overlay, f"8 after Claude review: {reviewed['good']} good, {reviewed['bad']} bad")])
    cv2.imwrite(str(final_folder / 'preview.png'), np.hstack([preview, column]))
    cv2.imwrite(str(final_folder / 'overlay_reviewed.png'), overlay)
    cv2.imwrite(str(final_folder / 'final_instances_reviewed.png'), reviewed_instance)
    cv2.imwrite(str(final_folder / 'final_semantic_reviewed.png'), semantic)
    (final_folder / 'bricks_reviewed.json').write_text(json.dumps(reviewed, indent=2))
    write_poses_csv(final_folder / 'poses_reviewed.csv', final)

    cost = f", ${meta['total_cost_usd']:.2f}" if meta.get('total_cost_usd') else ''
    print(f"Claude review ({meta['model']}{cost}): {recoloured} recoloured, "
          f"{len(changes) - recoloured} removed, {len(answers['missed'])} missed")
    for c in changes:
        what = f"{c['from']} -> {c['to']}" if c['action'] == 'recoloured' else f"removed ({c['reason']})"
        print(f"  #{c['id']}: {what}  {c['note']}")
    for m in answers['missed']:
        print(f'  missed: {m}')
    return final_folder / 'bricks_reviewed.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('final_folder', type=Path, help='03_final/.../<frame> folder containing bricks.json')
    parser.add_argument('--model', default='opus', help='claude CLI model alias or full name')
    parser.add_argument('--effort', default='medium', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    parser.add_argument('--timeout', type=int, default=900, help='seconds before the review is abandoned')
    args = parser.parse_args()
    review(args.final_folder.resolve(), args.model, args.effort, args.timeout)


if __name__ == '__main__':
    main()
