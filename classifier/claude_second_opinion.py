"""Second opinion on brick colours from Claude (vision), after the standalone classifier.

Claude sees an overview of the frame with every final brick numbered (so colours can be
compared under the same lighting) and a close-up of each brick. Classifier labels are not
shown. For each brick it returns one of the five colours (or "unsure"), a confidence and
flags. Geometry and poses are never changed.

Merge rule: agreement changes nothing. When Claude disagrees with medium or high confidence,
the brick keeps the classifier's colour but its colour confidence becomes "low" and Claude's
answer is recorded beside it, so the robot side can skip or re-check that brick.

Writes into the final result folder:
  bricks_second_opinion.json  bricks.json plus a `second_opinion` entry per brick
  second_opinion.png          overview: green = agree, red = disagree, grey = unsure
  second_opinion_raw.json     Claude's raw answer

Needs `pip install anthropic` in the environment and credentials (ANTHROPIC_API_KEY or
`ant auth login`). Run on an existing result:
  python classifier/claude_second_opinion.py <run>/03_final/capture/<recording>/<frame>
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

MODEL = 'claude-opus-5-5'
COLOURS = ['red', 'dark_red', 'dark_orange', 'medium_nougat', 'reddish_brown']
BAD_COLOURS = {'medium_nougat', 'reddish_brown'}
FLAGS = ['glare', 'merged', 'partial', 'not_top_layer', 'not_a_brick']

SYSTEM = """You check the colour of LEGO 2x4 bricks seen from above by a RealSense D435 colour \
camera over a light table. Every brick is one of five LEGO colours:
- red (bright, saturated red)
- dark_red (deep wine/maroon red)
- dark_orange (saturated orange-brown, clearly orange)
- medium_nougat (light tan / beige / flesh tone, pale and much lighter than dark_orange)
- reddish_brown (dark chocolate brown with a red tint, browner and less red than dark_red)
Lighting and exposure vary, so judge each brick relative to the other bricks in the same \
overview image rather than by absolute colour. Dark_red and reddish_brown are the hardest \
pair: dark_red looks red-purple, reddish_brown looks brown. Glare can wash a brick out to \
grey-white; then report the colour of the unaffected part, or "unsure".
Use "unsure" when you cannot tell; a wrong confident answer is worse than "unsure"."""

RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {'bricks': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {
            'id': {'type': 'integer'},
            'colour': {'type': 'string', 'enum': COLOURS + ['unsure']},
            'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
            'flags': {'type': 'array', 'items': {'type': 'string', 'enum': FLAGS}},
            'note': {'type': 'string'},
        },
        'required': ['id', 'colour', 'confidence', 'flags', 'note'],
        'additionalProperties': False}}},
    'required': ['bricks'],
    'additionalProperties': False,
}


def png_block(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('cannot encode image')
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                        'data': base64.standard_b64encode(data.tobytes()).decode()}}


def label(image, text, xy, scale=0.6):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def outline(image, corners, colour, grow=4):
    """Outline drawn just outside the footprint so the brick's own pixels stay untouched."""
    corners = np.asarray(corners, float)
    centre = corners.mean(axis=0)
    direction = corners - centre
    grown = centre + direction * (1 + grow / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1))
    cv2.polylines(image, [np.rint(grown).astype(np.int32)], True, colour, 2, cv2.LINE_AA)


def overview(bgr, bricks, colours=None, margin=60):
    """Crop around all bricks with each footprint outlined and numbered."""
    points = np.concatenate([np.asarray(b['footprint_px']) for b in bricks])
    x0, y0 = np.maximum(points.min(axis=0).astype(int) - margin, 0)
    x1, y1 = np.minimum(points.max(axis=0).astype(int) + margin, [bgr.shape[1], bgr.shape[0]])
    image = bgr.copy()
    for b in bricks:
        outline(image, b['footprint_px'], (colours or {}).get(b['id'], (255, 255, 255)))
        x, y = np.mean(b['footprint_px'], axis=0).astype(int)
        label(image, str(b['id']), (x - 8, y + 6))
    return image[y0:y1, x0:x1]


def close_up(bgr, brick, margin=0.35, min_side=160):
    """Axis-aligned crop around one footprint with some context, upscaled if small."""
    corners = np.asarray(brick['footprint_px'])
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    pad = margin * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(x0 - pad, 0)), int(max(y0 - pad, 0))
    x1, y1 = int(min(x1 + pad, bgr.shape[1])), int(min(y1 + pad, bgr.shape[0]))
    image = bgr.copy()
    outline(image, corners, (255, 255, 255))
    crop = image[y0:y1, x0:x1]
    scale = max(1.0, min_side / min(crop.shape[:2]))
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop


def ask_claude(bgr, bricks, effort='medium'):
    import anthropic

    content = [png_block(overview(bgr, bricks)),
               {'type': 'text', 'text': 'Overview: every top-layer brick is outlined in white '
                                        'and numbered. Close-ups follow.'}]
    for b in bricks:
        content += [{'type': 'text', 'text': f"Brick {b['id']}:"}, png_block(close_up(bgr, b))]
    content.append({'type': 'text', 'text':
                    f"Give the colour of each of the {len(bricks)} numbered bricks "
                    f"(ids {', '.join(str(b['id']) for b in bricks)}). Set flags when a brick "
                    'shows glare, looks like two bricks merged, is mostly hidden, does not lie on '
                    'the top layer, or is not a brick. Keep each note under 15 words.'})
    client = anthropic.Anthropic()
    # Server-side fallback: if the request is declined by a safety classifier, the API
    # re-runs it on a suitable fallback model within the same call.
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',
        output_config={'effort': effort,
                       'format': {'type': 'json_schema', 'schema': RESPONSE_SCHEMA}},
        messages=[{'role': 'user', 'content': content}],
    )
    if response.stop_reason == 'refusal':
        raise RuntimeError(f'Claude declined the request: {response.stop_details}')
    if response.stop_reason == 'max_tokens':
        raise RuntimeError('Claude response was cut off (max_tokens)')
    text = next(block.text for block in response.content if block.type == 'text')
    return json.loads(text), {'model': response.model, 'request_id': response._request_id,
                              'input_tokens': response.usage.input_tokens,
                              'output_tokens': response.usage.output_tokens}


def merge(result, answers):
    """Attach Claude's answer to every brick and lower confidence on disagreement."""
    by_id = {a['id']: a for a in answers['bricks']}
    summary = {'agree': 0, 'disagree': 0, 'unsure': 0, 'missing': 0}
    for brick in result['bricks']:
        if not brick.get('id'):
            continue
        answer = by_id.get(brick['id'])
        if answer is None:
            summary['missing'] += 1
            brick['second_opinion'] = {'status': 'missing'}
            continue
        if answer['colour'] == 'unsure' or answer['confidence'] == 'low':
            status = 'unsure'
        elif answer['colour'] == brick['colour']:
            status = 'agree'
        else:
            status = 'disagree'
            brick['colour_confidence'] = 'low'
        summary[status] += 1
        brick['second_opinion'] = {
            'status': status, 'colour': answer['colour'], 'confidence': answer['confidence'],
            'quality': ('bad' if answer['colour'] in BAD_COLOURS else 'good')
            if answer['colour'] != 'unsure' else None,
            'flags': answer['flags'], 'note': answer['note']}
    result['second_opinion_summary'] = summary
    return summary


def review(final_folder, effort='medium'):
    """Run the second opinion on a final result folder; returns the summary."""
    final_folder = Path(final_folder)
    result = json.loads((final_folder / 'bricks.json').read_text())
    recording = Path(result['recording'])
    rgb_path = recording / 'rgb' / f"{int(result['frame']):06d}.png"
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    bricks = [b for b in result['bricks'] if b.get('id')]
    if not bricks:
        print('No bricks to review.')
        return None
    answers, meta = ask_claude(bgr, bricks, effort)
    (final_folder / 'second_opinion_raw.json').write_text(json.dumps({**meta, **answers}, indent=2))
    summary = merge(result, answers)
    result['second_opinion_model'] = meta['model']
    (final_folder / 'bricks_second_opinion.json').write_text(json.dumps(result, indent=2))
    status_colour = {'agree': (60, 210, 60), 'disagree': (40, 40, 240)}
    colours = {b['id']: status_colour.get(b['second_opinion']['status'], (160, 160, 160)) for b in bricks}
    cv2.imwrite(str(final_folder / 'second_opinion.png'), overview(bgr, bricks, colours))
    print(f"Second opinion ({meta['model']}): {summary['agree']} agree, {summary['disagree']} disagree, "
          f"{summary['unsure']} unsure, {summary['missing']} missing "
          f"({meta['input_tokens']} in / {meta['output_tokens']} out tokens)")
    for b in bricks:
        o = b['second_opinion']
        if o['status'] == 'disagree':
            print(f"  #{b['id']}: classifier {b['colour']} ({b['quality']}), Claude {o['colour']} "
                  f"({o['quality']}, {o['confidence']}) {o['flags'] or ''} {o['note']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('final_folder', type=Path, help='03_final/.../<frame> folder containing bricks.json')
    parser.add_argument('--effort', default='medium', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    args = parser.parse_args()
    review(args.final_folder, args.effort)


if __name__ == '__main__':
    main()
