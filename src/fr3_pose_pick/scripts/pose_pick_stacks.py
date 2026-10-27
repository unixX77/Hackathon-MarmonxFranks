#!/usr/bin/python3 -s
"""AUTONOMOUS sorting BY COLOUR into stacks: every good brick goes onto the stack of its colour, bad bricks to the bin.

The same machinery as pose_pick_palletize.py (no questions, countdown, every safety check, the place aligned by the measured hand
position, the gripper's finger width confirms every grasp). The bricks carry their colour:

  ros2 run fr3_pose_pick pose_pick_stacks.py --config src/fr3_pose_pick/config/pose_pick.yaml \\
      --brick X1 Y1 Z1 A1 G red  --brick X2 Y2 Z2 A2 B  --brick X3 Y3 Z3 A3 G dark_red

(or one brick: --camera-xyz X Y Z --camera-angle A --label G --colour red), or from the camera pipeline through stack_listener.py.

  A STACK (one per good colour, centred on its entry in colour_stacks):
  - every brick lies with its long side along base x (pointing from the robot base toward the operator);
  - a layer is stack_side_by_side bricks next to each other (along base y) x stack_end_to_end bricks end to end (along base x),
    3 x 2 by default, with stack_gap (2 studs, 16 mm) between neighbours both ways: 80 x 80 mm for 3 x 2;
  - when a layer is full the next brick starts the layer above, straight on top of the first one (3 x 2 x n), up to
    stack_max_layers;
  - a layer fills row by row: along +y first, then the next row along +x.

  A brick is put down like a palletized one: over its slot, down to 4 cm, aligned by the measured hand position, then let go with the
  hand at palletize_row_drop_z (layer 1, measured on this plate) plus (layer - 1) x (brick height + stack_layer_gain). A dropped
  brick rests ON the studs below, not in them, so every layer stands about a stud (1.7 mm) higher than the model: the gain adds that.
  With palletize_place_mode: press it is pressed in instead.

  THE CENTRES: colour_stacks {colour: [x, y]} in the config, or recorded: move the hand (pointing down) over the middle of where a
  colour's stack should go and run  record_pose.py stack_<colour> --config ...  (e.g. stack_red); its x, y is the centre, the height
  does not matter. A good brick whose colour has no stack is LEFT ON THE PILE (said so). Each stack is remembered in
  pose_pick_stacks.json (current directory); its centre is fixed once a brick is on it (--reset forgets every stack).

Own options:
  --brick X Y Z A G COLOUR | --brick X Y Z A B     a brick to pick; repeat it for several
  --label G|B --colour NAME                         the single brick given with --camera-xyz (default G)
  --status       the stacks: bricks on each and where the next goes (moves nothing)
  --reset        forget every stack (moves nothing)
  --state FILE   default pose_pick_stacks.json
  --countdown SECONDS    as pose_pick_palletize.py
"""
import argparse
import builtins
import json
import math
import os
import sys
import time
from pathlib import Path

import pose_pick as pp
import pose_pick_auto as auto
import pose_pick_palletize as pd
import record_pose as rp

STATE_FILE = 'pose_pick_stacks.json'
RECORDED_PREFIX = 'stack_'     # record_pose.py stack_red: the hand's x, y there is the centre of the red stack


# ------------------------------------------------------------------------------------------------- the stacks
def layout(c):
    """(side by side, end to end, gap, max layers) from the config, checked. Raises ValueError."""
    across, along = int(c.get('stack_side_by_side', 3)), int(c.get('stack_end_to_end', 2))
    gap, layers = float(c.get('stack_gap', 0.016)), int(c.get('stack_max_layers', 6))
    if not (1 <= across <= 10 and 1 <= along <= 10 and 1 <= layers <= 10):
        raise ValueError('stack_side_by_side / stack_end_to_end must be 1 to 10 bricks, stack_max_layers 1 to 10')
    if not 0.008 <= gap <= 0.1:
        raise ValueError(f'stack_gap {gap} must be 8 mm to 10 cm, in METRES (0.016 is 2 studs): the fingers go between the bricks')
    return across, along, gap, layers


def footprint(c):
    """(x size, y size) of one stack layer, metres."""
    across, along, gap, _ = layout(c)
    length, width = c['block_size'][0], c['block_size'][1]
    return along * length + (along - 1) * gap, across * width + (across - 1) * gap


def stacks_from_config(c, config_path):
    """{colour: [x, y]}: colour_stacks from the config, and every pose recorded as stack_<colour> with record_pose.py (its hand x, y is
    the centre). Every stack inside the workspace and clear of the others. Raises ValueError."""
    stacks = c.get('colour_stacks') or {}
    if not isinstance(stacks, dict) or not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in stacks.values()):
        raise ValueError('colour_stacks must be {colour: [x, y]} (the centre of each colour\'s stack, base frame, metres)')
    stacks = dict(stacks)
    for name, record in rp.load_poses(rp.poses_path(config_path)).items():
        if not name.startswith(RECORDED_PREFIX):
            continue
        colour, centre = name[len(RECORDED_PREFIX):], [float(record['xyz'][0]), float(record['xyz'][1])]
        if colour in stacks and max(abs(float(a) - b) for a, b in zip(stacks[colour], centre)) > 1e-4:
            raise ValueError(f'The {colour} stack is both in colour_stacks {list(stacks[colour])} and recorded as {name} '
                             f'[{centre[0]:.4f}, {centre[1]:.4f}]: remove one of them')
        stacks[colour] = centre
    size_x, size_y = footprint(c)
    (x0, x1), (y0, y1) = c['workspace_x'], c['workspace_y']
    gap = layout(c)[2]
    for colour, (x, y) in stacks.items():
        if not (x0 <= x - size_x / 2 and x + size_x / 2 <= x1 and y0 <= y - size_y / 2 and y + size_y / 2 <= y1):
            raise ValueError(f'colour_stacks {colour} at [{x}, {y}]: the stack ({size_x * 1000:.0f} x {size_y * 1000:.0f} mm) reaches '
                             f'outside the workspace x {c["workspace_x"]}, y {c["workspace_y"]}')
    names = list(stacks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dx, dy = abs(stacks[a][0] - stacks[b][0]), abs(stacks[a][1] - stacks[b][1])
            if dx < size_x + gap and dy < size_y + gap:
                raise ValueError(f'colour_stacks {a} and {b} are too close: the centres must be at least {(size_x + gap) * 1000:.0f} mm '
                                 f'apart in x or {(size_y + gap) * 1000:.0f} mm in y')
    return {str(k): [float(v[0]), float(v[1])] for k, v in stacks.items()}


def load_state(path):
    path = Path(path)
    if not path.exists():
        return {'version': 1, 'stacks': {}}
    try:
        state = json.loads(path.read_text())
        assert state['version'] == 1 and isinstance(state['stacks'], dict)
    except (ValueError, KeyError, TypeError, AssertionError):
        raise ValueError(f'{path} is damaged or not a stacks file. Look at it, or start again with --reset.') from None
    return state


def save_state(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, path)                      # never a half-written file


def stack_of(state, colour, centre):
    """The record of `colour`'s stack, made if new. A stack with bricks on it keeps its centre."""
    record = state['stacks'].get(colour)
    if record and record['placed'] and [round(v, 5) for v in record['centre']] != [round(v, 5) for v in centre]:
        raise ValueError(f'The {colour} stack has {len(record["placed"])} brick(s) at {record["centre"]}, and colour_stacks now says '
                         f'{centre}: take them away and --reset, or put the old centre back')
    if not record or not record['placed']:
        record = state['stacks'][colour] = {'centre': list(centre), 'placed': []}
    return record


def slot(c, centre, index):
    """Slot `index` (from 0) of a stack: {'index', 'layer' (from 1), 'row', 'col', 'x', 'y'} (brick centre, base frame)."""
    across, along, gap, _ = layout(c)
    length, width = c['block_size'][0], c['block_size'][1]
    layer, rest = divmod(index, across * along)
    row, col = divmod(rest, across)                   # a row runs along +y; the next row is further along +x
    return {'index': index, 'layer': layer + 1, 'row': row + 1, 'col': col + 1,
            'x': centre[0] + (row - (along - 1) / 2) * (length + gap), 'y': centre[1] + (col - (across - 1) / 2) * (width + gap)}


def release_z(c, layer):
    """The hand height to let go at for a brick of `layer` (drop mode), or None: drop_height above the model's seat of that layer."""
    base = c.get('palletize_row_drop_z')
    if base is None:
        return None
    return float(base) + (layer - 1) * (float(c['block_size'][2]) + float(c.get('stack_layer_gain', 0.0)))


def status_lines(state, c, stacks):
    across, along, _, layers = layout(c)
    size_x, size_y = footprint(c)
    lines = [f'Stacks of {across} side by side x {along} end to end per layer (long sides along base x, {size_x * 1000:.0f} x '
             f'{size_y * 1000:.0f} mm), up to {layers} layers = {across * along * layers} bricks each.']
    for colour in sorted(set(stacks) | set(state['stacks'])):
        record = state['stacks'].get(colour)
        centre = (record or {}).get('centre') if record and record['placed'] else stacks.get(colour)
        count = len(record['placed']) if record else 0
        where = f'centre x={centre[0]:.4f} y={centre[1]:.4f}' if centre else 'NO centre in colour_stacks'
        line = f'  {colour:14s} {count:3d} brick{"s" if count != 1 else " "}  {where}'
        if centre and count < across * along * layers:
            s = slot(c, centre, count)
            line += f'; next: layer {s["layer"]} row {s["row"]} brick {s["col"]} at x={s["x"]:.4f} y={s["y"]:.4f}'
        elif centre:
            line += '; FULL'
        lines.append(line)
    if len(lines) == 1:
        lines.append('  no stacks: record_pose.py stack_<colour> over each stack centre, or colour_stacks {colour: [x, y]} in the config')
    return lines


# ------------------------------------------------------------------------------------------------- the mission
class StackMission(pd.PalletizeMission):
    """PalletizeMission whose place is the next slot of the stack of the brick's colour; main() sets the class attributes."""
    stacks_state = stacks_path = colour = slot = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_placed = self.record_stack if type(self).bin_pose is None else None

    def record_stack(self):
        cls = type(self)
        hand = self.release_hand
        cls.stacks_state['stacks'][cls.colour]['placed'].append({
            'slot': cls.slot['index'] + 1, 'layer': cls.slot['layer'], 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_xy': [round(cls.slot['x'], 5), round(cls.slot['y'], 5)],
            'released_hand_xy': [round(float(hand[0]), 5), round(float(hand[1]), 5)]})
        save_state(cls.stacks_path, cls.stacks_state)
        self.say(f'[stack {cls.colour}] brick {cls.slot["index"] + 1} is down (layer {cls.slot["layer"]} row {cls.slot["row"]} '
                 f'brick {cls.slot["col"]}).')

    def save_log(self):
        self.place_trace['colour'] = type(self).colour
        super().save_log()


# ---------------------------------------------------------------------------------------------------- main
def extract_bricks(argv):
    """Take every --brick X Y Z A G|B [COLOUR] out of the arguments: ([(x, y, z, angle, label, colour), ...], the rest)."""
    bricks, rest, i = [], [], 0
    while i < len(argv):
        if argv[i] != '--brick':
            rest.append(argv[i])
            i += 1
            continue
        values, i = [], i + 1
        while i < len(argv) and not argv[i].startswith('--'):
            values.append(argv[i])
            i += 1
        colour = values.pop() if len(values) == 6 else None
        x, y, z, angle, label = pd.parse_brick(values)
        if label == 'G' and not colour:
            raise ValueError(f'--brick {" ".join(values)}: a good brick needs its colour after the G (e.g. ... G red): it decides the stack')
        bricks.append((x, y, z, angle, label, colour))
    return bricks, rest


def own_parser():
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    own.add_argument('--config')
    own.add_argument('--state', default=STATE_FILE)
    own.add_argument('--label', type=str.upper, choices=sorted(pd.LABELS))
    own.add_argument('--colour')
    own.add_argument('--reset', action='store_true')
    own.add_argument('--status', action='store_true')
    return own


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        print(__doc__)
        return 0
    try:
        bricks, argv = extract_bricks(argv)
        args, rest = own_parser().parse_known_args(argv)
    except SystemExit:
        return 2
    except ValueError as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 2
    try:
        seconds, rest = auto.parse_own(rest, default_place=False)
        for arg in rest:
            if arg.split('=', 1)[0] in pd.PLACE_RELATED:
                raise ValueError(f'{arg.split("=", 1)[0]}: this script decides where each brick goes')
        path = Path(args.state)
        if args.reset:
            if path.exists():
                path.unlink()
                print(f'Every stack forgotten ({path} removed).')
            if not bricks and not any(arg.split('=', 1)[0] in pd.PICK_OPTIONS for arg in rest):
                return 0
        if not args.config:
            raise ValueError('--config is needed')
        import yaml
        c = yaml.safe_load(open(args.config, encoding='utf-8'))
        layout(c)
        stacks = stacks_from_config(c, args.config)
        state = load_state(path)
        if args.status:
            print('\n'.join(status_lines(state, c, stacks)))
            return 0
        drop = pd.check_mode(c)
        if bricks:
            if args.label is not None or args.colour is not None or any(arg.split('=', 1)[0] in pd.PICK_OPTIONS for arg in rest):
                raise ValueError('--brick replaces --camera-xyz / --camera-angle / --pose / --label / --colour: use one way to give the bricks')
            jobs = [(['--camera-xyz', pd.fmt(x), pd.fmt(y), pd.fmt(z), '--camera-angle', pd.fmt(a)], label, colour)
                    for x, y, z, a, label, colour in bricks]
        else:
            if (args.label or 'G') == 'G' and not args.colour:
                raise ValueError('a good brick needs --colour NAME: it decides the stack')
            jobs = [([], args.label or 'G', args.colour)]
        missing = sorted({colour for _, label, colour in jobs if label == 'G' and colour not in stacks})
        if missing:
            print(f'NOTE: no stack in colour_stacks for {", ".join(missing)}: those bricks are LEFT ON THE PILE.', flush=True)
            jobs = [job for job in jobs if job[1] == 'B' or job[2] in stacks]
        across, along, _, layers = layout(c)
        for colour in {colour for _, label, colour in jobs if label == 'G'}:
            record = stack_of(state, colour, stacks[colour])
            want = sum(1 for _, label, other in jobs if label == 'G' and other == colour)
            if len(record['placed']) + want > across * along * layers:
                raise ValueError(f'The {colour} stack has {len(record["placed"])} bricks and room for {across * along * layers}: '
                                 f'{want} more do not fit. Nothing was moved.')
        bin_pose = None
        if any(label == 'B' for _, label, _ in jobs):
            if pd.check_bad_mode(c) != 'bin':
                raise ValueError('pose_pick_stacks.py drops the bad bricks in the bin: set palletize_bad_mode: bin')
            poses = rp.load_poses(rp.poses_path(args.config))
            if 'bin' not in poses:
                raise ValueError('A bad brick goes to the bin pose and none is recorded: record_pose.py bin --config ' + args.config)
            rp.check_bin_pose(c, poses['bin'])
            bin_pose = poses['bin']
    except (ValueError, OSError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 2
    if not jobs:
        print('Nothing to pick.')
        return 0

    base = ['--config', args.config] + rest
    old_input, old_mission, old_argv, old_lookup = builtins.input, pp.Mission, sys.argv, pp.lookup_hand
    pp.Mission = StackMission
    if len(jobs) > 1 and '--ee-pose' not in rest:
        photo = {}

        def lookup_once(node, buffer, config, seconds=10.0):
            if 'pose' not in photo:                                  # the arm pose at the start of this command ...
                photo['pose'] = old_lookup(node, buffer, config, seconds)
            return photo['pose']                                    # ... is the camera pose of every picture
        pp.lookup_hand = lookup_once
        print('Several bricks: the pictures are taken to be from the arm pose it has now (give --ee-pose to say otherwise).', flush=True)
    print('Plan: ' + ', '.join(f'{n}={label}' + (f' {colour}' if label == 'G' else ' (bin)')
                               for n, (_, label, colour) in enumerate(jobs, 1)), flush=True)
    done = 0
    try:
        for number, (pick, label, colour) in enumerate(jobs, 1):
            cls = StackMission
            cls.state = cls.state_path = cls.seat_z = cls.ref = None
            cls.anchor, cls.wide, cls.rings, cls.label, cls.colour = False, False, None, label, colour
            builtins.input = auto.countdown_input(seconds if number == 1 else 0)         # one countdown per command
            if label == 'B':
                print(f'\n=== Brick {number} of {len(jobs)} [B = BAD]: into the BIN at x={bin_pose["xyz"][0]:.4f} '
                      f'y={bin_pose["xyz"][1]:.4f} z={bin_pose["xyz"][2]:.4f} ===', flush=True)
                cls.truth, cls.drop, cls.drop_z, cls.index, cls.bin_pose, cls.slot = None, None, None, 0, bin_pose, None
                place = ['--place-offset', '0', '0']                   # a nominal place: the brick is carried to the bin instead
            else:
                record = stack_of(state, colour, stacks[colour])
                s = slot(c, record['centre'], len(record['placed']))
                z = release_z(c, s['layer'])
                print(f'\n=== Brick {number} of {len(jobs)} [G = GOOD, {colour}]: {colour} stack, layer {s["layer"]} row {s["row"]} '
                      f'brick {s["col"]} at x={s["x"]:.4f} y={s["y"]:.4f}' + (f'; let go with the hand at z={z:.4f}'
                      if drop is not None and z is not None else '') + ' ===', flush=True)
                cls.truth, cls.drop, cls.drop_z, cls.index, cls.bin_pose = [s['x'], s['y']], drop, z, s['index'], None
                cls.stacks_state, cls.stacks_path, cls.slot = state, path, s
                if drop is None:                                       # press mode: search a few rings of studs around the aim
                    cls.wide, cls.rings = True, int(c.get('place_row_search_rings', 2))
                place = ['--place', pd.fmt(s['x']), pd.fmt(s['y']), '--place-layers', str(s['layer']), '--place-yaw', '0']
            sys.argv = [old_argv[0]] + base + pick + place + ['--no-inspect', '--no-step']
            code = pp.main()
            if code:
                return code
            done += 1
        return 0
    except KeyboardInterrupt:
        print('\nCANCELLED (Ctrl+C). The arm stays where it is; the hand was not opened or closed automatically.')
        return 130
    except Exception as problem:
        counts = ', '.join(f'{k} {len(v["placed"])}' for k, v in state['stacks'].items()) or 'none'
        print(f'\nSTACKING STOPPED at brick {done + 1} of {len(jobs)}: {problem}\nNothing further was commanded. The arm stays where it '
              'is and the hand was not opened or closed automatically; a brick that is in the hand is still there. '
              f'Bricks on the stacks: {counts} (see --status).', file=sys.stderr, flush=True)
        return 1
    finally:
        builtins.input, pp.Mission, sys.argv, pp.lookup_hand = old_input, old_mission, old_argv, old_lookup


if __name__ == '__main__':
    sys.exit(main())
