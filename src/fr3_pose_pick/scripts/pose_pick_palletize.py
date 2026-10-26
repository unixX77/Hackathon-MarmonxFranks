#!/usr/bin/python3 -s
"""AUTONOMOUS incremental palletizing and sorting: pick each brick you give it and drop it into the GOOD row or the BAD row.

Same machinery as pose_pick_auto.py (no questions, no corrections, a countdown instead of GO, every safety check, stop and explain
on any failure, the gripper's finger width confirms every grasp). Each brick is labelled with a capital letter:

  G = good brick   ->  the GOOD row, 40 cm to the robot's left of the stack (palletize_start_offset 0 0.40)
  B = bad brick    ->  the BAD row, the same 40 cm to the left and 10 cm closer to the robot (palletize_bad_start_offset -0.10 0.40)

and goes NEXT TO the one before in its row, along +y, with --gap between them (default 16 mm, so the centres are one brick width
plus the gap = 32 mm apart), at the angle of the first. Both rows are measured from the first brick that was ever picked, so they
line up whichever label comes first. Each row is remembered in a state file (pose_pick_palletize.json for G,
pose_pick_palletize_bad.json for B), so every run, or every --brick, takes the next slot of its row.

HOW THE BRICK IS PUT DOWN (palletize_place_mode in the config, 'drop' by default): it goes over the slot, comes down to 4 cm and
nudges itself onto the right x, y by the measured hand position, goes straight down to palletize_drop_height (1 mm) above the seat,
lets go, and moves on to the next brick. Nothing is pressed onto the studs and nothing is searched, so a brick may rest on top of
its studs; set 'press' for the stud-seating place (the first brick of a row searches the stud cell, see the README).

One brick per command, as the poses arrive from your friend (the label is optional here and means good):

  ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml \\
      --camera-xyz X Y Z --camera-angle A --label G

or several bricks in one go, each ending with G or B (required). Start the command with the arm at the pose the pictures were taken
from: that pose is remembered for every brick (or give it with --ee-pose X Y Z QX QY QZ QW):

  ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml \\
      --brick X1 Y1 Z1 A1 G  --brick X2 Y2 Z2 A2 B  --brick X3 Y3 Z3 A3 G

The other options are those of pose_pick.py, except the place ones (this script decides where each brick goes) and the
questions. Own options:
  --brick X Y Z A G|B    a brick to pick: camera point, angle, and G (good) or B (bad); repeat it for several
  --label G|B            the label of the one brick given with --camera-xyz (default G)
  --start-offset DX DY   where the first GOOD brick goes from the first pick (config palletize_start_offset, 0 0.40)
  --bad-start-offset DX DY   the same for the BAD row (config palletize_bad_start_offset, -0.10 0.40)
  --gap METRES           space between neighbouring bricks (default 0.016); may be changed while at most one brick is down
  --row-yaw DEG          angle of the long side of the placed bricks from base x (default: the angle the first one was picked at)
  --anchor X Y Z A       start the GOOD row from a brick you pressed onto the plate by hand, for the 'press' mode (see the README)
  --status               show both rows, what is placed and where the next ones go (moves nothing)
  --shift DX DY          move the row of --label by this much (metres, at most 1 cm per call; moves nothing)
  --reset                forget both rows and start again
  --state FILE           the GOOD state file (default pose_pick_palletize.json in the current directory); the BAD one is FILE_bad
  --countdown SECONDS    3 to 60, default 5, before the first brick; the next bricks of one command follow at once
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

STATE_FILE = 'pose_pick_palletize.json'
DEFAULT_START_OFFSET = (0.0, 0.08)           # used only when the config has no palletize_start_offset
DEFAULT_BAD_START_OFFSET = (-0.10, 0.40)     # ... or no palletize_bad_start_offset
DEFAULT_GAP = 0.016
LABELS = {'G': 'good', 'B': 'bad'}
PLACE_RELATED = auto.PLACE_OPTIONS + ('--place-layers', '--place-yaw', '--place-camera-angle', '--place-camera-quat',
                                      '--place-is-tcp', '--pallet-file', '--pallet-slot')
PICK_OPTIONS = ('--pose', '--camera-xyz', '--camera-angle', '--camera-quat', '--yaw', '--quat')
MIN_START_DISTANCE = 0.05      # the first slot must be at least this far from the pick: clear of the stack and of the fingers
MIN_GAP = 0.008                # the fingers (and the studs' neighbours) need room between two bricks


# ------------------------------------------------------------------------------------------------- the row
def row_path(path, label):
    """The state file of a row: the given file for the good bricks, the same name with _bad before the extension for the bad."""
    path = Path(path)
    return path if label == 'G' else path.with_name(path.stem + '_bad' + path.suffix)


def new_state(start_offset, gap, row_yaw_deg, width, label='G'):
    return {'version': 1, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'start_offset': [float(v) for v in start_offset],
            'gap': float(gap), 'pitch': float(width + gap), 'row_yaw_deg': float(row_yaw_deg), 'origin': None, 'placed': [],
            'seat_z': None, 'anchored': False, 'row_yaw_fixed': False, 'ref': None, 'label': label}


def load_state(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
        assert state['version'] == 1 and len(state['start_offset']) == 2 and state['pitch'] > 0
        assert state['origin'] is None or len(state['origin']) == 2
        assert isinstance(state['placed'], list)
        state.setdefault('seat_z', None)
        state.setdefault('anchored', False)
        state.setdefault('row_yaw_fixed', False)
        state.setdefault('ref', None)
        state.setdefault('label', 'G')
    except (ValueError, KeyError, TypeError, AssertionError):
        raise ValueError(f'{path} is damaged or not a row file. Look at it, or start again with --reset.') from None
    return state


def save_state(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, path)                      # never a half-written row file


def slot_xy(state, index):
    """Measured centre of slot `index` (from 0), robot coordinates: the first brick's, then one pitch further along +y each."""
    return [state['origin'][0], state['origin'][1] + index * state['pitch']]


def check_row(c, state):
    """Everything that can be checked before a brick moves; returns notes to print. Raises ValueError."""
    import pallet as pl
    length, width = c['block_size'][0], c['block_size'][1]
    notes = []
    if not state.get('anchored') and math.hypot(*state['start_offset']) < MIN_START_DISTANCE:
        raise ValueError(f'--start-offset {state["start_offset"]} is less than {MIN_START_DISTANCE * 100:.0f} cm from the pick: the '
                         'first slot must be clear of the stack and of the fingers (metres: 0.08 is 8 cm)')
    if not MIN_GAP <= state['gap'] <= 0.1:
        raise ValueError(f'--gap {state["gap"]} must be {MIN_GAP * 1000:.0f} mm to 10 cm (it is in metres: 0.016 is 16 mm)')
    d = (0.0, state['pitch'])
    tip_t, tip_w = c.get('finger_tip_thickness'), c.get('finger_tip_width')
    if tip_t is None or tip_w is None:
        notes.append('NOTE: finger_tip_thickness / finger_tip_width are not set in the config, so the room for the fingers between '
                     f'two bricks ({state["gap"] * 1000:.0f} mm apart) was NOT checked.')
    else:
        open_clearance = (c['open_width'] - c.get('hand_zero_width', 0.0) - width) / 2
        if pl.finger_conflict(d, length, width, state['row_yaw_deg'], open_clearance, tip_t, tip_w):
            raise ValueError(f'A finger tip ({tip_t * 1000:.1f} mm thick) would hit the brick next to it when a brick is released '
                             f'{state["gap"] * 1000:.0f} mm from its neighbour: increase --gap.')
    return notes


def check_mode(c):
    """palletize_place_mode / palletize_drop_height must make sense. Returns the drop height (m) or None for the press mode."""
    mode = c.get('palletize_place_mode', 'drop')
    if mode not in ('drop', 'press'):
        raise ValueError("palletize_place_mode must be 'drop' (let go just above the seat, nothing pressed) or 'press' (seat the brick)")
    if mode == 'press':
        return None
    height = c.get('palletize_drop_height', 0.001)
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not 0.0 <= height <= 0.01:
        raise ValueError('palletize_drop_height is in METRES, 0 to 1 cm (0.001 is 1 mm above the seat)')
    return float(height)


def status_lines(state, c=None):
    path_note = f'gap {state["gap"] * 1000:.0f} mm, centres {state["pitch"] * 1000:.0f} mm apart, long side at {state["row_yaw_deg"]:.1f} deg'
    if state.get('anchored'):
        path_note += f', measured from the anchor brick, seat height z={state["seat_z"]:.4f}'
    if state['origin'] is None:
        return [f'Row: nothing placed yet ({path_note}).',
                f'The first brick goes {state["start_offset"][0] * 100:+.1f} cm in x and {state["start_offset"][1] * 100:+.1f} cm in y from where it is picked.']
    placed = len(state['placed'])
    lines = [f'Row: {placed} brick{"s" if placed != 1 else ""} placed ({path_note}).']
    for index in range(placed + 1):
        x, y = slot_xy(state, index)
        status = f'placed {state["placed"][index]["time"][-8:]}' if index < placed else 'NEXT'
        lines.append(f'  slot {index + 1:2d}  x={x:.4f} y={y:.4f}   {status}')
    if c is not None:
        limit = c['workspace_y'][1] - c['block_size'][1] / 2 - 0.002
        room = int((limit - slot_xy(state, placed)[1]) // state['pitch']) + 1
        lines.append(f'  room for {max(room, 0)} more brick{"s" if room != 1 else ""} before y reaches the workspace limit {c["workspace_y"][1]}')
    return lines


# ----------------------------------------------------------------------------------------------- the mission
class PalletizeMission(auto.AutoMission):
    """AutoMission whose place is the next slot of the row of its label. main() sets the class attributes before each brick."""
    state = state_path = truth = seat_z = ref = None
    index = 0
    label = 'G'
    drop = None                                          # drop mode: the height above the seat (m); None: press the brick in
    anchor = False                                       # this brick is the anchor: it only defines the row
    wide = False                                         # press mode: search the studs in rings around the aim
    rings = None                                         # how many rings (None: the whole cell, for the first brick)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls = type(self)
        self.truth_target = cls.truth                    # None for the very first brick: it is set down relative to its own pick
        self.seat_z = cls.seat_z                         # the row's measured seat height
        self.drop_height = cls.drop
        self.drop_fine = bool(self.c.get('palletize_drop_fine', False))
        self.search_wide = cls.wide
        self.search_rings = cls.rings
        self.on_placed = self.record_slot

    def record_slot(self):
        cls = type(self)
        hand = self.release_hand
        if cls.state.get('ref') is None:                 # the reference of the whole layout: where the very first brick was picked
            first = cls.ref if cls.ref is not None else self.grasp_pick
            cls.state['ref'] = [round(float(first[0]), 5), round(float(first[1]), 5)]
        if cls.anchor:
            # The anchor sat seated on the studs when the hand closed on it: that hand position, angle and height ARE the row.
            hand = self.grasp_pick
            yaw = ((float(self.grasp_hand_yaw_deg) + 90.0) % 180.0) - 90.0        # a line: -90 to 90 deg
            cls.state.update(origin=[round(float(hand[0]), 5), round(float(hand[1]), 5)], row_yaw_deg=round(yaw, 3),
                             seat_z=round(float(hand[2]), 5), anchored=True)
        elif cls.state['origin'] is None:
            # The first brick defines the row, wherever it ended up: the hand's position, angle and height.
            cls.state['origin'] = [round(float(hand[0]), 5), round(float(hand[1]), 5)]
            cls.state['seat_z'] = round(float(hand[2]), 5)
            if not cls.state.get('row_yaw_fixed') and self.release_yaw_deg is not None:
                cls.state['row_yaw_deg'] = round(((float(self.release_yaw_deg) + 90.0) % 180.0) - 90.0, 3)
        elif cls.drop is None and not cls.state.get('anchored') and cls.index > 0:
            # Press mode: every brick that seats is another measurement of where the studs are (the hand can be up to ~0.5 mm off
            # the grid point and the brick still drops in): the origin becomes the mean of them all.
            n = len(cls.state['placed'])
            sample = [float(hand[0]), float(hand[1]) - cls.index * cls.state['pitch']]
            cls.state['origin'] = [round((cls.state['origin'][k] * n + sample[k]) / (n + 1), 5) for k in range(2)]
        cls.state['placed'].append({'slot': cls.index + 1, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'label': cls.label,
                                    'released_hand_xy': [round(float(hand[0]), 5), round(float(hand[1]), 5)]})
        save_state(cls.state_path, cls.state)
        x, y = slot_xy(cls.state, cls.index + 1)
        self.say(f'[row {cls.label} {LABELS[cls.label]}] brick {cls.index + 1} is down. The next {LABELS[cls.label]} brick goes to '
                 f'slot {cls.index + 2} at x={x:.4f} y={y:.4f}')

    def save_log(self):
        self.place_trace['slot'] = type(self).index + 1
        self.place_trace['label'] = type(self).label
        super().save_log()


# ---------------------------------------------------------------------------------------------------- main
def own_parser():
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    own.add_argument('--config')
    own.add_argument('--state', default=STATE_FILE)
    own.add_argument('--start-offset', nargs=2, type=float, metavar=('DX', 'DY'))
    own.add_argument('--bad-start-offset', nargs=2, type=float, metavar=('DX', 'DY'))
    own.add_argument('--gap', type=float)
    own.add_argument('--row-yaw', type=float)
    own.add_argument('--label', type=str.upper, choices=sorted(LABELS))
    own.add_argument('--anchor', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'ANGLE'))
    own.add_argument('--shift', nargs=2, type=float, metavar=('DX', 'DY'))
    own.add_argument('--reset', action='store_true')
    own.add_argument('--status', action='store_true')
    return own


def parse_brick(values):
    """['0.09', '-0.04', '0.26', '-86.4', 'B'] -> (x, y, z, angle, 'B'). The label is required: a forgotten one must not sort silently."""
    text = ' '.join(values)
    if len(values) != 5:
        raise ValueError(f'--brick {text}: it needs x y z angle AND a label, G (good) or B (bad), e.g. --brick 0.09 -0.04 0.26 -86.4 G')
    try:
        x, y, z, angle = (float(v) for v in values[:4])
    except ValueError:
        raise ValueError(f'--brick {text}: the first four values are numbers (x, y, z in metres, the angle in degrees)') from None
    label = values[4].upper()
    if label not in LABELS:
        raise ValueError(f'--brick {text}: the last value must be G (good) or B (bad), not {values[4]!r}')
    return x, y, z, angle, label


def extract_bricks(argv):
    """Take every --brick X Y Z A G|B out of the arguments: ([(x, y, z, angle, label), ...], the other arguments)."""
    bricks, rest, i = [], [], 0
    while i < len(argv):
        if argv[i] == '--brick':
            values, i = [], i + 1
            while i < len(argv) and not argv[i].startswith('--'):
                values.append(argv[i])
                i += 1
            bricks.append(parse_brick(values))
            continue
        rest.append(argv[i])
        i += 1
    return bricks, rest


def fmt(value):
    return f'{value:.6f}'


def prepare_row(label, args, c, path, anchoring=False):
    """The state of the row of `label`: the saved one (its geometry cannot change, except the gap while at most one brick is
    down) or a new one from the options and the config."""
    state = load_state(path)
    start = (args.start_offset if label == 'G' else args.bad_start_offset)
    if state is not None and not state['placed'] and anchoring:
        state = None                                      # an empty row file is just a plan: the anchor decides
    if state is not None:
        for name, given, kept in (('--start-offset' if label == 'G' else '--bad-start-offset', start, state['start_offset']),
                                  ('--row-yaw', args.row_yaw, state['row_yaw_deg'])):
            if given is not None and list(_flat(given)) != list(_flat(kept)):
                raise ValueError(f'The {LABELS[label]} row was started with {name} {kept}: it cannot change in the middle. '
                                 'Use --reset for a new row.')
        if args.gap is not None and abs(args.gap - state['gap']) > 1e-9:
            if len(state['placed']) > 1:
                raise ValueError(f'The {LABELS[label]} row has {len(state["placed"])} bricks {state["gap"] * 1000:.0f} mm apart: the '
                                 'gap cannot change in the middle. Use --reset for a new row.')
            state['gap'], state['pitch'] = float(args.gap), float(c['block_size'][1] + args.gap)     # only the next slots move
            save_state(path, state)
            print(f'Gap changed to {args.gap * 1000:.1f} mm (centres {state["pitch"] * 1000:.1f} mm apart): only '
                  f'{len(state["placed"])} brick is down, so only the next slots move.', flush=True)
        return state
    default = c.get('palletize_start_offset') or DEFAULT_START_OFFSET if label == 'G' else \
        c.get('palletize_bad_start_offset') or DEFAULT_BAD_START_OFFSET
    state = new_state(start or default, args.gap if args.gap is not None else DEFAULT_GAP,
                      args.row_yaw if args.row_yaw is not None else 0.0, c['block_size'][1], label)
    state['row_yaw_fixed'] = args.row_yaw is not None
    return state


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
            if arg.split('=', 1)[0] in PLACE_RELATED:
                raise ValueError(f'{arg.split("=", 1)[0]}: this script decides where each brick goes; use pose_pick_auto.py or '
                                 'pose_pick.py to choose the place yourself')
        good_path = Path(args.state)
        paths = {'G': good_path, 'B': row_path(good_path, 'B')}
        if args.reset:
            for label, path in paths.items():
                if path.exists():
                    path.unlink()
                    print(f'{LABELS[label].capitalize()} row forgotten ({path} removed).')
        if args.reset and not bricks and not args.anchor and not any(arg.split('=', 1)[0] in PICK_OPTIONS for arg in rest):
            return 0                                           # only a reset was asked for
        c = None
        if args.config:
            import yaml
            c = yaml.safe_load(open(args.config, encoding='utf-8'))
        if args.status or args.shift:
            shift_label = args.label or 'G'
            states = {label: load_state(path) for label, path in paths.items()}
            if args.shift:
                state = states[shift_label]
                if state is None or state['origin'] is None:
                    raise ValueError(f'--shift moves a row that has a first brick; the {LABELS[shift_label]} row has none yet')
                if max(abs(args.shift[0]), abs(args.shift[1])) > 0.01:
                    raise ValueError('--shift is in metres and at most 1 cm per call (0.001 is 1 mm)')
                state['origin'] = [state['origin'][0] + args.shift[0], state['origin'][1] + args.shift[1]]
                save_state(paths[shift_label], state)
                print(f'The {LABELS[shift_label]} row moved by dx={args.shift[0] * 1000:+.1f} mm dy={args.shift[1] * 1000:+.1f} mm.')
            if states['G'] is None and states['B'] is None:
                print('Row: nothing placed yet (no state file).')
            for label in ('G', 'B'):
                if states[label] is not None:
                    print(f'--- {label}: {LABELS[label].upper()} bricks')
                    print('\n'.join(status_lines(states[label], c)))
                elif label == 'B' and states['G'] is not None:
                    print('--- B: BAD bricks\nRow: nothing placed yet.')
            return 0
        if c is None:
            raise ValueError('--config is needed')
        drop = check_mode(c)
        jobs = []                                              # (the pick options, the label)
        if args.anchor:
            if bricks or any(arg.split('=', 1)[0] in PICK_OPTIONS for arg in rest):
                raise ValueError('--anchor is the brick: leave out --brick / --camera-xyz / --camera-angle / --pose')
            if args.start_offset is not None or args.row_yaw is not None:
                raise ValueError('--anchor takes the position and the angle of the row from the brick: leave out --start-offset / --row-yaw')
            if (args.label or 'G') != 'G':
                raise ValueError('--anchor starts the GOOD row (label G)')
            existing = load_state(paths['G'])
            if existing is not None and existing['placed']:
                raise ValueError(f'The good row already has {len(existing["placed"])} brick(s): --anchor measures a NEW row from a brick '
                                 'you pressed onto the plate by hand. Use --reset first.')
            x, y, z, a = args.anchor
            jobs = [(['--camera-xyz', fmt(x), fmt(y), fmt(z), '--camera-angle', fmt(a)], 'G')]
        elif bricks:
            if args.label is not None or any(arg.split('=', 1)[0] in PICK_OPTIONS for arg in rest):
                raise ValueError('--brick replaces --camera-xyz / --camera-angle / --pose / --label: use one way to give the bricks')
            jobs = [(['--camera-xyz', fmt(x), fmt(y), fmt(z), '--camera-angle', fmt(a)], label) for x, y, z, a, label in bricks]
        else:
            jobs = [([], args.label or 'G')]
        states, notes = {}, []
        for label in sorted({label for _, label in jobs}):
            states[label] = prepare_row(label, args, c, paths[label], anchoring=bool(args.anchor) and label == 'G')
            notes += [n for n in check_row(c, states[label]) if n not in notes]
    except (ValueError, OSError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 2

    base = ['--config', args.config] + rest
    old_input, old_mission, old_argv, old_lookup = builtins.input, pp.Mission, sys.argv, pp.lookup_hand
    pp.Mission = PalletizeMission
    if len(jobs) > 1 and '--ee-pose' not in rest:
        photo = {}

        def lookup_once(node, buffer, config, seconds=10.0):
            if 'pose' not in photo:                                  # the arm pose at the start of this command ...
                photo['pose'] = old_lookup(node, buffer, config, seconds)
            return photo['pose']                                    # ... is the camera pose of every picture
        pp.lookup_hand = lookup_once
        print('Several bricks: the pictures are taken to be from the arm pose it has now (give --ee-pose to say otherwise).', flush=True)
    if len(jobs) > 1:
        print('Plan: ' + ', '.join(f'{n}={label} ({LABELS[label]})' for n, (_, label) in enumerate(jobs, 1)), flush=True)
    mode = f'drop {drop * 1000:.1f} mm above the seat' if drop is not None else 'press into the studs'
    done = 0
    try:
        for number, (pick, label) in enumerate(jobs, 1):
            path = paths[label]
            state = load_state(path) or states[label]          # the mission saved the row after the last release
            other = load_state(paths['B' if label == 'G' else 'G'])
            ref = state.get('ref') or (other or {}).get('ref')
            index = len(state['placed'])
            is_anchor = bool(args.anchor) and number == 1
            yaw_args = ['--place-yaw', fmt(state['row_yaw_deg'])] if (state['origin'] is not None or state.get('row_yaw_fixed')) else []
            if is_anchor:
                place, yaw_args, truth = list(auto.SAME_POSE), [], None          # picked up and put back where it is
                where = 'the anchor brick: picked up and put back where it is; the row is measured from it'
            elif state['origin'] is None:
                dx, dy = state['start_offset']
                if ref is None:                                # the very first brick: relative to its own pick, which becomes the reference
                    place, truth = ['--place-offset', fmt(dx), fmt(dy)], None
                    where = f'{dx * 100:+.1f} cm x, {dy * 100:+.1f} cm y from its pick'
                else:                                          # the other row exists: both rows hang from the same reference
                    truth = [ref[0] + dx, ref[1] + dy]
                    place = ['--place', fmt(truth[0]), fmt(truth[1])]
                    where = f'x={truth[0]:.4f} y={truth[1]:.4f} ({dx * 100:+.1f} cm x, {dy * 100:+.1f} cm y from the first pick)'
            else:
                truth = slot_xy(state, index)
                place = ['--place', fmt(truth[0]), fmt(truth[1])]
                where = f'x={truth[0]:.4f} y={truth[1]:.4f}'
            print(f'\n=== Brick {number} of {len(jobs)} [{label} = {LABELS[label].upper()}]: {LABELS[label]} row slot {index + 1}, '
                  f'{where}; {"anchor" if is_anchor else mode} ===', flush=True)
            if number == 1 and notes:
                print('\n'.join(notes), flush=True)
            PalletizeMission.state, PalletizeMission.state_path, PalletizeMission.index, PalletizeMission.truth = \
                state, path, index, truth
            PalletizeMission.anchor, PalletizeMission.seat_z, PalletizeMission.ref = is_anchor, state.get('seat_z'), ref
            PalletizeMission.label = label
            PalletizeMission.drop = None if is_anchor else drop
            PalletizeMission.wide = drop is None and not is_anchor
            PalletizeMission.rings = None if state['origin'] is None else int(c.get('place_row_search_rings', 2))
            builtins.input = auto.countdown_input(seconds if number == 1 else 0)         # one countdown per command
            sys.argv = [old_argv[0]] + base + pick + place + yaw_args + ['--no-inspect', '--no-step']
            code = pp.main()
            if code:
                return code
            done += 1
        return 0
    except KeyboardInterrupt:
        print('\nCANCELLED (Ctrl+C). The arm stays where it is; the hand was not opened or closed automatically.')
        return 130
    except Exception as problem:
        rows = ', '.join(f'{LABELS[k]} {len(load_state(p)["placed"]) if load_state(p) else 0}' for k, p in paths.items())
        print(f'\nPALLETIZING STOPPED at brick {done + 1} of {len(jobs)}: {problem}\n'
              'Nothing further was commanded. The arm stays where it is and the hand was not opened or closed automatically; '
              f'a brick that is in the hand is still there. Bricks already down: {rows} (see --status).', file=sys.stderr, flush=True)
        return 1
    finally:
        builtins.input, pp.Mission, sys.argv, pp.lookup_hand = old_input, old_mission, old_argv, old_lookup


def _flat(value):
    return [round(float(v), 9) for v in (value if isinstance(value, (list, tuple)) else [value])]


if __name__ == '__main__':
    sys.exit(main())
