#!/usr/bin/python3 -s
"""AUTONOMOUS incremental palletizing: pick each brick you give it and set it down in a row, one brick after the other.

Same machinery as pose_pick_auto.py (no questions, no corrections, the countdown instead of GO, every safety check, stop
and explain on any failure, the gripper's finger width confirms every grasp, the brick is set down where the hand
MEASURABLY must be and retried 0.7 mm to the side if it sits on top of the studs). What is different is WHERE the brick goes:

  * the first brick is set down --start-offset away from where it was picked (default: palletize_start_offset in the config,
    0 +0.098: about 10 cm to the robot's left, along +y, clear of the stack it came from), on the table / baseplate;
  * every next brick goes NEXT TO the one before, along +y, with --gap between them (default 16 mm, so the centres are one
    brick width plus the gap = 32 mm apart), with the long side along x (--row-yaw 0);
  * the row is remembered in a state file (pose_pick_palletize.json), so each run, or each --brick, takes the next slot.
    The slots are measured, not commanded: slot 1 is wherever the first brick really ended up (the hand position at the
    release), the others are exact multiples of the pitch from it.

THE ROW MUST LAND ON THE STUDS. A brick drops onto its studs only if it is within about 0.5 mm of them (they repeat every 8 mm)
and turned the same way, and the robot cannot see the studs of the plate the row goes on. So the best way to start a row is an
ANCHOR: press one brick by hand onto the plate where the row should begin, then

  ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml --anchor X Y Z A

(the camera numbers of that brick). The robot picks it up and puts it back, which it can do reliably, and measures from that: the
row starts at the anchor's position, with its angle and the height of its seat; the anchor is slot 1 and the next bricks follow
beside it. Without an anchor the first brick goes --start-offset from its pick and may sit on top of the studs.

One brick per command, as the poses arrive from your friend:

  ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml \\
      --camera-xyz X Y Z --camera-angle A

or several bricks in one go. Start the command with the arm at the pose the pictures were taken from: that pose is
remembered for every brick (or give it with --ee-pose X Y Z QX QY QZ QW):

  ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml \\
      --brick X1 Y1 Z1 A1  --brick X2 Y2 Z2 A2  --brick X3 Y3 Z3 A3

The other options are those of pose_pick.py, except the place ones (this script decides where each brick goes) and the
questions. Own options:
  --start-offset DX DY   where the FIRST brick goes, from where it was picked (default: palletize_start_offset in the config,
                         0 0.098); fixed by the first brick
  --gap METRES           space between neighbouring bricks (default 0.016); may be changed while at most one brick is down
  --row-yaw DEG          angle of the long side of the placed bricks from base x (default 0)
  --anchor X Y Z A       start the row from a brick you pressed onto the plate by hand (see above)
  --brick X Y Z A        a brick to pick: camera point and angle (repeat it for several; all from the same picture pose)
  --status               show the row, what is placed and where the next one goes (moves nothing)
  --shift DX DY          move the whole row by this much (metres, at most 1 cm per call; moves nothing)
  --reset                forget the row and start a new one
  --state FILE           the state file (default pose_pick_palletize.json in the current directory)
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
DEFAULT_START_OFFSET = (0.0, 0.08)      # used only when the config has no palletize_start_offset
DEFAULT_GAP = 0.016
PLACE_RELATED = auto.PLACE_OPTIONS + ('--place-layers', '--place-yaw', '--place-camera-angle', '--place-camera-quat',
                                      '--place-is-tcp', '--pallet-file', '--pallet-slot')
PICK_OPTIONS = ('--pose', '--camera-xyz', '--camera-angle', '--camera-quat', '--yaw', '--quat')
MIN_START_DISTANCE = 0.05      # the first slot must be at least this far from the pick: clear of the stack and of the fingers
MIN_GAP = 0.008                # the fingers (and the studs' neighbours) need room between two bricks


# ------------------------------------------------------------------------------------------------- the row
def new_state(start_offset, gap, row_yaw_deg, width):
    return {'version': 1, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'start_offset': [float(v) for v in start_offset],
            'gap': float(gap), 'pitch': float(width + gap), 'row_yaw_deg': float(row_yaw_deg), 'origin': None, 'placed': [],
            'seat_z': None, 'anchored': False}


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
    """AutoMission whose place is the next slot of the row. main() sets the class attributes before each brick."""
    state = state_path = truth = seat_z = None
    index = 0
    anchor = False                                       # this brick is the anchor: it only defines the row

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.truth_target = type(self).truth             # None for the first brick: it is set down relative to its own pick
        self.seat_z = type(self).seat_z                  # the row's measured seat height (anchored rows)
        self.on_placed = self.record_slot

    def record_slot(self):
        cls = type(self)
        hand = self.release_hand
        if cls.anchor:
            # The anchor sat seated on the studs when the hand closed on it: that hand position, angle and height ARE the row.
            hand = self.grasp_pick
            yaw = ((float(self.grasp_hand_yaw_deg) + 90.0) % 180.0) - 90.0        # a line: -90 to 90 deg
            cls.state.update(origin=[round(float(hand[0]), 5), round(float(hand[1]), 5)], row_yaw_deg=round(yaw, 3),
                             seat_z=round(float(hand[2]), 5), anchored=True)
        elif cls.state['origin'] is None:                # the first brick defines the row: where the hand really was
            cls.state['origin'] = [round(float(hand[0]), 5), round(float(hand[1]), 5)]
        cls.state['placed'].append({'slot': cls.index + 1, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'released_hand_xy': [round(float(hand[0]), 5), round(float(hand[1]), 5)]})
        save_state(cls.state_path, cls.state)
        x, y = slot_xy(cls.state, cls.index + 1)
        self.say(f'[row] brick {cls.index + 1} is down. The next one goes to slot {cls.index + 2} at x={x:.4f} y={y:.4f}')

    def save_log(self):
        self.place_trace['slot'] = type(self).index + 1
        super().save_log()


# ---------------------------------------------------------------------------------------------------- main
def own_parser():
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    own.add_argument('--config')
    own.add_argument('--state', default=STATE_FILE)
    own.add_argument('--start-offset', nargs=2, type=float, metavar=('DX', 'DY'))
    own.add_argument('--gap', type=float)
    own.add_argument('--row-yaw', type=float)
    own.add_argument('--brick', nargs=4, type=float, action='append', metavar=('X', 'Y', 'Z', 'ANGLE'))
    own.add_argument('--anchor', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'ANGLE'))
    own.add_argument('--shift', nargs=2, type=float, metavar=('DX', 'DY'))
    own.add_argument('--reset', action='store_true')
    own.add_argument('--status', action='store_true')
    return own


def fmt(value):
    return f'{value:.6f}'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        print(__doc__)
        return 0
    try:
        args, rest = own_parser().parse_known_args(argv)
    except SystemExit:
        return 2
    try:
        seconds, rest = auto.parse_own(rest, default_place=False)
        for arg in rest:
            if arg.split('=', 1)[0] in PLACE_RELATED:
                raise ValueError(f'{arg.split("=", 1)[0]}: this script decides where each brick goes; use pose_pick_auto.py or '
                                 'pose_pick.py to choose the place yourself')
        path = Path(args.state)
        if args.reset and path.exists():
            path.unlink()
            print(f'Row forgotten ({path} removed).')
        if args.reset and not args.brick and not args.anchor and not any(arg.split('=', 1)[0] in PICK_OPTIONS for arg in rest):
            return 0                                           # only a reset was asked for
        state = load_state(path)
        c = None
        if args.config:
            import yaml
            c = yaml.safe_load(open(args.config, encoding='utf-8'))
        if args.status or args.shift:
            if state is None:
                print('Row: nothing placed yet (no state file).')
                return 0
            if args.shift:
                if state['origin'] is None:
                    raise ValueError('--shift moves a row that has a first brick; there is none yet (use --start-offset instead)')
                if max(abs(args.shift[0]), abs(args.shift[1])) > 0.01:
                    raise ValueError('--shift is in metres and at most 1 cm per call (0.001 is 1 mm)')
                state['origin'] = [state['origin'][0] + args.shift[0], state['origin'][1] + args.shift[1]]
                save_state(path, state)
                print(f'Row moved by dx={args.shift[0] * 1000:+.1f} mm dy={args.shift[1] * 1000:+.1f} mm.')
            print('\n'.join(status_lines(state, c)))
            return 0
        if c is None:
            raise ValueError('--config is needed')
        picks = []
        if args.anchor:
            if args.brick or any(arg.split('=', 1)[0] in PICK_OPTIONS for arg in rest):
                raise ValueError('--anchor is the brick: leave out --brick / --camera-xyz / --camera-angle / --pose')
            if args.start_offset is not None or args.row_yaw is not None:
                raise ValueError('--anchor takes the position and the angle of the row from the brick: leave out --start-offset / --row-yaw')
            if state is not None and state['placed']:
                raise ValueError(f'The row already has {len(state["placed"])} brick(s): --anchor measures a NEW row from a brick you '
                                 'pressed onto the plate by hand. Use --reset first.')
            x, y, z, a = args.anchor
            picks = [['--camera-xyz', fmt(x), fmt(y), fmt(z), '--camera-angle', fmt(a)]]
        elif args.brick:
            if any(arg.split('=', 1)[0] in PICK_OPTIONS for arg in rest):
                raise ValueError('--brick replaces --camera-xyz / --camera-angle / --pose: use one way to give the bricks')
            picks = [['--camera-xyz', fmt(x), fmt(y), fmt(z), '--camera-angle', fmt(a)] for x, y, z, a in args.brick]
        else:
            picks = [[]]
        if state is not None and not state['placed'] and args.anchor:
            state = None                                      # an empty row file is just a plan: the anchor decides
        if state is not None:                              # a started row keeps its geometry
            for name, given, kept in (('--start-offset', args.start_offset, state['start_offset']),
                                      ('--row-yaw', args.row_yaw, state['row_yaw_deg'])):
                if given is not None and list(_flat(given)) != list(_flat(kept)):
                    raise ValueError(f'The row was started with {name} {kept}: it cannot change in the middle. Use --reset for a new row.')
            if args.gap is not None and abs(args.gap - state['gap']) > 1e-9:
                if len(state['placed']) > 1:
                    raise ValueError(f'The row has {len(state["placed"])} bricks {state["gap"] * 1000:.0f} mm apart: the gap cannot '
                                     'change in the middle. Use --reset for a new row.')
                state['gap'], state['pitch'] = float(args.gap), float(c['block_size'][1] + args.gap)     # only the next slots move
                save_state(path, state)
                print(f'Gap changed to {args.gap * 1000:.1f} mm (centres {state["pitch"] * 1000:.1f} mm apart): only '
                      f'{len(state["placed"])} brick is down, so only the next slots move.', flush=True)
        else:
            state = new_state(args.start_offset or c.get('palletize_start_offset') or DEFAULT_START_OFFSET,
                              args.gap if args.gap is not None else DEFAULT_GAP,
                              args.row_yaw if args.row_yaw is not None else 0.0, c['block_size'][1])
        notes = check_row(c, state)
    except (ValueError, OSError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 2

    base = ['--config', args.config] + rest
    old_input, old_mission, old_argv, old_lookup = builtins.input, pp.Mission, sys.argv, pp.lookup_hand
    pp.Mission = PalletizeMission
    if len(picks) > 1 and '--ee-pose' not in rest:
        photo = {}

        def lookup_once(node, buffer, config, seconds=10.0):
            if 'pose' not in photo:                                  # the arm pose at the start of this command ...
                photo['pose'] = old_lookup(node, buffer, config, seconds)
            return photo['pose']                                    # ... is the camera pose of every picture
        pp.lookup_hand = lookup_once
        print('Several bricks: the pictures are taken to be from the arm pose it has now (give --ee-pose to say otherwise).', flush=True)
    done = 0
    try:
        for number, pick in enumerate(picks, 1):
            state = load_state(path) or state                  # the mission saved the row after the last release
            index = len(state['placed'])
            is_anchor = bool(args.anchor) and number == 1
            if is_anchor:
                place, yaw_args, truth = list(auto.SAME_POSE), [], None          # picked up and put back where it is
                where = 'the anchor brick: picked up and put back where it is; the row is measured from it'
            elif state['origin'] is None:
                place = ['--place-offset', fmt(state['start_offset'][0]), fmt(state['start_offset'][1])]
                yaw_args, truth = ['--place-yaw', fmt(state['row_yaw_deg'])], None
                where = (f'{state["start_offset"][0] * 100:+.1f} cm x, {state["start_offset"][1] * 100:+.1f} cm y from its pick '
                         '(no anchor: it may sit on top of the studs)')
            else:
                truth = slot_xy(state, index)
                place = ['--place', fmt(truth[0]), fmt(truth[1])]
                yaw_args = ['--place-yaw', fmt(state['row_yaw_deg'])]
                where = f'x={truth[0]:.4f} y={truth[1]:.4f}'
            print(f'\n=== Brick {number} of {len(picks)}: row slot {index + 1}, {where} ===', flush=True)
            if number == 1 and notes:
                print('\n'.join(notes), flush=True)
            PalletizeMission.state, PalletizeMission.state_path, PalletizeMission.index, PalletizeMission.truth = \
                state, path, index, truth
            PalletizeMission.anchor, PalletizeMission.seat_z = is_anchor, state.get('seat_z')
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
        print(f'\nPALLETIZING STOPPED at brick {done + 1} of {len(picks)}: {problem}\n'
              'Nothing further was commanded. The arm stays where it is and the hand was not opened or closed automatically; '
              f'a brick that is in the hand is still there. Bricks already in the row: {len(load_state(path)["placed"]) if load_state(path) else 0} '
              '(see --status).', file=sys.stderr, flush=True)
        return 1
    finally:
        builtins.input, pp.Mission, sys.argv, pp.lookup_hand = old_input, old_mission, old_argv, old_lookup


def _flat(value):
    return [round(float(v), 9) for v in (value if isinstance(value, (list, tuple)) else [value])]


if __name__ == '__main__':
    sys.exit(main())
