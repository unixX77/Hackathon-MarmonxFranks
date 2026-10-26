#!/usr/bin/python3 -s
"""AUTONOMOUS pose-driven pick and place: the arm goes, picks the brick and puts it down. No question, no correction.

This is the hands-off version of pose_pick.py, which stays the interactive one (inspection stops, step mode,
corrections). Both share the same planning, geometry checks and motion code and the same config, so every value you
locked in config/pose_pick.yaml (grasp height, centring, place height, grasp force, baseplate) is used as it is.
What is different here:

  * It asks nothing. The "Type GO" question is a countdown (default 5 s) in which Ctrl+C or the e-stop cancels, and
    the inspection stops, step mode and the correction prompts are always off. It refuses --step, --inspect and
    --place-inspect, and it answers no prompt at all: a question that still shows up stops the run before it moves.
  * It sets the brick down where it MEASURABLY must be, not where the arm was told to go. The arm lands 1 to 2 mm off its
    target, by an amount that depends on the approach and on where it is, and a brick needs about 0.5 mm to drop onto its studs.
    So the place comes down in three steps: a stop 4 cm up where it nudges x and y until the measured hand position is right,
    a second stop 8 mm above the seat where it does that again (a short descent cannot drag the hand far), then the press. If
    the brick still sits on the studs it lifts and tries again shifted by 0.7 mm in a small spiral before it gives up.
  * It trusts the gripper, not the pose: after the hand closes and again after the lift it reads the finger width the
    gripper reports (the Franka Hand has no force sensor, its effort reads 0, but the width is exact) and stops when it
    does not fit a held brick: nothing between the fingers, the wrong object, or a brick that slipped or fell.
  * Every safety check of pose_pick.py stays: the hand must start pointing down, bad poses and geometry are refused,
    each move is compared with where the hand really is (stop above 10 mm), and a downward move that ends short of
    its target (something is in the way) never closes the hand or releases the brick in the air.
  * Anything that goes wrong STOPS the run where it is and says why (exit code 1). The arm is not moved to "fix" it, the
    hand is not opened or closed by itself, and a brick already in the hand stays in the hand.
  * The result of every run (done or stopped) goes to pose_pick_auto_log.json, not to pose_pick_offsets.json, so these
    runs never dilute measure_grasp_offset.py.

Use it exactly like pose_pick.py (same options), plus --countdown SECONDS (3 to 60, default 5). Where the brick goes:
by default it is put back DOWN IN THE SAME POSE it was picked from (same x, y and angle, and in the same layer: a brick
taken from on top of another goes back on top of it), so the whole run is "pick it up and set it down right there". Give --place, --place-offset, --place-camera-xyz or --pallet to put it
somewhere else instead:

  ros2 run fr3_pose_pick pose_pick_auto.py --config src/fr3_pose_pick/config/pose_pick.yaml \\
      --camera-xyz X Y Z --camera-angle A

With --camera-xyz the picture's hand pose is "the arm right now": run it right after the picture, with the arm still,
or give --ee-pose X Y Z QX QY QZ QW for the pose the picture was taken from.
"""
import builtins
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import pose_pick as pp

AUTO_LOG = 'pose_pick_auto_log.json'
FORBIDDEN = ('--step', '--inspect', '--place-inspect')      # each would bring a question back
PLACE_OPTIONS = ('--place', '--place-offset', '--place-camera-xyz', '--pallet')   # any of these says where it goes
SAME_POSE = ['--place-offset', '0', '0', '--place-layers', 'same']   # the pose AND the layer it was picked from


def place_given(args):
    return any(arg.split('=', 1)[0] in PLACE_OPTIONS for arg in args)


def check_auto_settings(c):
    """The settings only the autonomous scripts read. Raises ValueError before anything moves."""
    transit = c.get('transit_velocity_scale')
    if transit is not None and (isinstance(transit, bool) or not isinstance(transit, (int, float)) or not 0 < transit <= 0.2):
        raise ValueError('transit_velocity_scale must be a fraction of the full speed, above 0 and at most 0.2 (0.1 is 10 %)')
    fine = float(c.get('place_fine_height', 0.008))
    if not 0.004 <= fine < float(c.get('inspect_height', 0.04)):
        raise ValueError('place_fine_height must be from 4 mm (clear of the studs) to below inspect_height (metres: 0.008 is 8 mm)')


class GraspNotConfirmed(RuntimeError):
    """The gripper's finger width does not fit a held brick (missed, slipping or dropped)."""


class AutoMission(pp.Mission):
    """The Mission of pose_pick.py with its own log and a stop-and-explain failure path. Same motions."""

    stage_pick = grasp_pick = truth_target = release_hand = seat_z = release_yaw_deg = None      # filled in by the run
    search_wide = False            # a row brick: look for the studs in rings around the aligned position (see search_pattern)
    drop_height = None             # 'drop' mode (palletizing): let go this far above the seat, no press and no search
    drop_fine = False              # drop mode: still stop 8 mm up and align again before dropping
    drop_z = None                  # drop mode: let go with the hand at this height (m) instead of drop_height above the model's seat
    search_rings = None            # how many rings of place_first_search_step (None: place_first_search_rings, the whole 8 mm cell)
    # moves that are far from the brick and the plate: these run at transit_velocity_scale, the approach and the press at velocity_scale
    TRANSIT = ('Hover above the brick', 'Lift the brick', 'Carry the brick above the place pose', 'Back off straight up',
               'Carry the brick to the bin pose', 'Carry the brick above the bin', 'Back off from the bin')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure = None
        self.stage_pick = None        # measured hand position at the inspection height of the pick
        self.grasp_pick = None        # measured hand position when the hand closed on the brick
        self.truth_target = None      # palletizing: where the hand must really be at the release (x, y), robot coordinates
        self.release_hand = None      # measured hand position at the release
        self.place_trace = {}
        check_auto_settings(self.c)

    def grasp_alert(self, when, problem):
        raise GraspNotConfirmed(f'Grasp not confirmed {when}: {problem}. The brick was not carried on.')

    # ------------------------------------------------------------------------------------- the pick, measured
    def go(self, label, xyz, yaw=None, watch_blocked=False, blocked_height=None):
        """Mission.go, with the far-away moves (TRANSIT) at transit_velocity_scale and, if given, another limit for
        'stopped short of the target'. The slow approach and the press keep velocity_scale."""
        normal = getattr(self.mover, 'scale', None)
        if label in self.TRANSIT and normal is not None and self.c.get('transit_velocity_scale') is not None:
            self.mover.scale = max(normal, float(self.c['transit_velocity_scale']))
        kept = self.c.get('blocked_height')
        if blocked_height is not None:
            self.c['blocked_height'] = blocked_height
        try:
            super().go(label, xyz, yaw, watch_blocked)
        finally:
            if normal is not None:
                self.mover.scale = normal
            if blocked_height is not None:
                if kept is None:
                    self.c.pop('blocked_height', None)
                else:
                    self.c['blocked_height'] = kept
        if label == 'Come down to the inspection height':             # where the hand really is, not where it was sent
            self.stage_pick = np.array(self.mover.hand_pose()[:3, 3], dtype=float)
        elif label == 'Lower to the brick':
            self.grasp_pick = np.array(self.mover.hand_pose()[:3, 3], dtype=float)

    def hand_xy(self):
        self.spin(0.3)
        return np.array(self.mover.hand_pose()[:3, 3][:2], dtype=float)

    def centring_offset(self, yaw):
        """The sideways shift Mover.move adds to every point it is sent to (tcp_correction_hand, turned with the hand)."""
        import smooth_motion as sm
        turn = sm.nearest_yaw(yaw, sm.hand_yaw(self.mover.hand_pose()[:3, :3]))
        return np.array(sm.hand_frame_to_base(self.c.get('tcp_correction_hand') or (0.0, 0.0), turn), dtype=float)

    # ------------------------------------------------------------------------------------ the place, measured
    def place_hand_target(self):
        """Where the hand must REALLY be (x, y) when the brick is let go. Not where the arm is told to go: the arm lands
        1 to 2 mm off its target, by an amount that depends on the approach, and a brick set on studs needs 0.5 mm.
        Put back in the same pose: the measured pose of the hand when it closed on the brick. Put elsewhere (an offset, or
        a place typed in the same frame as the pick): that pose plus the same displacement. Palletizing: the measured
        position of the row (truth_target)."""
        if self.truth_target is not None:
            return np.array(self.truth_target[:2], dtype=float)
        if self.grasp_pick is None:
            raise RuntimeError('there is no measured grasp pose to put the brick back by: it was not picked in this run')
        nominal = np.array(self.place_plan()['centre'][:2], dtype=float)
        return nominal + (self.grasp_pick[:2] - np.array(self.plan['centre'][:2], dtype=float))

    AUTO_KEYS = pp.PLACE_KEYS + ('fine',)

    def aim_plan(self, where, target):
        """Shift the place plan so that the point Mover.move commands is `target` (hand x, y)."""
        centre = np.array(target, dtype=float) - self.centring_offset(where['yaw'])
        shift = centre - np.array(where['centre'][:2], dtype=float)
        pp.apply_nudge(where, float(shift[0]), float(shift[1]), 0.0, 0.0, self.AUTO_KEYS)
        return where

    def pick_drift(self):
        """(how far the straight descent of the pick dragged the hand sideways (x, y), the height that descent covered)."""
        if self.grasp_pick is None or self.stage_pick is None:
            return np.zeros(2), 0.04
        return self.grasp_pick[:2] - self.stage_pick[:2], max(float(self.stage_pick[2] - self.grasp_pick[2]), 0.01)

    def align_at(self, where, key, want, label, tolerance, tries, limit):
        """Nudge the hand, at the stop where[key], until its MEASURED x, y is within `tolerance` of `want`. Returns what is left."""
        size = 0.0
        for attempt in range(tries + 1):
            error = np.array(want, dtype=float) - self.hand_xy()
            size = float(np.hypot(error[0], error[1]))
            self.say(f'  alignment ({label}) {attempt}: the hand is {size * 1000:.2f} mm from where it must be '
                     f'(dx={error[0] * 1000:+.2f} dy={error[1] * 1000:+.2f})')
            if size <= tolerance:
                break
            if size > limit:
                raise RuntimeError(f'The hand is {size * 1000:.1f} mm from where the brick must be set down ({label} stop), more '
                                   f'than allowed ({limit * 1000:.0f} mm, place_align_max / place_fine_max): something is wrong with the pose or the arm. '
                                   'The brick was NOT released.')
            if attempt == tries:
                self.say(f'  alignment ({label}): {size * 1000:.2f} mm left after {tries} corrections; going on')
                break
            pp.apply_nudge(where, float(error[0]), float(error[1]), 0.0, 0.0, self.AUTO_KEYS)
            self.go('Align the brick over its place' + ('' if label == 'coarse' else f' ({label})'), where[key], where['yaw'])
        return size

    SEARCH = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))      # in units of the search step

    def search_pattern(self):
        """(cells, step): the sideways tries for a brick that sits on the studs, in units of step, from the middle outwards.
        The first brick of a row has no known stud grid: it tries the whole 8 mm cell of the studs, ring by ring
        (place_first_search_rings x place_first_search_step each way, 4 x 1 mm = 81 spots); the next row bricks fewer rings
        (place_row_search_rings, 2 = 25 spots), since the row says where the studs are to within a millimetre or so."""
        if self.search_wide:
            rings = int(self.search_rings or self.c.get('place_first_search_rings', 4))
            step = float(self.c.get('place_first_search_step', 0.001))
            cells = sorted(((i, j) for i in range(-rings, rings + 1) for j in range(-rings, rings + 1)),
                           key=lambda cell: (max(abs(cell[0]), abs(cell[1])), math.atan2(cell[1], cell[0]) % (2 * math.pi)))
            return cells, step
        tries = min(int(self.c.get('place_search_tries', 4)), len(self.SEARCH) - 1)
        return list(self.SEARCH[:tries + 1]), float(self.c.get('place_search_step', 0.0007))

    def lower_for_release(self, where):
        """Straight down onto the seat. A brick that stops short (about 1.7 mm = a stud's height) is sitting ON the studs, not in
        them: lift a little, shift to the next spot of the search pattern, and press again."""
        pattern, step = self.search_pattern()
        for k, (ux, uy) in enumerate(pattern):
            try:
                self.go('Lower the brick to the place pose', where['release'], where['yaw'], watch_blocked=True,
                        blocked_height=self.seated_limit())
                self.place_trace['search_tries'] = k
                self.place_trace['search_shift_mm'] = [round(ux * step * 1000, 2), round(uy * step * 1000, 2)]
                if k:
                    self.say(f'The brick went in after {k} shifted tries, {ux * step * 1000:+.1f} mm in x and {uy * step * 1000:+.1f} '
                             'mm in y from the aligned position.')
                return where
            except pp.Blocked as problem:
                self.blocked.append(problem.info)
                if k == len(pattern) - 1:
                    self.say(str(problem))
                    hint = ('No seat within +/-' + f'{max(max(abs(i), abs(j)) for i, j in pattern) * step * 1000:.0f} mm of the '
                            'aligned position: the plate may be turned, or higher or lower than the model. Press a brick onto the plate '
                            'by hand and start the row with --anchor. ' if self.search_wide else '')
                    raise RuntimeError(f'{problem} {hint}The brick was NOT released (it is still in the hand), after '
                                       f'{k + 1} tries.') from None
                nx, ny = pattern[k + 1]
                if self.search_wide:
                    self.say(f'  search {k + 1}/{len(pattern)}: on the studs ({problem.info["short_mm"]:+.1f} mm up) at '
                             f'({ux * step * 1000:+.0f}, {uy * step * 1000:+.0f}) mm; next ({nx * step * 1000:+.0f}, {ny * step * 1000:+.0f}) mm')
                else:
                    self.say(str(problem))
                    self.say(f'The brick sits on top of the studs: lifting and trying again {math.hypot(nx, ny) * step * 1000:.1f} mm '
                             'to the side of the aligned position.')
                pp.apply_nudge(where, (nx - ux) * step, (ny - uy) * step, 0.0, 0.0, self.AUTO_KEYS)
                self.go('Lift and shift to try again', where['fine'], where['yaw'])
        return where

    def lower_to_drop(self, where):
        """Straight down to the drop height. If the controller aborts the move (the brick touched the plate or a neighbour: the 18:36
        run) say what to change instead of only 'execution failed'."""
        try:
            self.go('Lower the brick to the drop height', where['release'], where['yaw'])
        except RuntimeError as problem:
            if 'execution failed' not in str(problem):
                raise
            raise RuntimeError(f'{problem}. The last move stopped before the drop height z={where["release"][2]:.4f}: the brick probably '
                               'touched the plate or a neighbour there. The robot may need its errors recovered in Desk. Raise '
                               'palletize_row_drop_z (or palletize_drop_height) in the config, and check the brick is still in the hand. '
                               'The brick was NOT released.') from None

    def seated_limit(self):
        """How far above its target the hand may end the press. A seated brick stops AT the seat, which is place_margin's
        depth above the target (1 mm for -0.001); a brick resting on the studs is a stud's height (1.7 mm) higher: the limit
        is between the two (it was 2.0 mm before, which let a brick on the studs pass at +2.0)."""
        return max(0.0012, -float(self.c.get('place_margin', 0.0)) + 0.0008)

    def place_brick(self):
        """Set the brick down where it measurably was (or where the row measurably goes): carry it over, come down to the height
        the pick stopped at and align by the measured hand position, come down to a second stop 8 mm above the seat and align
        again (a short descent drags the hand sideways by a few tenths of a mm, a long one by 2 mm and not the same every time),
        press it in, let go, back off."""
        fine_height = float(self.c.get('place_fine_height', 0.008))
        where = self.place_plan()
        if self.drop_height is not None:      # drop mode: hover drop_height above the seat and let go, nothing is pressed in
            seat = float(self.seat_z) if self.seat_z is not None else where['grip'][2]
            height = float(self.drop_z) if self.drop_z is not None else seat + float(self.drop_height)
            pp.apply_nudge(where, 0.0, 0.0, 0.0, height - where['release'][2], self.AUTO_KEYS)
        elif self.seat_z is not None:         # the row's seat height, measured on the anchor brick, not the model's
            lift = float(self.seat_z) + float(self.c.get('place_margin', 0.0)) - where['release'][2]
            if abs(lift) > 0.004:
                raise RuntimeError(f'The measured seat height of the row is {lift * 1000:+.1f} mm from the model: more than 4 mm '
                                   'means the anchor brick or the plate is not what the config says. Nothing was moved.')
            pp.apply_nudge(where, 0.0, 0.0, 0.0, lift, self.AUTO_KEYS)
        where['fine'] = [where['release'][0], where['release'][1], where['release'][2] + fine_height]
        target = self.place_hand_target()
        where = self.aim_plan(where, target)
        drift, span = self.pick_drift()
        ahead = fine_height / span                     # the share of the pick's descent that is still ahead at the fine stop
        self.show_target()
        self.go('Carry the brick above the place pose', where['above'], where['yaw'])
        self.go('Lower the brick to the alignment height', where['inspect'], where['yaw'])
        coarse = self.align_at(where, 'inspect', target - drift, 'coarse', float(self.c.get('place_align_tolerance', 0.001)),
                               int(self.c.get('place_align_tries', 2)), float(self.c.get('place_align_max', 0.008)))
        self.place_trace.update({'drift_mm': [round(float(v) * 1000, 2) for v in drift], 'coarse_left_mm': round(coarse * 1000, 2)})
        if self.drop_height is not None and not self.drop_fine:
            # straight down to drop_height above the seat and let go: whatever the studs do, the brick is down, on to the next one
            self.lower_to_drop(where)
            self.place_trace['drop_height_mm'] = round(float(self.drop_height) * 1000, 2)
        else:
            self.go('Lower the brick to the fine height', where['fine'], where['yaw'])
            fine = self.align_at(where, 'fine', target - drift * ahead, 'fine', float(self.c.get('place_fine_tolerance', 0.0003)),
                                 int(self.c.get('place_fine_tries', 3)), float(self.c.get('place_fine_max', 0.004)))
            self.place_trace['fine_left_mm'] = round(fine * 1000, 2)
            if self.drop_height is not None:
                self.lower_to_drop(where)
                self.place_trace['drop_height_mm'] = round(float(self.drop_height) * 1000, 2)
            else:
                where = self.lower_for_release(where)
        self.release_hand = np.array(self.mover.hand_pose()[:3, 3], dtype=float)
        import smooth_motion as sm
        self.release_yaw_deg = math.degrees(sm.hand_yaw(self.mover.hand_pose()[:3, :3]))
        self.mover.hand()                       # open: the brick stays where it is
        self.mover.detach_block()
        self.say(f'Brick {"dropped" if self.drop_height is not None else "released"} with the hand '
                 f'{np.hypot(*(self.release_hand[:2] - target)) * 1000:.2f} mm from its target')
        if self.on_placed is not None:
            self.on_placed()                    # before the back-off: the slot is taken even if the back-off fails
        self.go('Back off straight up', where['above'], where['yaw'])

    def save_log(self):
        path = Path(self.c.get('auto_log', AUTO_LOG))
        try:
            history = json.loads(path.read_text()) if path.exists() else []
        except ValueError:
            history = []
        entry = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'autonomous': True,
                 'result': 'stopped' if self.failure else 'done', 'failure': self.failure,
                 'pose_given': self.given, 'grip_xyz': self.plan['grip'], 'yaw_deg': math.degrees(self.plan['yaw']),
                 'tcp_correction_used': list(self.c.get('tcp_correction_hand') or [0.0, 0.0]),
                 'grasp_force': self.c.get('grasp_force'), 'blocked': self.blocked,
                 'finger_width_mm': {k: round(v * 1000, 2) for k, v in self.grasp_widths.items()},
                 'pick_stage_xy': None if self.stage_pick is None else [round(float(v), 5) for v in self.stage_pick[:2]],
                 'pick_grasp_xy': None if self.grasp_pick is None else [round(float(v), 5) for v in self.grasp_pick[:2]],
                 'place': getattr(self, 'place_trace', {})}
        if self.place is not None:
            where = self.place_plan()
            entry['place_xyz'], entry['place_yaw_deg'] = where['release'], math.degrees(where['yaw'])
        history.append(entry)
        path.write_text(json.dumps(history, indent=2))
        self.say(f'Run logged in {path}')

    def run(self):
        try:
            super().run()
        except Exception as problem:
            self.failure = f'{type(problem).__name__}: {problem}'
            try:
                self.save_log()
            except Exception as log_problem:             # the original problem is the one to report
                print(f'WARNING: could not write the log ({log_problem})', flush=True)
            raise


def countdown_input(seconds):
    """Replaces input(): the GO question becomes a countdown; any other question is an error, never an answer."""
    def answer(prompt=''):
        if 'Type GO' not in str(prompt):
            raise RuntimeError(f'the autonomous script never asks questions, but something asked {str(prompt)!r}. '
                               'Nothing was answered for you; run pose_pick.py for an interactive run.')
        if seconds <= 0:
            print('AUTONOMOUS: next brick, no countdown (Ctrl+C or the e-stop cancels).', flush=True)
            return 'GO'
        print('AUTONOMOUS: no questions, no corrections. Hold the e-stop, clear the workspace.\n'
              f'The arm starts in {seconds} s; Ctrl+C or the e-stop cancels.', flush=True)
        for left in range(seconds, 0, -1):
            print(f'  {left}...', flush=True)
            time.sleep(1.0)
        print('  go', flush=True)
        return 'GO'
    return answer


def parse_own(argv, default_place=True):
    """(countdown seconds, the rest of the arguments for pose_pick.py). Raises ValueError for a bad combination.
    default_place: with no place option, add 'the same pose and layer' (the palletizing script decides the place itself)."""
    rest, seconds, i = [], 5, 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--countdown' or arg.startswith('--countdown='):
            if arg == '--countdown':
                i += 1
                if i >= len(argv):
                    raise ValueError('--countdown needs a number of seconds (3 to 60)')
                value = argv[i]
            else:
                value = arg.split('=', 1)[1]
            try:
                seconds = int(value)
            except ValueError:
                raise ValueError(f'--countdown must be a whole number of seconds (3 to 60), not {value!r}') from None
            if not 3 <= seconds <= 60:
                raise ValueError('--countdown must be 3 to 60 seconds: time to reach the e-stop, not so long that it is forgotten')
        else:
            rest.append(arg)
        i += 1
    for arg in rest:
        if arg.split('=', 1)[0] in FORBIDDEN:
            raise ValueError(f'{arg.split("=", 1)[0]} brings a question back, and this script never asks: '
                             'use pose_pick.py for an interactive run')
    if default_place and not place_given(rest):
        rest += SAME_POSE if not any(arg.split('=', 1)[0] == '--place-layers' for arg in rest) else SAME_POSE[:3]
        # nothing said: put it back where it was picked, in the layer it was picked from
    return seconds, rest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        print(__doc__ + '\nAll other options are those of pose_pick.py (run it with -h).')
        return 0
    try:
        seconds, rest = parse_own(argv)
    except ValueError as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 2
    old_input, old_mission, old_argv = builtins.input, pp.Mission, sys.argv
    builtins.input, pp.Mission = countdown_input(seconds), AutoMission
    if not place_given(argv):
        print('Place: none given, so the brick is put back down in the same pose it was picked from.', flush=True)
    sys.argv = [old_argv[0]] + rest + ['--no-inspect', '--no-step']
    try:
        return pp.main()
    except KeyboardInterrupt:
        print('\nCANCELLED (Ctrl+C). The arm stays where it is; the hand was not opened or closed automatically.')
        return 130
    except Exception as problem:
        print(f'\nAUTONOMOUS RUN STOPPED: {problem}\n'
              'Nothing further was commanded. The arm stays where it is and the hand was not opened or closed '
              'automatically; a brick that is in the hand is still there.', file=sys.stderr, flush=True)
        return 1
    finally:
        builtins.input, pp.Mission, sys.argv = old_input, old_mission, old_argv


if __name__ == '__main__':
    sys.exit(main())
