#!/usr/bin/python3 -s
"""AUTONOMOUS pose-driven pick and place: the arm goes, picks the brick and puts it down. No question, no correction.

This is the hands-off version of pose_pick.py, which stays the interactive one (inspection stops, step mode,
corrections). Both share the same planning, geometry checks and motion code and the same config, so every value you
locked in config/pose_pick.yaml (grasp height, centring, place height, grasp force, baseplate) is used as it is.
What is different here:

  * It asks nothing. The "Type GO" question is a countdown (default 5 s) in which Ctrl+C or the e-stop cancels, and
    the inspection stops, step mode and the correction prompts are always off. It refuses --step, --inspect and
    --place-inspect, and it answers no prompt at all: a question that still shows up stops the run before it moves.
  * Every safety check of pose_pick.py stays: the hand must start pointing down, bad poses and geometry are refused,
    each move is compared with where the hand really is (stop above 10 mm), and a downward move that ends short of
    its target (something is in the way) never closes the hand or releases the brick in the air.
  * Anything that goes wrong STOPS the run where it is and says why (exit code 1). The arm is not moved to "fix" it, the
    hand is not opened or closed by itself, and a brick already in the hand stays in the hand.
  * The result of every run (done or stopped) goes to pose_pick_auto_log.json, not to pose_pick_offsets.json, so these
    runs never dilute measure_grasp_offset.py.

Use it exactly like pose_pick.py (same options), plus --countdown SECONDS (3 to 60, default 5):

  ros2 run fr3_pose_pick pose_pick_auto.py --config src/fr3_pose_pick/config/pose_pick.yaml \\
      --camera-xyz X Y Z --camera-angle A --place-offset 0 0.05

With --camera-xyz the picture's hand pose is "the arm right now": run it right after the picture, with the arm still,
or give --ee-pose X Y Z QX QY QZ QW for the pose the picture was taken from.
"""
import builtins
import json
import math
import sys
import time
from pathlib import Path

import pose_pick as pp

AUTO_LOG = 'pose_pick_auto_log.json'
FORBIDDEN = ('--step', '--inspect', '--place-inspect')      # each would bring a question back


class AutoMission(pp.Mission):
    """The Mission of pose_pick.py with its own log and a stop-and-explain failure path. Same motions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure = None

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
                 'grasp_force': self.c.get('grasp_force'), 'blocked': self.blocked}
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
        print('AUTONOMOUS: no questions, no corrections. Hold the e-stop, clear the workspace.\n'
              f'The arm starts in {seconds} s; Ctrl+C or the e-stop cancels.', flush=True)
        for left in range(seconds, 0, -1):
            print(f'  {left}...', flush=True)
            time.sleep(1.0)
        print('  go', flush=True)
        return 'GO'
    return answer


def parse_own(argv):
    """(countdown seconds, the rest of the arguments for pose_pick.py). Raises ValueError for a bad combination."""
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
