#!/usr/bin/python3 -s
"""Measure the permanent grasp-centring offset (tcp_correction_hand) from your inspection-stop corrections.

How to use it
  1. Run a few picks with the inspection stop (the default on the real robot). At `correction>`, nudge the
     hand with x / y / yaw until the fingers look centred on the brick (a scale or a ruler helps), then `ok`.
     Do it at least 3 times, better 5, with the brick in different places and angles.
  2. python3 src/fr3_pose_pick/scripts/measure_grasp_offset.py --last 5
     It reads pose_pick_offsets.json, turns every correction into the hand's own frame (so the angle of the
     brick does not matter), averages them and prints the line for config/pose_pick.yaml:
         tcp_correction_hand: [along, across]
     --apply CONFIG writes that line into the file for you (the old file is kept as CONFIG.bak).
  3. Run again: the fingers should now come down centred. Any correction you still type is added on top of
     the value in force, so you can repeat the measurement to refine it.

A run where you only pressed `ok` without looking counts as "no correction needed", so use --last to
take only the runs where you really checked.
"""
import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path


# ------------------------------------------------------------------ pure logic (no ROS)
def base_to_hand(dx, dy, yaw):
    """A base-frame vector (dx, dy) as (along, across) in the hand frame of a hand-down hand turned by `yaw`.
    Hand x is (cos, sin) and hand y is (sin, -cos) in the base frame (inverse of smooth_motion.hand_frame_to_base)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c + dy * s, dx * s - dy * c


def implied_correction(entry):
    """(along, across) in metres: the total hand-frame correction this pick turned out to need = the config value
    that was in force + the correction typed at the inspection stop. None for runs logged without a hand yaw."""
    if entry.get('hand_yaw_deg') is None or 'correction_dx_dy_m' not in entry:
        return None
    dx, dy = entry['correction_dx_dy_m']
    along, across = base_to_hand(dx, dy, math.radians(entry['hand_yaw_deg']))
    used = entry.get('tcp_correction_used') or [0.0, 0.0]
    return used[0] + along, used[1] + across


def summarise(entries):
    """Mean and spread of the implied corrections. Returns None when no usable run."""
    values = [v for v in (implied_correction(e) for e in entries) if v is not None]
    if not values:
        return None
    n = len(values)
    mean = [sum(v[i] for v in values) / n for i in range(2)]
    std = [math.sqrt(sum((v[i] - mean[i]) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0 for i in range(2)]
    yaws = [e['correction_yaw_deg'] for e in entries if e.get('hand_yaw_deg') is not None and 'correction_yaw_deg' in e]
    return {'n': n, 'mean': mean, 'std': std, 'yaw_mean_deg': sum(yaws) / len(yaws) if yaws else 0.0}


def config_line(mean):
    return f'tcp_correction_hand: [{mean[0]:.4f}, {mean[1]:.4f}]'


def apply_to_config(path, mean):
    """Replace the tcp_correction_hand line in the YAML file (keeping a .bak copy). Returns True if replaced."""
    path = Path(path)
    text = path.read_text()
    new, count = re.subn(r'^tcp_correction_hand:.*$', config_line(mean), text, flags=re.MULTILINE)
    if count != 1:
        return False
    shutil.copy2(path, str(path) + '.bak')
    path.write_text(new)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log', default='pose_pick_offsets.json', help='the log pose_pick.py writes (default: ./%(default)s)')
    parser.add_argument('--last', type=int, default=5, help='use only the last N runs (default 5)')
    parser.add_argument('--apply', metavar='CONFIG', help='write the result into this YAML file (keeps CONFIG.bak)')
    args = parser.parse_args()
    try:
        history = json.loads(Path(args.log).read_text())
    except (OSError, ValueError) as problem:
        sys.exit(f'Cannot read {args.log}: {problem}. Run from the folder where pose_pick.py was started.')
    usable = [e for e in history if e.get('hand_yaw_deg') is not None][-max(args.last, 1):]
    result = summarise(usable)
    if result is None:
        sys.exit('No usable runs in the log yet. Only runs made after the centring update record the hand yaw: '
                 'do a few picks with the inspection stop first.')
    along, across = result['mean']
    sa, sc = result['std']
    print(f'Runs used: {result["n"]} (the last {args.last} that have a hand yaw)')
    print(f'Along the long side : {along * 1000:+6.1f} mm   (spread {sa * 1000:.1f} mm)')
    print(f'Across the long side: {across * 1000:+6.1f} mm   (spread {sc * 1000:.1f} mm)')
    print(f'At hand yaw 0 that means: base x {along * 1000:+.1f} mm (positive = away from the robot), '
          f'base y {-across * 1000:+.1f} mm (positive = to its left)')
    print(f'Average yaw correction you typed: {result["yaw_mean_deg"]:+.1f} deg (not applied: a steady angle offset is '
          'a different problem)')
    if result['n'] < 3:
        print('WARNING: fewer than 3 runs: one pick is only a rough guess. Do more.')
    if max(sa, sc) > 0.0015:
        print('WARNING: the runs disagree by more than 1.5 mm. The offset may depend on where the brick is, or your '
              'corrections were not consistent. Check before storing it.')
    print('\nPut this in src/fr3_pose_pick/config/pose_pick.yaml:\n  ' + config_line(result['mean']))
    if args.apply:
        if abs(along) > 0.020 or abs(across) > 0.020:
            sys.exit('Refusing to write a correction over 20 mm: that is not a small centring offset.')
        print('written to ' + args.apply + ' (old file kept as .bak)' if apply_to_config(args.apply, result['mean'])
              else 'Could not find a single tcp_correction_hand line in ' + args.apply)


if __name__ == '__main__':
    main()
