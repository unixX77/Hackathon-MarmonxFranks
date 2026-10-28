#!/usr/bin/python3 -s
"""Look at and manage the pallet (no robot, no ROS needed).

    pallet_tool.py status          the grid, which slots are filled, which one is next
    pallet_tool.py check           the geometry checks (inside the allowed window, gaps, room for the fingers)
    pallet_tool.py mark 3          record that slot 3 holds a brick (placed by hand, or by an earlier run)
    pallet_tool.py unmark 3        record that slot 3 is free again (you took the brick off)
    pallet_tool.py skip            leave the next free slot empty (a bad spot) and move on to the one after
    pallet_tool.py reset           everything is free again (the baseplate is empty)

Slots are numbered from 1. --pallet FILE and --config FILE default to the ones next to this script's package.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pallet as pl                                            # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['status', 'check', 'mark', 'unmark', 'skip', 'reset'])
    parser.add_argument('slot', nargs='?', type=int, help='slot number for mark / unmark (from 1)')
    parser.add_argument('--pallet', default=str(CONFIG_DIR / 'pallet.yaml'))
    parser.add_argument('--config', default=str(CONFIG_DIR / 'pose_pick.yaml'))
    args = parser.parse_args()
    try:
        p = pl.load_pallet(args.pallet)
        state = pl.PalletState(pl.state_path(p, args.pallet))
        if args.command == 'check':
            c = yaml.safe_load(open(args.config, encoding='utf-8'))
            for note in pl.check_pallet(p, c):
                print(note)
            print('Pallet geometry OK.')
        elif args.command in ('mark', 'unmark'):
            if args.slot is None:
                parser.error(f'{args.command} needs a slot number')
            index = args.slot - 1
            pl.slot_row_place(p, index)
            if args.command == 'mark':
                state.mark(index, 'marked', pl.slot_xy(p, index))
            else:
                state.unmark(index)
        elif args.command == 'skip':
            index = state.next_free(p)
            if index is None:
                raise ValueError('The pallet is full: nothing to skip')
            state.mark(index, 'skipped', pl.slot_xy(p, index))
            print(f'Skipped {pl.slot_label(p, index)}.')
        elif args.command == 'reset':
            state.reset()
            print('All slots are free again.')
        nxt = state.next_free(p)
        print('\n'.join(pl.render(p, state, nxt)))
        if nxt is not None:
            x, y = pl.slot_xy(p, nxt)
            print(f'Next: {pl.slot_label(p, nxt)} at x={x:.3f} y={y:.3f}')
        else:
            print('The pallet is full.')
        print(f'State file: {state.path}')
    except ValueError as problem:
        sys.exit(f'ERROR: {problem}')


if __name__ == '__main__':
    main()
