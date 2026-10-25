#!/usr/bin/env python3
"""Combined single-brick pick/lift/place/retreat; preview unless an execution flag is supplied.

Imports Picker behavior from pick.py via place.py; no duplicated picking logic.
"""
from pathlib import Path

import yaml

from pick import run_cli
from place import PickPlaceWorkflow, PlacePicker, PlaceTarget


def configure_parser(parser):
    parser.add_argument('--place-target', required=True, type=Path,
                        help='Desired block CENTER in fr3_link0, plus release/approach clearances')


def prepare_request(picker, args):
    picker.phase = 'empty'
    picker.failed_place_approaches = []
    picker.selected_place_approach = None
    picker.place_target = PlaceTarget.from_config(yaml.safe_load(args.place_target.read_text()), picker.c)
    # Legacy place TCP geometry remains internally consistent with the requested
    # block center. Execution uses the actual captured grip to calculate TCP.
    picker.c['place'] = [a-b for a, b in zip(picker.place_target.center,
                                           picker.c['block_center_offset_world'])]
    picker.log_event('placement_request', block_center_xyz_m=picker.place_target.center,
                     release_clearance_m=picker.place_target.release_clearance,
                     approach_clearance_m=picker.place_target.approach_clearance)


def main():
    return run_cli(picker_class=PlacePicker, workflow_class=PickPlaceWorkflow,
                   node_name='pick_place_final', description=__doc__,
                   configure_parser=configure_parser, prepare_request=prepare_request)


if __name__ == '__main__':
    raise SystemExit(main())
