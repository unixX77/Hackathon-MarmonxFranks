"""Pallet: the grid, the saved state, the tool, and the tip / baseplate checks."""
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pl = load('pallet')
pp = load('pose_pick')
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))
GOOD = {'origin': [0.36, -0.10], 'step_along': [0.04, 0.0], 'step_row': [0.0, -0.04], 'per_row': 5, 'rows': 3}


def pallet(**kw):
    return pl.normalise({**GOOD, **kw})


# ---------------------------------------------------------------------------- the grid
def test_slots_fill_a_row_then_the_next_row():
    p = pallet()
    assert pl.slot_count(p) == 15
    assert pl.slot_xy(p, 0) == pytest.approx((0.36, -0.10))
    assert pl.slot_xy(p, 1) == pytest.approx((0.40, -0.10))          # +4 cm in x: the next place in the row
    assert pl.slot_xy(p, 4) == pytest.approx((0.52, -0.10))          # the last place of row 1
    assert pl.slot_xy(p, 5) == pytest.approx((0.36, -0.14))          # row 2 starts 4 cm to the robot's right
    assert pl.slot_xy(p, 14) == pytest.approx((0.52, -0.18))
    assert pl.slot_label(p, 5) == 'slot 6 of 15 (row 2, place 1)'
    for bad in (-1, 15):
        with pytest.raises(ValueError, match='does not exist'):
            pl.slot_xy(p, bad)


def test_a_diagonal_pallet_and_other_directions():
    p = pallet(step_along=[0.0, 0.045], step_row=[0.04, 0.0], per_row=3, rows=2)
    assert pl.slot_xy(p, 2) == pytest.approx((0.36, -0.01))
    assert pl.slot_xy(p, 3) == pytest.approx((0.40, -0.10))


def test_bad_pallet_files_are_refused_with_a_clear_message():
    for key in ('origin', 'step_along', 'step_row', 'per_row', 'rows'):
        with pytest.raises(ValueError, match=key):
            pl.normalise({k: v for k, v in GOOD.items() if k != key})
    for bad in (dict(origin=[0.3]), dict(step_along=[0.04, 'a']), dict(per_row=0), dict(rows=2.5), dict(layer=0),
                dict(step_along=[4.0, 0.0]), dict(step_along=[0.0, 0.0]), dict(step_row=[0.0, 0.0]),
                dict(per_row=30, rows=30), dict(area=[0.5, 0.4, 0.0, 0.1])):
        with pytest.raises(ValueError):
            pallet(**bad)
    with pytest.raises(ValueError, match='mapping'):
        pl.normalise([1, 2])


# ------------------------------------------------------------------------- geometry
def test_gap_between_bricks():
    # 32 x 16 mm bricks, long side along x
    assert pl.brick_gap((0.040, 0.0), 0.032, 0.016, 0.0) == pytest.approx(0.008)       # 4 cm along the long side
    assert pl.brick_gap((0.0, 0.040), 0.032, 0.016, 0.0) == pytest.approx(0.024)       # 4 cm across it
    assert pl.brick_gap((0.020, 0.0), 0.032, 0.016, 0.0) < 0                           # overlapping
    assert pl.brick_gap((0.040, 0.030), 0.032, 0.016, 0.0) == pytest.approx(math.hypot(0.008, 0.014))
    assert pl.brick_gap((0.0, 0.040), 0.032, 0.016, 90.0) == pytest.approx(0.008)      # turned 90: roles swap


def test_the_shipped_template_is_refused_until_it_is_set_up():
    template = pl.load_pallet(ROOT / 'config' / 'pallet.yaml')
    assert template['needs_setup'] is True
    with pytest.raises(ValueError, match='still the template'):
        pl.check_pallet(template, CONFIG)


def test_a_good_pallet_passes_and_reports_the_closest_gap():
    notes = pl.check_pallet(pallet(), CONFIG)
    assert any('8.0 mm' in n for n in notes)                                            # 4 cm pitch - 32 mm brick
    assert any('finger_tip' in n for n in notes)                                        # tips unknown: said so


def test_slots_outside_the_window_or_area_are_listed():
    with pytest.raises(ValueError, match='outside the allowed window'):
        pl.check_pallet(pallet(origin=[0.50, -0.10]), CONFIG)                           # x reaches 0.66 > 0.60
    with pytest.raises(ValueError, match='leaves the pallet area'):
        pl.check_pallet(pallet(area=[0.33, 0.50, -0.20, -0.08]), CONFIG)
    pl.check_pallet(pallet(area=[0.33, 0.55, -0.20, -0.08]), CONFIG)


def test_bricks_too_close_or_overlapping_are_refused():
    with pytest.raises(ValueError, match='would be .* apart'):
        pl.check_pallet(pallet(step_along=[0.034, 0.0]), CONFIG)                        # 2 mm gap
    with pytest.raises(ValueError, match='apart'):
        pl.check_pallet(pallet(step_along=[0.025, 0.0]), CONFIG)                        # overlap
    with pytest.raises(ValueError, match='apart'):
        pl.check_pallet(pallet(step_row=[0.0, -0.018]), CONFIG)                         # rows 2 mm apart across the width


def test_the_finger_tips_must_fit_between_neighbouring_bricks():
    tips = dict(CONFIG, finger_tip_thickness=0.010, finger_tip_width=0.020)
    # rows 4 cm apart across the width: 24 mm between the long edges; a tip 10 mm thick + 1 mm room per side fits
    assert not any('finger_tip_' in n for n in pl.check_pallet(pallet(), tips))
    with pytest.raises(ValueError, match='finger tip'):
        pl.check_pallet(pallet(step_row=[0.0, -0.025]), tips)                           # 9 mm between the long edges: a 10 mm tip + 1 mm does not fit
    # bricks in a row (same line, along the long side) never need the room between them for the tips
    pl.check_pallet(pallet(rows=1), tips)


def test_the_finger_rule_itself():
    args = dict(length=0.032, width=0.016, yaw_deg=0.0, open_clearance=0.001, tip_thickness=0.010, tip_width=0.020)
    assert not pl.finger_conflict((0.040, 0.0), **args)                  # same line: the tips stand beside the brick
    assert pl.finger_conflict((0.0, 0.025), **args)                      # neighbour row 2.5 cm away across: tip hits it
    assert not pl.finger_conflict((0.0, 0.030), **args)                  # 3 cm away: the tip (1 + 10 mm) just fits
    assert not pl.finger_conflict((0.0, 0.040), **args)                  # 4 cm away: clear
    assert not pl.finger_conflict((0.060, 0.025), **args)                # far along the long side: clear


# ------------------------------------------------------------------------- the state
def test_state_fills_in_order_and_survives_a_restart(tmp_path):
    p, path = pallet(), tmp_path / 'state.json'
    state = pl.PalletState(path)
    assert state.next_free(p) == 0
    state.mark(0, 'placed', (0.36, -0.10))
    state.mark(1, 'placed', (0.40, -0.10))
    again = pl.PalletState(path)                                         # a new run, a new process
    assert sorted(again.filled) == [0, 1] and again.next_free(p) == 2
    again.unmark(0)
    assert pl.PalletState(path).next_free(p) == 0                        # a freed slot is the next one
    assert json.loads(path.read_text())['filled']['1']['how'] == 'placed'


def test_choosing_a_slot():
    p = pallet(per_row=2, rows=1)
    state = pl.PalletState(Path('/nonexistent/never_written.json'))
    assert pl.choose_slot(p, state) == 0
    assert pl.choose_slot(p, state, 2) == 1                              # slot numbers from 1
    state.filled = {0: {'how': 'placed', 'time': 't'}, 1: {'how': 'skipped', 'time': 't'}}
    with pytest.raises(ValueError, match='full'):
        pl.choose_slot(p, state)
    with pytest.raises(ValueError, match='already filled'):
        pl.choose_slot(p, state, 1)
    with pytest.raises(ValueError, match='does not exist'):
        pl.choose_slot(p, state, 3)


def test_a_damaged_state_file_is_never_silently_ignored(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text('{ not json')
    with pytest.raises(ValueError, match='unreadable'):
        pl.PalletState(path)                  # starting from slot 1 again would put a brick on top of another one


def test_the_state_file_is_next_to_the_pallet_file_whatever_the_folder(tmp_path):
    p = pallet()
    assert pl.state_path(p, tmp_path / 'p.yaml') == (tmp_path / 'pallet_state.json').resolve()
    assert pl.state_path(pallet(state_file='/tmp/x.json'), tmp_path / 'p.yaml') == Path('/tmp/x.json')


def test_the_picture_of_the_pallet(tmp_path):
    p = pallet(per_row=3, rows=2)
    state = pl.PalletState(tmp_path / 's.json')
    state.mark(0, 'placed', (0, 0))
    state.mark(1, 'skipped', (0, 0))
    text = '\n'.join(pl.render(p, state, 2))
    assert '# s >' in text and '. . .' in text and '2 of 6 slots filled' in text


def test_the_command_line_tool(tmp_path):
    pallet_file = tmp_path / 'pallet.yaml'
    pallet_file.write_text(yaml.safe_dump({**GOOD, 'per_row': 3, 'rows': 1}))
    config = ROOT / 'config' / 'pose_pick.yaml'

    def tool(*args):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'scripts' / 'pallet_tool.py'), *args,
                               '--pallet', str(pallet_file), '--config', str(config)], capture_output=True, text=True)
    out = tool('status')
    assert out.returncode == 0 and 'Next: slot 1 of 3' in out.stdout
    assert 'Next: slot 2 of 3' in tool('mark', '1').stdout
    assert 'Skipped slot 2' in tool('skip').stdout and 'Next: slot 3 of 3' in tool('status').stdout
    assert 'Next: slot 2 of 3' in tool('unmark', '2').stdout
    assert 'Pallet geometry OK' in tool('check').stdout
    tool('mark', '2')
    assert 'The pallet is full' in tool('mark', '3').stdout
    assert 'All slots are free' in tool('reset').stdout and 'Next: slot 1 of 3' in tool('status').stdout
    bad = tool('mark')
    assert bad.returncode != 0
    assert tool('mark', '9').returncode != 0 and 'does not exist' in tool('mark', '9').stderr


# ----------------------------------------------------- the tips and the baseplate (pose_pick.py)
def test_a_baseplate_raises_the_surface_and_the_table_once():
    c = dict(CONFIG, baseplate_thickness=0.0032, table_center=list(CONFIG['table_center']))
    pp.apply_baseplate(c)
    pp.apply_baseplate(c)                                                # twice: still only once
    assert c['table_top_z'] == pytest.approx(CONFIG['table_top_z'] + 0.0032)
    assert c['table_center'][2] == pytest.approx(CONFIG['table_center'][2] + 0.0032)
    top = c['table_center'][2] + c['table_size'][2] / 2
    assert top == pytest.approx(c['table_top_z'])                        # the scene's table top is the plate top
    flat = pp.plan_pick(c, 0.40, 0.0)
    assert flat['centre'][2] == pytest.approx(CONFIG['table_top_z'] + 0.0032 + c['block_size'][2] / 2)
    for bad in (0.05, -0.001, 'a', True):
        with pytest.raises(ValueError, match='baseplate_thickness'):
            pp.apply_baseplate(dict(CONFIG, baseplate_thickness=bad, table_center=list(CONFIG['table_center'])))
    untouched = dict(CONFIG, table_center=list(CONFIG['table_center']))
    assert pp.apply_baseplate(untouched)['table_top_z'] == CONFIG['table_top_z']        # 0 = no baseplate


def test_the_shipped_tool_geometry_is_valid():
    pp.check_tool_geometry(CONFIG)
    assert CONFIG['grasp_tcp_height'] >= CONFIG['fingertip_below_tcp'] + CONFIG['fingertip_table_margin']


def test_longer_tips_are_refused_before_they_can_hit_the_table():
    pp.check_tool_geometry(dict(CONFIG, fingertip_below_tcp=0.011))                     # 11 mm + 1 mm margin = 12: just fits (no float trap)
    with pytest.raises(ValueError, match='too low'):
        pp.check_tool_geometry(dict(CONFIG, fingertip_below_tcp=0.014))                 # 3 mm below the brick's bottom
    for bad in (-0.001, 0.08, 'x'):
        with pytest.raises(ValueError, match='fingertip_below_tcp'):
            pp.check_tool_geometry(dict(CONFIG, fingertip_below_tcp=bad))
    pp.check_tool_geometry({k: v for k, v in CONFIG.items() if k != 'fingertip_below_tcp'})   # old configs still load


def test_the_fingertip_floor_follows_the_tip_length():
    long_tips = dict(CONFIG, fingertip_below_tcp=0.011, grasp_tcp_height=0.0125)
    assert pp.plan_pick(long_tips, 0.40, 0.0)['grip'][2] == pytest.approx(CONFIG['table_top_z'] + 0.0125)
    with pytest.raises(ValueError, match='fingertips'):
        pp.plan_pick(dict(long_tips, fingertip_table_margin=0.003), 0.40, 0.0)          # needs 14 mm, has 12.5
    studs = dict(CONFIG, fingertip_table_margin=0.0025)                                 # on a baseplate: studs 1.7 mm
    assert pp.plan_pick(studs, 0.40, 0.0)['grip'][2] == pytest.approx(CONFIG['table_top_z'] + 0.012)


def test_printed_tips_that_do_not_close_fully_change_the_widths():
    printed = dict(CONFIG, hand_zero_width=0.006, open_width=0.024, grasp_width=0.021)  # tips touch at 6 mm
    note = pp.jaw_clearance_note(printed)
    assert note and '1.0 mm of room' in note                                            # 24 - 6 - 16 = 2 mm = 1 per side
    assert pp.jaw_clearance_note(dict(printed, open_width=0.030)) is None
    assert pp.task_config(printed, pp.plan_pick(printed, 0.40, 0.0))['hand_zero_width'] == pytest.approx(0.006)


def test_config_check_knows_about_the_zero_width():
    pytest.importorskip('rclpy')
    module = load('pick_place')
    printed = dict(CONFIG, hand_zero_width=0.006, open_width=0.024, grasp_width=0.021)
    cfg = pp.task_config(printed, pp.plan_pick(printed, 0.40, 0.0))
    assert module.validate(cfg)
    with pytest.raises(ValueError, match='of which 6.0 mm is the tips'):
        module.validate(dict(cfg, open_width=0.0225))                                   # only 0.5 mm room once the tips are counted
    with pytest.raises(ValueError, match='could never close'):
        module.validate(dict(cfg, grasp_width=0.004))                                   # below what the tips allow


# ------------------------------------------------- the slot is recorded when the hand lets go
def test_the_pallet_slot_is_recorded_before_the_back_off():
    order = []
    mission = pp.Mission.__new__(pp.Mission)
    where = {'above': [0.4, -0.1, 0.1], 'release': [0.4, -0.1, 0.0], 'yaw': 0.0}
    mission.place_plan = lambda: where
    mission.show_target = lambda: None
    mission.say = lambda text: order.append(('say', text))
    mission.go = lambda label, xyz, yaw=None: order.append(('go', label))
    mission.mover = types.SimpleNamespace(hand=lambda close=False: order.append(('hand', 'close' if close else 'open')),
                                          detach_block=lambda: order.append(('detach',)))
    mission.on_placed = lambda: order.append(('recorded',))
    mission.place_inspect = False
    mission.place_brick()
    names = [item[1] if item[0] == 'go' else item[0] for item in order]
    assert names.index('detach') < names.index('recorded') < names.index('Back off straight up')


def test_suggested_widths_for_printed_tips():
    hz = load('measure_hand_zero')
    stock = hz.suggest_widths(0.0, 0.016)
    assert stock['open_width'] == pytest.approx(0.018) and stock['grasp_width'] == pytest.approx(0.015)   # today's config
    printed = hz.suggest_widths(0.006, 0.016)
    assert printed == pytest.approx({'hand_zero_width': 0.006, 'open_width': 0.024, 'grasp_width': 0.021})
    cfg = dict(CONFIG, **{k: printed[k] for k in printed})
    assert pp.jaw_clearance_note(cfg) is None or '1.0 mm of room' in pp.jaw_clearance_note(cfg)
    report = '\n'.join(hz.format_report(0.006, 0.016))
    assert 'hand_zero_width: 0.0060' in report and 'open_width: 0.0240' in report and 'grasp_width: 0.0210' in report
    assert 'WARNING' in '\n'.join(hz.format_report(0.05, 0.016))
    assert 'not homed' in '\n'.join(hz.format_report(-0.004, 0.016))
