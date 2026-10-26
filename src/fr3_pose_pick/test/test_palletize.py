"""Incremental palletizing: the row state, the slots, the arguments of each brick, and the aligned place into the next slot."""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('pal_script', ROOT / 'scripts' / 'pose_pick_palletize.py')
pal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pal)
auto, pp = pal.auto, pal.pp
import test_aligned_place as fake  # noqa: E402  (the fake arm)

CONFIG_PATH = ROOT / 'config' / 'pose_pick.yaml'
CONFIG = yaml.safe_load(open(CONFIG_PATH, encoding='utf-8'))
WIDTH = CONFIG['block_size'][1]


def started(tmp_path, placed=0, origin=(0.4395, -0.0192), **kwargs):
    state = pal.new_state(kwargs.get('start_offset', (0.0, 0.08)), kwargs.get('gap', 0.016), 0.0, WIDTH)
    if placed:
        state['origin'] = list(origin)
        state['placed'] = [{'slot': i + 1, 'time': f'2026-10-26 16:{30 + i}:00', 'released_hand_xy': list(origin)} for i in range(placed)]
    path = tmp_path / 'row.json'
    pal.save_state(path, state)
    return path, state


# ------------------------------------------------------------------------------------------------- the row
def test_the_row_is_one_brick_width_plus_the_gap_apart():
    state = pal.new_state((0.0, 0.08), 0.016, 0.0, WIDTH)
    assert state['pitch'] == pytest.approx(0.032)                                         # 16 mm brick + 16 mm gap
    state['origin'] = [0.44, -0.02]
    assert [round(v, 5) for v in pal.slot_xy(state, 0)] == [0.44, -0.02]
    assert [round(v, 5) for v in pal.slot_xy(state, 3)] == [0.44, 0.076]                  # 3 pitches along +y


def test_the_state_is_saved_whole_and_a_damaged_file_is_refused(tmp_path):
    path, state = started(tmp_path, placed=2)
    assert pal.load_state(path) == state and not list(tmp_path.glob('*.tmp'))
    assert pal.load_state(tmp_path / 'none.json') is None
    path.write_text('{"version": 1')
    with pytest.raises(ValueError, match='damaged'):
        pal.load_state(path)
    path.write_text(json.dumps({'version': 2}))
    with pytest.raises(ValueError, match='damaged'):
        pal.load_state(path)


def test_the_row_is_checked_before_anything_moves():
    state = pal.new_state((0.0, 0.08), 0.016, 0.0, WIDTH)
    assert 'NOT checked' in pal.check_row(CONFIG, state)[0]                              # tip size unknown: said so
    for bad_offset in ((0.0, 0.03), (0.02, 0.0)):
        with pytest.raises(ValueError, match='clear of the stack'):
            pal.check_row(CONFIG, pal.new_state(bad_offset, 0.016, 0.0, WIDTH))
    for bad_gap in (0.004, 0.16, 16.0):
        with pytest.raises(ValueError, match='--gap'):
            pal.check_row(CONFIG, pal.new_state((0.0, 0.08), bad_gap, 0.0, WIDTH))
    thick = dict(CONFIG, finger_tip_thickness=0.02, finger_tip_width=0.012)
    with pytest.raises(ValueError, match='finger tip'):
        pal.check_row(thick, pal.new_state((0.0, 0.08), 0.016, 0.0, WIDTH))              # a 20 mm tip does not fit in a 16 mm gap
    thin = dict(CONFIG, finger_tip_thickness=0.008, finger_tip_width=0.012)
    assert pal.check_row(thin, pal.new_state((0.0, 0.08), 0.016, 0.0, WIDTH)) == []


def test_the_status_shows_the_slots_and_the_room_left(tmp_path):
    state = pal.new_state((0.0, 0.08), 0.016, 0.0, WIDTH)
    assert 'nothing placed yet' in pal.status_lines(state)[0] and '+8.0 cm in y' in pal.status_lines(state)[1]
    _, state = started(tmp_path, placed=2)
    lines = pal.status_lines(state, CONFIG)
    assert lines[0].startswith('Row: 2 bricks placed') and 'NEXT' in lines[3] and 'slot  3' in lines[3]
    assert 'x=0.4395 y=0.0448' in lines[3]                                                # -0.0192 + 2 * 0.032
    room = int(lines[-1].split()[2])
    assert room == int(((CONFIG['workspace_y'][1] - 0.008 - 0.002) - 0.0448) // 0.032) + 1


# ------------------------------------------------------------------------------------------- the arguments
class Capture:
    """Stands in for pose_pick.main: records what it was given and plays the part of the mission's release."""
    def __init__(self, fail_on=None, hand=(0.4395, -0.0192), grasp_yaw=2.4, release_yaw=1.7):
        self.calls, self.fail_on, self.hand, self.countdowns = [], fail_on, hand, []
        self.anchors, self.seat_zs, self.grasp_yaw, self.release_yaw, self.wides = [], [], grasp_yaw, release_yaw, []

    def __call__(self):
        self.calls.append(list(sys.argv[1:]))
        if self.fail_on == len(self.calls):
            raise RuntimeError('Lower the brick to the place pose: the hand stopped +2.7 mm above its target')
        cls = pal.PalletizeMission
        mission = cls.__new__(cls)
        self.anchors.append(cls.anchor)
        self.seat_zs.append(cls.seat_z)
        self.wides.append(cls.wide)
        mission.release_yaw_deg = self.release_yaw
        mission.grasp_pick = np.array([0.4263, -0.0056, 0.0002])           # the hand's measured pose at the grasp
        mission.grasp_hand_yaw_deg = self.grasp_yaw
        mission.release_hand = np.array([self.hand[0], self.hand[1] + (len(cls.state['placed']) * 0.032), 0.0003])
        mission.say = lambda text: None
        mission.on_placed = mission.record_slot
        mission.on_placed()
        return 0


@pytest.fixture
def run(monkeypatch, tmp_path):
    def go(argv, capture=None):
        capture = capture or Capture()
        monkeypatch.setattr(pp, 'main', capture)
        monkeypatch.setattr(auto, 'countdown_input', lambda seconds: capture.countdowns.append(seconds) or builtins.input)
        path = str(tmp_path / 'row.json')
        code = pal.main(['--config', str(CONFIG_PATH), '--state', path] + argv)
        return code, capture
    return go


def test_the_first_brick_is_set_down_relative_to_its_own_pick(run, tmp_path):
    code, capture = run(['--camera-xyz', '0.087692', '-0.041569', '0.261043', '--camera-angle', '-86.4'])
    assert code == 0
    argv = capture.calls[0]
    assert argv[:3] == ['--config', str(CONFIG_PATH), '--camera-xyz'] and '--no-inspect' in argv and '--no-step' in argv
    assert argv[argv.index('--place-offset'):argv.index('--place-offset') + 3] == [
        '--place-offset', f'{CONFIG["palletize_start_offset"][0]:.6f}', f'{CONFIG["palletize_start_offset"][1]:.6f}']
    assert CONFIG['palletize_start_offset'] == [0.0, 0.40]
    assert '--place' not in argv and '--place-yaw' not in argv                               # at the angle it was picked at
    assert capture.wides == [True]                                                           # and the studs are searched for
    state = pal.load_state(tmp_path / 'row.json')
    assert state['origin'] == [0.4395, -0.0192] and len(state['placed']) == 1                # measured at the release
    assert state['row_yaw_deg'] == pytest.approx(1.7) and state['seat_z'] == pytest.approx(0.0003)    # learned where it seated


def test_every_next_brick_goes_to_the_next_slot_along_y(run, tmp_path):
    for _ in range(2):                                              # two commands fill slots 1 and 2 ...
        run(['--camera-xyz', '0.0877', '-0.0416', '0.2610', '--camera-angle', '-86.4'])
    state = pal.load_state(tmp_path / 'row.json')
    assert [e['slot'] for e in state['placed']] == [1, 2]
    code, capture = run(['--camera-xyz', '0.0877', '-0.0416', '0.2610', '--camera-angle', '-86.4'])       # ... and this is slot 3
    argv = capture.calls[0]
    place = argv.index('--place')
    assert '--place-offset' not in argv                                                      # now an absolute, measured slot
    x, y = float(argv[place + 1]), float(argv[place + 2])
    assert (x, y) == pytest.approx((0.4395, -0.0192 + 2 * 0.032))                            # slot 3: two pitches from the first


def test_several_bricks_in_one_command_fill_consecutive_slots(run, tmp_path):
    code, capture = run(['--ee-pose', '0.33', '-0.055', '0.22', '1', '0', '0', '0', '--countdown', '9',
                         '--brick', '0.0877', '-0.0416', '0.2610', '-86.4',
                         '--brick', '0.0900', '-0.0300', '0.2600', '-85.0',
                         '--brick', '0.0850', '-0.0500', '0.2620', '-87.0'])
    assert code == 0 and len(capture.calls) == 3
    assert [call[call.index('--camera-xyz') + 1] for call in capture.calls] == ['0.087700', '0.090000', '0.085000']
    assert all('--ee-pose' in call for call in capture.calls)
    assert '--place-offset' in capture.calls[0] and '--place-offset' not in capture.calls[1]
    ys = [float(call[call.index('--place') + 2]) for call in capture.calls[1:]]
    assert ys == pytest.approx([-0.0192 + 0.032, -0.0192 + 0.064], abs=1e-6)
    assert capture.countdowns == [9, 0, 0]                                                   # one countdown per command, then at once
    assert [e['slot'] for e in pal.load_state(tmp_path / 'row.json')['placed']] == [1, 2, 3]


def test_a_failure_stops_the_batch_and_the_row_keeps_what_is_down(run, tmp_path, capsys):
    code, capture = run(['--ee-pose', '0.33', '-0.055', '0.22', '1', '0', '0', '0',
                         '--brick', '0.0877', '-0.0416', '0.2610', '-86.4', '--brick', '0.0900', '-0.0300', '0.2600', '-85.0',
                         '--brick', '0.0850', '-0.0500', '0.2620', '-87.0'], Capture(fail_on=2))
    assert code == 1 and len(capture.calls) == 2                                             # the third brick was never started
    err = capsys.readouterr().err
    assert 'STOPPED at brick 2 of 3' in err and 'already in the row: 1' in err and 'still there' in err
    assert len(pal.load_state(tmp_path / 'row.json')['placed']) == 1


@pytest.mark.parametrize('extra, message', [
    (['--place', '0.4', '0.1'], 'decides where each brick goes'),
    (['--place-offset', '0', '0.05'], 'decides where each brick goes'),
    (['--pallet'], 'decides where each brick goes'),
    (['--place-layers', '2'], 'decides where each brick goes'),
    (['--step'], 'brings a question back'),
    (['--brick', '1', '2', '3', '4', '--camera-xyz', '0', '0', '0.3'], 'use one way'),
    (['--start-offset', '0', '0.02', '--camera-xyz', '0', '0', '0.3'], 'clear of the stack'),
])
def test_bad_commands_are_refused_before_anything_starts(run, extra, message, capsys):
    code, capture = run(extra)
    assert code == 2 and not capture.calls
    assert message in capsys.readouterr().err


def test_a_started_row_keeps_its_offset_and_angle(run, tmp_path, capsys):
    pick = ['--camera-xyz', '0.0877', '-0.0416', '0.2610', '--camera-angle', '-86.4']
    run(pick)
    for option in (['--start-offset', '0', '0.05'], ['--row-yaw', '3']):
        code, capture = run(pick + option)
        assert code == 2 and 'cannot change in the middle' in capsys.readouterr().err and not capture.calls


def test_the_gap_may_change_while_at_most_one_brick_is_down_and_not_after(run, tmp_path, capsys):
    pick = ['--camera-xyz', '0.0877', '-0.0416', '0.2610', '--camera-angle', '-86.4']
    run(pick + ['--gap', '0.02'])                                                            # brick 1 with a 20 mm gap
    code, capture = run(pick + ['--gap', '0.017'])                                           # one brick down: slot 2 can still move
    assert code == 0 and 'Gap changed to 17.0 mm' in capsys.readouterr().out
    state = pal.load_state(tmp_path / 'row.json')
    assert state['pitch'] == pytest.approx(0.033) and len(state['placed']) == 2
    assert pal.slot_xy(state, 1)[1] - pal.slot_xy(state, 0)[1] == pytest.approx(0.033)
    code, capture = run(pick + ['--gap', '0.02'])                                            # two bricks down: refused
    assert code == 2 and 'gap cannot change in the middle' in capsys.readouterr().err and not capture.calls


def test_status_shift_and_reset_move_nothing(run, tmp_path, capsys):
    code, _ = run(['--status'])
    assert code == 0 and 'nothing placed yet' in capsys.readouterr().out
    run(['--camera-xyz', '0.0877', '-0.0416', '0.2610', '--camera-angle', '-86.4'])
    capsys.readouterr()
    code, capture = run(['--shift', '0.001', '-0.0005'])
    out = capsys.readouterr().out
    assert code == 0 and not capture.calls and 'dx=+1.0 mm dy=-0.5 mm' in out
    assert pal.load_state(tmp_path / 'row.json')['origin'] == pytest.approx([0.4405, -0.0197])
    code, _ = run(['--shift', '0.05', '0'])
    assert code == 2 and 'at most 1 cm' in capsys.readouterr().err
    code, capture = run(['--reset'])
    assert code == 0 and not capture.calls and not (tmp_path / 'row.json').exists()


# ------------------------------------------------------------------- the aligned place into a row slot
def row_mission(monkeypatch, origin_known, placed=0):
    """A mission that has picked a brick in layer 2 and is about to set it down in the row (the fake arm of the aligned place)."""
    m = fake.mission()
    m.__class__ = pal.PalletizeMission
    state = pal.new_state((0.0, 0.08), 0.016, 0.0, WIDTH)
    cls = pal.PalletizeMission
    if origin_known:
        state['origin'] = [0.4395, -0.0192]
        state['placed'] = [{'slot': i + 1, 'time': '2026-10-26 16:30:00', 'released_hand_xy': [0, 0]} for i in range(placed)]
    cls.state, cls.index = state, placed
    cls.state_path = None
    cls.truth = pal.slot_xy(state, placed) if origin_known else None
    m.truth_target = cls.truth
    m.on_placed = m.record_slot
    monkeypatch.setattr(pal, 'save_state', lambda path, state: None)        # the file is tested elsewhere
    if origin_known:
        m.place = {'xy': tuple(cls.truth), 'z': None, 'is_tcp': False, 'yaw': 0.0, 'offset': None}
    else:
        m.place = {'offset': (0.0, 0.08), 'xy': None, 'z': None, 'is_tcp': False, 'yaw': 0.0}
    return m, state


def test_the_first_brick_lands_8_cm_from_where_it_really_was_and_defines_the_row(monkeypatch):
    m, state = row_mission(monkeypatch, False)
    m.mover.seat = np.append(m.grasp_pick[:2] + [0.0, 0.08], 0.0)           # the studs: 8 cm from the pick, truth
    m.place_brick()
    assert m.release_hand[:2] == pytest.approx(m.grasp_pick[:2] + [0.0, 0.08], abs=0.0005)
    assert state['origin'] == pytest.approx([round(float(m.release_hand[0]), 5), round(float(m.release_hand[1]), 5)])
    assert len(state['placed']) == 1 and m.mover.opened == 1


def test_a_later_brick_lands_on_the_measured_slot_whatever_its_own_pick_error_was(monkeypatch):
    m, state = row_mission(monkeypatch, True, placed=2)
    slot = np.array(pal.slot_xy(state, 2))
    m.mover.seat = np.append(slot, 0.0)
    m.place_brick()
    assert m.release_hand[:2] == pytest.approx(slot, abs=0.0005)             # slot 3 = 2 pitches from slot 1, to 0.5 mm
    assert len(state['placed']) == 3 and m.mover.opened == 1 and not m.blocked


def test_the_start_offset_comes_from_the_config_and_an_option_overrides_it(run, tmp_path):
    code, capture = run(['--camera-xyz', '0.0877', '-0.0416', '0.2610', '--camera-angle', '-86.4', '--start-offset', '0.0', '0.1'])
    assert code == 0 and capture.calls[0][capture.calls[0].index('--place-offset') + 2] == '0.100000'
    assert pal.load_state(tmp_path / 'row.json')['start_offset'] == [0.0, 0.1]


def test_several_bricks_use_the_arm_pose_the_command_started_with(run, monkeypatch, tmp_path):
    """After brick 1 the arm is somewhere else; the pictures were all taken from where it was at the start."""
    poses = iter([np.full((4, 4), 1.0), np.full((4, 4), 2.0), np.full((4, 4), 3.0)])
    monkeypatch.setattr(pp, 'lookup_hand', lambda node, buffer, c, seconds=10.0: next(poses))
    seen = []
    capture = Capture()
    original_call = capture.__call__.__func__

    def call(self):
        seen.append(pp.lookup_hand(None, None, {}))
        return original_call(self)
    monkeypatch.setattr(Capture, '__call__', call)
    code, _ = run(['--brick', '0.0877', '-0.0416', '0.2610', '-86.4', '--brick', '0.09', '-0.03', '0.26', '-85',
                   '--brick', '0.085', '-0.05', '0.262', '-87'], capture)
    assert code == 0 and len(seen) == 3
    assert all(pose is seen[0] for pose in seen) and seen[0][0, 0] == 1.0                    # read once, shared by all three
    assert pp.lookup_hand.__name__ == '<lambda>'                                             # and put back afterwards


def test_with_an_ee_pose_the_lookup_is_left_alone(run, monkeypatch):
    marker = lambda node, buffer, c, seconds=10.0: 'real'          # noqa: E731
    monkeypatch.setattr(pp, 'lookup_hand', marker)
    capture = Capture()
    code, _ = run(['--ee-pose', '0.33', '-0.055', '0.22', '1', '0', '0', '0', '--brick', '0.0877', '-0.0416', '0.2610', '-86.4',
                   '--brick', '0.09', '-0.03', '0.26', '-85'], capture)
    assert code == 0 and pp.lookup_hand is marker


# --------------------------------------------------------------------------------------------- the anchor
ANCHOR = ['--anchor', '0.1', '-0.03', '0.17', '-88']
PICK = ['--camera-xyz', '0.0877', '-0.0416', '0.2610', '--camera-angle', '-86.4']


def test_the_anchor_defines_the_row_from_a_brick_that_is_seated(run, tmp_path):
    code, capture = run(ANCHOR)
    assert code == 0 and capture.anchors == [True]
    argv = capture.calls[0]
    assert argv[argv.index('--place-offset'):argv.index('--place-offset') + 3] == ['--place-offset', '0', '0']
    assert argv[argv.index('--place-layers') + 1] == 'same' and '--place-yaw' not in argv       # put back exactly where it was
    state = pal.load_state(tmp_path / 'row.json')
    assert state['anchored'] and state['origin'] == [0.4263, -0.0056]                          # the measured grasp, not a guess
    assert state['row_yaw_deg'] == pytest.approx(2.4) and state['seat_z'] == pytest.approx(0.0002)
    assert [e['slot'] for e in state['placed']] == [1]


def test_the_next_bricks_follow_the_anchor_at_its_angle_and_seat_height(run, tmp_path):
    run(ANCHOR)
    code, capture = run(PICK)
    argv = capture.calls[0]
    place = argv.index('--place')
    assert (float(argv[place + 1]), float(argv[place + 2])) == pytest.approx((0.4263, -0.0056 + 0.032))   # slot 2, one pitch on
    assert argv[argv.index('--place-yaw') + 1] == '2.400000'                                       # the anchor's angle, not 0
    assert capture.anchors == [False] and capture.seat_zs == [pytest.approx(0.0002)]              # and its measured seat height


@pytest.mark.parametrize('yaw, expected', [(2.4, 2.4), (-3.0, -3.0), (178.0, -2.0), (-178.0, 2.0), (90.0, -90.0)])
def test_the_anchor_angle_is_a_line_between_minus_90_and_90(run, tmp_path, yaw, expected):
    run(ANCHOR, Capture(grasp_yaw=yaw))
    assert pal.load_state(tmp_path / 'row.json')['row_yaw_deg'] == pytest.approx(expected)


@pytest.mark.parametrize('extra, message', [
    (['--brick', '1', '2', '3', '4'], 'anchor is the brick'),
    (['--camera-xyz', '0', '0', '0.3'], 'anchor is the brick'),
    (['--start-offset', '0', '0.1'], 'leave out --start-offset'),
    (['--row-yaw', '3'], 'leave out --start-offset'),
])
def test_the_anchor_is_refused_with_other_ways_to_say_where(run, extra, message, capsys):
    code, capture = run(ANCHOR + extra)
    assert code == 2 and not capture.calls and message in capsys.readouterr().err


def test_an_anchor_needs_an_empty_row_and_a_reset_makes_one(run, tmp_path, capsys):
    run(PICK)                                                                                      # a row with one brick
    code, capture = run(ANCHOR)
    assert code == 2 and 'Use --reset first' in capsys.readouterr().err and not capture.calls
    code, capture = run(['--reset'] + ANCHOR)                                                      # forget it and start from the anchor
    assert code == 0 and capture.anchors == [True]
    state = pal.load_state(tmp_path / 'row.json')
    assert state['anchored'] and len(state['placed']) == 1


def test_the_status_says_when_a_row_is_measured_from_an_anchor(run, tmp_path, capsys):
    run(ANCHOR)
    capsys.readouterr()
    run(['--status'])
    out = capsys.readouterr().out
    assert 'measured from the anchor brick, seat height z=0.0002' in out and 'long side at 2.4 deg' in out


def test_an_old_row_file_without_the_new_fields_still_loads(tmp_path):
    path, state = started(tmp_path, placed=1)
    saved = json.loads(path.read_text())
    saved.pop('seat_z'), saved.pop('anchored')
    path.write_text(json.dumps(saved))
    loaded = pal.load_state(path)
    assert loaded['seat_z'] is None and loaded['anchored'] is False


# ------------------------------------------------------------------- the first brick finds its own seat
def test_only_the_first_brick_of_an_unanchored_row_searches_widely(run, tmp_path):
    code, capture = run(PICK)
    assert capture.wides == [True] and capture.anchors == [False]
    code, capture = run(PICK)                                                                # slot 2: the row is known now
    assert capture.wides == [False]
    code, capture = run(['--reset'] + ANCHOR)
    assert capture.wides == [False] and capture.anchors == [True]                            # an anchor is already seated


def test_the_next_bricks_use_the_angle_and_seat_height_of_the_first_that_seated(run, tmp_path):
    run(PICK, Capture(release_yaw=1.7))
    code, capture = run(PICK)
    argv = capture.calls[0]
    assert argv[argv.index('--place-yaw') + 1] == '1.700000' and capture.seat_zs == [pytest.approx(0.0003)]


def test_a_row_yaw_given_by_hand_is_used_for_the_first_brick_and_not_replaced(run, tmp_path):
    code, capture = run(PICK + ['--row-yaw', '0'], Capture(release_yaw=1.7))
    assert capture.calls[0][capture.calls[0].index('--place-yaw') + 1] == '0.000000'
    assert pal.load_state(tmp_path / 'row.json')['row_yaw_deg'] == 0.0                       # the hand's 1.7 deg does not overwrite it


def test_the_far_start_and_the_window_for_it():
    assert CONFIG['palletize_start_offset'][1] == pytest.approx(0.40)
    far = CONFIG['workspace_y'][1] - 0.4 + 0.02                                            # a stack at y = -0.10 ... -0.12
    for pick_y in (-0.12, -0.10, -0.08):
        pp.plan_place(CONFIG, 0.44, pick_y + CONFIG['palletize_start_offset'][1], 0.0)    # the first slot is inside the window
    with pytest.raises(ValueError, match='outside the allowed window'):
        pp.plan_place(CONFIG, 0.44, CONFIG['workspace_y'][1] + 0.01, 0.0)
    assert far > 0
