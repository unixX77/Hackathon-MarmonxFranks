"""The place inspection stop: correct x, y, z, yaw of the place by eye before the brick is lowered."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('pp_place', ROOT / 'scripts' / 'pose_pick.py')
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))


def mission(answers=(), inspect=True, place=None):
    """A Mission without ROS: records every move and hand action, answers the prompt from `answers`."""
    log = []
    m = pp.Mission.__new__(pp.Mission)
    m.c = dict(CONFIG, place_margin=0.002)         # the tests below are written for a release 2 mm above the seat
    m.plan = pp.plan_pick(CONFIG, 0.40, 0.036)
    m.place = place or {'offset': (0.0, 0.07), 'xy': None, 'z': None, 'is_tcp': False, 'yaw': None}
    m.place_nudge = np.zeros(4)
    m.blocked = []
    m.place_inspect = inspect
    m.on_placed = lambda: log.append(('placed',))
    m.say = lambda text: None
    m.show_target = lambda: None
    m.go = lambda label, xyz, yaw=None, watch_blocked=False: log.append(('go', label, [round(v, 5) for v in xyz]))
    m.mover = types.SimpleNamespace(hand=lambda close=False: log.append(('hand', 'close' if close else 'open')),
                                    detach_block=lambda: log.append(('detach',)))
    m.answers = list(answers)
    return m, log


@pytest.fixture
def typed(monkeypatch):
    def feed(m):
        monkeypatch.setattr('builtins.input', lambda prompt='': m.answers.pop(0))
    return feed


# ----------------------------------------------------------------------------- the plan
def test_the_place_plan_has_an_inspection_point_above_the_release_height():
    m, _ = mission()
    where = m.place_plan()
    assert where['place_inspect'][2] - where['release'][2] == pytest.approx(CONFIG['place_inspect_height'])
    assert where['place_inspect'][:2] == where['release'][:2]


def test_a_correction_moves_every_point_of_the_place_together():
    m, _ = mission()
    base = m.place_plan()
    m.place_nudge = np.array([0.002, -0.001, 3.0, -0.002])               # dx, dy, dyaw (deg), dz
    moved = m.place_plan()
    for key in pp.PLACE_KEYS:
        assert moved[key][0] - base[key][0] == pytest.approx(0.002)
        assert moved[key][1] - base[key][1] == pytest.approx(-0.001)
        assert moved[key][2] - base[key][2] == pytest.approx(-0.002)
    assert np.degrees(moved['yaw'] - base['yaw']) == pytest.approx(3.0)
    # and a place that follows the pick still follows it after a correction at the pick
    pp.apply_nudge(m.plan, 0.003, 0.0, 0.0, 0.0)
    assert m.place_plan()['centre'][0] - moved['centre'][0] == pytest.approx(0.003)


# ----------------------------------------------------------------------------- what is allowed
def test_the_shipped_place_may_be_pressed_only_a_little_more():
    """place_margin -1 mm (pressed onto the studs) + place_max_press 3 mm: 2 mm more is the limit, never beyond."""
    assert CONFIG['place_margin'] + CONFIG['place_max_press'] > 0
    pp.check_place_nudge(CONFIG, [0, 0, 0, -(CONFIG['place_margin'] + CONFIG['place_max_press'])], (0.40, 0.10))
    with pytest.raises(ValueError, match='into the surface'):
        pp.check_place_nudge(CONFIG, [0, 0, 0, -(CONFIG['place_margin'] + CONFIG['place_max_press']) - 0.0002], (0.40, 0.10))
    assert pp.plan_place(CONFIG, 0.40, 0.10, 0.0)['release'][2] - pp.plan_place(CONFIG, 0.40, 0.10, 0.0)['grip'][2] \
        == pytest.approx(-0.001)


def test_what_corrections_at_the_place_are_allowed():
    c = dict(CONFIG, place_margin=0.002)
    pp.check_place_nudge(c, [0.002, -0.001, 3.0, -0.002], (0.40, 0.10))                  # small: fine
    pp.check_place_nudge(c, [0.0, 0.0, 0.0, -0.005], (0.40, 0.10))                       # margin 2 mm + 3 mm of press: the limit
    with pytest.raises(ValueError, match='into the surface'):
        pp.check_place_nudge(c, [0.0, 0.0, 0.0, -0.0061], (0.40, 0.10))
    pp.check_place_nudge(c, [0.0, 0.0, 0.0, 0.012], (0.40, 0.10))                        # up is only limited by the 15 mm total
    with pytest.raises(ValueError, match='more than allowed'):
        pp.check_place_nudge(c, [0.0, 0.0, 0.0, 0.016], (0.40, 0.10))
    with pytest.raises(ValueError, match='more than allowed'):
        pp.check_place_nudge(c, [0.025, 0.025, 0.0, 0.0], (0.40, 0.10))                  # 35 mm sideways
    with pytest.raises(ValueError, match='more than allowed'):
        pp.check_place_nudge(c, [0.0, 0.0, 50.0, 0.0], (0.40, 0.10))
    with pytest.raises(ValueError, match='allowed window'):
        pp.check_place_nudge(c, [0.0, 0.0, 0.0, 0.0], (0.61, 0.0))
    deeper = dict(c, place_margin=0.0, place_max_press=0.0005)                          # pressing onto studs, only 0.5 mm
    pp.check_place_nudge(deeper, [0, 0, 0, -0.0005], (0.40, 0.10))
    with pytest.raises(ValueError, match='into the surface'):
        pp.check_place_nudge(deeper, [0, 0, 0, -0.001], (0.40, 0.10))


# -------------------------------------------------------------------------------- the flow
def test_without_the_stop_nothing_changes():
    m, log = mission(inspect=False)
    m.place_brick()
    assert [e[1] for e in log if e[0] == 'go'] == ['Carry the brick above the place pose',
                                                   'Lower the brick to the place pose', 'Back off straight up']


def test_the_stop_asks_before_lowering_and_applies_the_correction(typed):
    m, log = mission(['z -0.002', 'x 0.001 yaw 2', 'ok'])
    typed(m)
    planned = m.place_plan()
    m.place_brick()
    names = [e[1] if e[0] == 'go' else e[0] for e in log]
    assert names == ['Carry the brick above the place pose', 'Lower the brick to the place inspection height',
                     'Place correction', 'Place correction', 'Lower the brick to the place pose', 'hand', 'detach',
                     'placed', 'Back off straight up']
    lower = [e for e in log if e[0] == 'go' and e[1] == 'Lower the brick to the place pose'][0][2]
    assert lower[2] == pytest.approx(planned['release'][2] - 0.002, abs=1e-5)       # 2 mm lower
    assert lower[0] == pytest.approx(planned['release'][0] + 0.001, abs=1e-5)       # 1 mm further out
    back_off = [e for e in log if e[0] == 'go' and e[1] == 'Back off straight up'][0][2]
    assert back_off[0] == pytest.approx(planned['above'][0] + 0.001, abs=1e-5)      # backs off straight up from where it was
    assert back_off[2] == pytest.approx(planned['above'][2] - 0.002, abs=1e-5)
    assert m.place_nudge == pytest.approx([0.001, 0.0, 2.0, -0.002])


def test_too_deep_a_correction_is_refused_and_the_prompt_goes_on(typed, capsys):
    m, log = mission(['z -0.005', 'z -0.001', 'ok'])                     # -5 mm is the limit, -6 mm is not
    typed(m)
    m.place_brick()
    assert m.place_nudge[3] == pytest.approx(-0.005)
    out = capsys.readouterr().out
    assert 'into the surface' in out and 'Total place correction so far' in out


def test_a_wrong_answer_does_not_move_anything(typed, capsys):
    m, log = mission(['hello', 'x 0.5', 'z 0.02', 'ok'])
    typed(m)
    m.place_brick()
    assert not np.any(m.place_nudge)
    assert [e[1] for e in log if e[0] == 'go'].count('Place correction') == 0
    assert 'Type e.g.' in capsys.readouterr().out


def test_quitting_at_the_stop_leaves_the_brick_in_the_hand(typed):
    from smooth_motion import StepAbort
    m, log = mission(['q'])
    typed(m)
    with pytest.raises(StepAbort, match='still in the hand'):
        m.place_brick()
    assert ('hand', 'open') not in log and ('detach',) not in log and ('placed',) not in log     # nothing was released


def test_the_correction_is_written_to_the_log(tmp_path, typed):
    m, _ = mission(['z -0.002', 'ok'])
    typed(m)
    m.place_brick()
    m.c['offsets_log'] = str(tmp_path / 'log.json')
    m.nudge = np.zeros(4)
    m.grasp_hand_yaw_deg = 0.0
    m.given = {}
    m.save_log()
    import json
    entry = json.loads((tmp_path / 'log.json').read_text())[-1]
    assert entry['place_correction'] == {'dx_dy_m': [0.0, 0.0], 'dz_m': pytest.approx(-0.002), 'yaw_deg': 0.0, 'inspected': True}
