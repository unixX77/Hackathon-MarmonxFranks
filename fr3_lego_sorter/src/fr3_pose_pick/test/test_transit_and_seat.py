"""Far-away moves at transit speed, no countdown for the next bricks, the row's measured seat height, the seated-or-not limit."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_aligned_place as fake  # noqa: E402  (the fake arm and the mission around it)

auto, pp, CONFIG = fake.auto, fake.pp, fake.CONFIG


def mission(**kwargs):
    m = fake.mission(**kwargs)
    m.mover.scale = 0.05
    m.scales = []
    plain_move = m.mover.move

    def move(label, xyz, yaw=None):
        m.scales.append((label, m.mover.scale))                    # the speed the move is planned with
        return plain_move(label, xyz, yaw)
    m.mover.move = move
    return m


# ------------------------------------------------------------------------------------------------- speed
def test_far_moves_run_at_transit_speed_and_the_approach_and_press_do_not():
    m = mission(config={'transit_velocity_scale': 0.10, 'fast_velocity_scale': 0.50})
    m.place_brick()
    speeds = {label: scale for label, scale in m.scales}
    for label in auto.AutoMission.TRANSIT:
        if label in speeds:                                       # the sideways travel high up: fast; up: transit
            assert speeds[label] == pytest.approx(0.50 if label in auto.AutoMission.FAST else 0.10), label
    for label in ('Lower the brick to the alignment height', 'Align the brick over its place', 'Lower the brick to the fine height',
                  'Align the brick over its place (fine)', 'Lower the brick to the place pose'):
        assert speeds[label] == pytest.approx(0.05), label                       # the speed the pick and the press have always had
    assert m.mover.scale == pytest.approx(0.05)                                  # and it is put back after every move


def test_the_speed_is_put_back_even_if_a_move_fails():
    m = mission(config={'transit_velocity_scale': 0.10})

    def failing(label, xyz, yaw=None):
        raise RuntimeError('straight line not possible')
    m.mover.move = failing
    with pytest.raises(RuntimeError):
        m.go('Lift the brick', m.plan['above'])
    assert m.mover.scale == pytest.approx(0.05)


def test_without_the_setting_nothing_gets_faster():
    m = mission(config={'transit_velocity_scale': None})
    m.go('Lift the brick', m.plan['above'])
    assert m.scales[-1] == ('Lift the brick', 0.05)


def test_the_transit_speed_is_checked_before_anything_moves():
    auto.check_auto_settings(CONFIG)
    assert CONFIG['transit_velocity_scale'] == pytest.approx(0.10)
    for bad in (0.0, -0.1, 0.3, 'fast', True):
        with pytest.raises(ValueError, match='transit_velocity_scale'):
            auto.check_auto_settings(dict(CONFIG, transit_velocity_scale=bad))
    for bad in (0.002, 0.04, 0.08):
        with pytest.raises(ValueError, match='place_fine_height'):
            auto.check_auto_settings(dict(CONFIG, place_fine_height=bad))


# ---------------------------------------------------------------------------------------------- countdown
def test_a_countdown_of_zero_goes_at_once(monkeypatch, capsys):
    monkeypatch.setattr(auto.time, 'sleep', lambda s: pytest.fail('must not wait'))
    assert auto.countdown_input(0)('Hold the e-stop, clear the workspace. Type GO to start: ') == 'GO'
    assert 'no countdown' in capsys.readouterr().out
    with pytest.raises(RuntimeError, match='never asks questions'):
        auto.countdown_input(0)('  correction> ')                              # still never answers anything else


# ----------------------------------------------------------------------------------- the row's seat height
def test_the_press_goes_to_the_measured_seat_not_the_models():
    m = fake.mission()
    model_release = m.place_plan()['release'][2]
    m.seat_z = model_release - CONFIG['place_margin'] + 0.0012               # the row's plate is 1.2 mm higher than the model says
    m.place_brick()
    press = [entry for entry in m.mover.log if entry[0] == 'Lower the brick to the place pose'][0]
    assert press[1][2] == pytest.approx(m.seat_z + CONFIG['place_margin'])
    fine = [entry for entry in m.mover.log if entry[0] == 'Lower the brick to the fine height'][0]
    assert fine[1][2] == pytest.approx(m.seat_z + CONFIG['place_margin'] + CONFIG['place_fine_height'])


def test_a_seat_height_far_from_the_model_stops_before_moving():
    m = fake.mission()
    m.seat_z = m.place_plan()['release'][2] - CONFIG['place_margin'] + 0.006
    with pytest.raises(RuntimeError, match='Nothing was moved'):
        m.place_brick()
    assert m.mover.opened == 0 and not [e for e in m.mover.log if e[0].startswith('Carry')]


# ------------------------------------------------------------------------------------- seated or on the studs
def test_the_seated_limit_sits_between_a_seated_brick_and_one_on_the_studs():
    m = fake.mission()
    assert m.seated_limit() == pytest.approx(0.0018)                           # place_margin -1 mm: seated = +1.0, on the studs = +2.7
    m.c['place_margin'] = 0.0
    assert m.seated_limit() == pytest.approx(0.0012)
    m.c['place_margin'] = 0.002
    assert m.seated_limit() == pytest.approx(0.0012)


def test_the_blocked_limit_is_only_changed_for_the_one_move():
    m = fake.mission()
    m.c['blocked_height'] = 0.002
    m.go('Lower the brick to the place pose', m.place_plan()['release'], 0.0, watch_blocked=False, blocked_height=0.0018)
    assert m.c['blocked_height'] == 0.002
    del m.c['blocked_height']
    m.go('Lower the brick to the place pose', m.place_plan()['release'], 0.0, watch_blocked=False, blocked_height=0.0018)
    assert 'blocked_height' not in m.c
