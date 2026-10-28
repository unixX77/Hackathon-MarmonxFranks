"""Same layer for the place, the planning-scene table that lets a brick be pressed onto the studs, the spline for short
moves, and the gripper's finger width as feedback on the grasp."""
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


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pp = load('pp_feedback', 'pose_pick.py')
auto = load('auto_feedback', 'pose_pick_auto.py')
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))


def with_baseplate(**changes):
    c = dict(CONFIG, **changes)
    c['table_center'] = list(CONFIG['table_center'])
    pp.apply_baseplate(c)
    return c


# ------------------------------------------------------------------------------------------ --place-layers same
def test_same_is_a_place_layer_and_only_a_place_layer():
    assert pp.layers_arg('same') == 'same' and pp.layers_arg(' SAME ') == 'same' and pp.layers_arg('auto') == 'auto'
    assert pp.layers_arg('3') == 3


@pytest.mark.parametrize('argv, message', [
    (['--layers', 'same', '--camera-xyz', '0', '0', '0.3'], 'only for --place-layers'),
    (['--pose', '0.4', '0.1', '--place-camera-xyz', '0', '0', '0.3', '--place-layers', 'same'], 'needs --place X Y or --place-offset'),
    (['--pose', '0.4', '0.1', '--place', '0.4', '0.2', '0.0', '--place-layers', 'same'], 'sets the height'),
    (['--pose', '0.4', '0.1', '--pallet', '--place-layers', 'same'], 'needs --place'),
])
def test_bad_uses_of_same_are_refused_before_anything_starts(monkeypatch, argv, message):
    monkeypatch.setattr(sys, 'argv', ['pose_pick.py', '--config', str(ROOT / 'config' / 'pose_pick.yaml')] + argv)
    with pytest.raises(ValueError, match=message):
        pp.main()


# --------------------------------------------------------------------- the planning scene and the 1 mm press
def attached_bottom(c, release_hand_z):
    """Lowest point of the attached collision box (attach_block makes it 2 mm shorter than the brick)."""
    offset = c['grasp_tcp_height'] - c['block_size'][2] / 2
    return release_hand_z - offset - (c['block_size'][2] - 0.002) / 2


def scene_top(c):
    return c['table_center'][2] + c['table_size'][2] / 2


def test_without_the_sink_the_pressed_brick_is_a_collision_and_with_it_it_is_not():
    """The real failure: 'straight line not possible (reached 95 %)' - the last waypoint of the lowering was 1 mm into the table."""
    c = with_baseplate(scene_table_sink=0.0)
    release = pp.plan_place(c, 0.44, -0.10, 0.0)['release'][2]
    assert release - pp.plan_place(c, 0.44, -0.10, 0.0)['grip'][2] == pytest.approx(-0.001)
    assert attached_bottom(c, release) <= scene_top(pp.scene_config(c)) + 1e-9        # touching or into the table: refused
    c = with_baseplate()
    assert CONFIG['scene_table_sink'] > 0
    assert attached_bottom(c, release) > scene_top(pp.scene_config(c))                # clear of the lowered scene table
    # and a correction that presses the full place_max_press more than the margin still clears it
    assert attached_bottom(c, release - (c['place_max_press'] + c['place_margin'])) > scene_top(pp.scene_config(c))


def test_only_the_planning_scene_table_moves():
    c = with_baseplate()
    scene = pp.scene_config(c)
    assert scene['table_center'][2] == pytest.approx(c['table_center'][2] - c['scene_table_sink'])
    assert scene['table_center'][:2] == c['table_center'][:2] and c['table_center'][2] == pytest.approx(CONFIG['table_center'][2] + 0.0032)
    assert c['table_top_z'] == pytest.approx(CONFIG['table_top_z'] + 0.0032)          # the checked heights are untouched
    assert pp.scene_config(dict(c, scene_table_sink=0.0)) is not None and pp.scene_config(dict(c, scene_table_sink=0.0))['table_center'] == c['table_center']


def test_the_sink_must_cover_the_allowed_press():
    pp.check_scene_sink(CONFIG)
    pp.check_scene_sink(dict(CONFIG, scene_table_sink=0.0, place_max_press=0.0))
    with pytest.raises(ValueError, match='less than'):
        pp.check_scene_sink(dict(CONFIG, scene_table_sink=0.002))                       # 3 mm may be pressed, 2 mm sunk
    for bad in (-0.001, 0.02, 'x', True):
        with pytest.raises(ValueError, match='scene_table_sink'):
            pp.check_scene_sink(dict(CONFIG, scene_table_sink=bad))


# ---------------------------------------------------------------------------------------- spline for short moves
def test_the_spline_degree_is_one_scipy_accepts_for_every_number_of_points():
    from scipy.interpolate import make_interp_spline
    ts = pytest.importorskip('trajectory_smoothing')
    assert [ts.spline_degree(n) for n in range(2, 9)] == [1, 2, 3, 3, 5, 5, 5]
    for n in range(2, 12):
        x = np.linspace(0.0, 1.0, n)
        make_interp_spline(x, np.random.default_rng(n).normal(size=(n, 7)), k=ts.spline_degree(n))     # must not raise


# ------------------------------------------------------------------------------------ the gripper's finger width
def mission(widths=None, tolerance=None):
    m = pp.Mission.__new__(pp.Mission)
    m.c = dict(CONFIG, **({'grasp_check_tolerance': tolerance} if tolerance else {}))
    m.grasp_widths, m.said = {}, []
    m.say = m.said.append
    m.spin = lambda seconds: None
    readings = list(widths or [])
    m.mover = types.SimpleNamespace(finger_width=lambda: readings.pop(0) if readings else None)
    return m


def test_a_held_brick_reads_the_brick_width():
    m = mission([0.0158, 0.0159])
    m.check_grasp('after closing')
    m.check_grasp('after the lift')
    assert m.grasp_widths == {'after closing': 0.0158, 'after the lift': 0.0159}
    assert all('WARNING' not in line for line in m.said) and 'fingers at 15.8 mm' in m.said[0]


@pytest.mark.parametrize('width, text', [(None, 'no finger width'), (0.0005, 'not (or not only) between'),
                                         (0.0700, 'not (or not only) between'), (0.0121, 'not (or not only) between')])
def test_a_missed_grasp_or_a_silent_gripper_is_reported(width, text):
    m = mission([width])
    assert text in m.grasp_problem('after closing', width)
    m.check_grasp('after closing')
    assert any('WARNING' in line and text in line for line in m.said)                        # here: a warning, nothing stops


def test_a_brick_that_slips_or_falls_during_the_lift_is_reported():
    m = mission()
    m.grasp_widths['after closing'] = 0.0158
    assert m.grasp_problem('after the lift', 0.0160) is None                                 # 0.2 mm: noise
    wider = 0.0158 + m.c.get('hold_check_tolerance', 0.001) + 0.0005                         # just over the tolerance
    assert 'slipping or fell' in m.grasp_problem('after the lift', wider)
    assert 'between' in m.grasp_problem('after the lift', 0.0001)                            # the fingers closed on nothing


def test_the_autonomous_script_stops_on_it_before_carrying_on():
    m = auto.AutoMission.__new__(auto.AutoMission)
    m.c, m.grasp_widths, m.say, m.spin = dict(CONFIG), {}, lambda text: None, lambda seconds: None
    m.mover = types.SimpleNamespace(finger_width=lambda: 0.0004)
    with pytest.raises(auto.GraspNotConfirmed, match='not carried on'):
        m.check_grasp('after closing')
    m.mover = types.SimpleNamespace(finger_width=lambda: None)
    with pytest.raises(auto.GraspNotConfirmed, match='no finger width'):
        m.check_grasp('after closing')
    m.mover = types.SimpleNamespace(finger_width=lambda: 0.0158)
    m.check_grasp('after closing')                                                            # a held brick: no error


def test_the_widths_go_to_the_autonomous_log(tmp_path):
    m = auto.AutoMission.__new__(auto.AutoMission)
    m.c = dict(CONFIG, auto_log=str(tmp_path / 'auto.json'))
    m.plan, m.place, m.blocked, m.given, m.failure = pp.plan_pick(CONFIG, 0.40, 0.04), None, [], {}, None
    m.grasp_widths = {'after closing': 0.01583, 'after the lift': 0.01586}
    m.say = lambda text: None
    m.save_log()
    assert json.loads((tmp_path / 'auto.json').read_text())[-1]['finger_width_mm'] == {'after closing': 15.83, 'after the lift': 15.86}


def test_the_mover_adds_the_two_finger_positions():
    pytest.importorskip('rclpy')
    sm = load('sm_feedback', 'smooth_motion.py')
    mover = sm.Mover.__new__(sm.Mover)
    mover.finger_positions, mover.finger_stamp = {}, None
    assert mover.finger_width() is None                                                       # nothing received yet
    message = types.SimpleNamespace(name=['fr3_finger_joint1', 'fr3_finger_joint2'], position=[0.0079, 0.0079], effort=[0.0, 0.0])
    mover._on_fingers(message)
    assert mover.finger_width() == pytest.approx(0.0158)
    mover.finger_stamp -= 5.0
    assert mover.finger_width() is None                                                       # an old reading is not trusted
    mover._on_fingers(types.SimpleNamespace(name=['other'], position=[1.0]))
    assert 'other' not in mover.finger_positions


def test_the_shipped_config_is_consistent_with_these_checks():
    pp.check_scene_sink(CONFIG)
    assert CONFIG['scene_table_sink'] >= CONFIG['place_max_press']
    assert 0 < CONFIG['hold_check_tolerance'] < CONFIG['grasp_check_tolerance']
