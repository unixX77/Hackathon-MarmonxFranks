"""The two-stage closing (gentle first, full grip second) and the record of what closing did to the hand."""
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
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))
import test_aligned_place as fake  # noqa: E402

auto = fake.auto


# ------------------------------------------------------------------------------------------------ the two stages
def mover(config, fail_at=None):
    pytest.importorskip('franka_msgs')
    spec = importlib.util.spec_from_file_location('sm_gentle', ROOT / 'scripts' / 'smooth_motion.py')
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    sent = []

    class Client:
        def __init__(self, kind):
            self.kind = kind

        def send_goal_async(self, goal):
            sent.append((self.kind, round(goal.width, 4), round(goal.speed, 4), round(getattr(goal, 'force', 0.0), 1)))
            return types.SimpleNamespace(accepted=True, get_result_async=lambda: len(sent))

    m = sm.Mover.__new__(sm.Mover)
    m.c = dict(CONFIG, **config)
    m.ask_step = lambda label, detail: sent.append(('step', label, detail))
    m.move_client, m.grasp_client = Client('open'), Client('grasp')
    m.wait = lambda future, timeout: (types.SimpleNamespace(status=4, result=types.SimpleNamespace(success=fail_at != future))
                                      if isinstance(future, int) else future)
    return m, sent, sm


def test_closing_is_gentle_first_and_firm_second(monkeypatch):
    m, sent, sm = mover({})
    pauses = []
    monkeypatch.setattr(sm.time, 'sleep', pauses.append)
    m.hand(close=True)
    goals = [g for g in sent if g[0] == 'grasp']
    assert goals == [('grasp', 0.015, 0.01, 8.0), ('grasp', 0.015, 0.01, 40.0)]       # slow and 8 N, then 40 N
    assert pauses == [0.3]                                                          # the brick settles in between
    detail = [g for g in sent if g[0] == 'step'][0][2]
    assert '40 N' in detail and 'first 8 N' in detail


def test_opening_is_one_move_at_the_normal_speed_and_unchanged(monkeypatch):
    m, sent, sm = mover({})
    monkeypatch.setattr(sm.time, 'sleep', lambda s: pytest.fail('opening must not wait'))
    m.hand()
    assert [g for g in sent if g[0] != 'step'] == [('open', 0.019, 0.02, 0.0)]


def test_without_a_first_force_it_closes_in_one_go_as_before(monkeypatch):
    m, sent, sm = mover({'grasp_force_first': None})
    monkeypatch.setattr(sm.time, 'sleep', lambda s: pytest.fail('one stage must not wait'))
    m.hand(close=True)
    assert [g for g in sent if g[0] == 'grasp'] == [('grasp', 0.015, 0.01, 40.0)]


def test_a_first_force_that_is_not_gentler_is_one_stage(monkeypatch):
    m, sent, sm = mover({'grasp_force_first': 40.0})
    monkeypatch.setattr(sm.time, 'sleep', lambda s: pytest.fail('one stage must not wait'))
    m.hand(close=True)
    assert len([g for g in sent if g[0] == 'grasp']) == 1


def test_a_miss_in_the_gentle_stage_stops_before_the_firm_one(monkeypatch):
    m, sent, sm = mover({}, fail_at=2)                                              # the first grasp goal (item 2: the step note is item 1) fails
    monkeypatch.setattr(sm.time, 'sleep', lambda s: None)
    with pytest.raises(RuntimeError, match='Grasp failed'):
        m.hand(close=True)
    assert len([g for g in sent if g[0] == 'grasp']) == 1


def test_the_gentle_settings_are_checked_before_anything_moves():
    pytest.importorskip('rclpy')
    spec = importlib.util.spec_from_file_location('pick_place_gentle', ROOT / 'scripts' / 'pick_place.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import pose_pick as pp
    cfg = pp.task_config(CONFIG, pp.plan_pick(CONFIG, 0.40, 0.0))
    assert module.validate(cfg)
    assert module.validate({k: v for k, v in cfg.items() if k not in ('grasp_force_first', 'grasp_speed', 'grasp_settle')})   # optional
    for key, bad in (('grasp_force_first', 50.0), ('grasp_force_first', -1.0), ('grasp_speed', 0.5), ('grasp_settle', 10.0),
                     ('grasp_force_first', True)):
        with pytest.raises(ValueError):
            module.validate(dict(cfg, **{key: bad}))
    with pytest.raises(ValueError, match='must not be more than grasp_force'):
        module.validate(dict(cfg, grasp_force_first=15.0, grasp_force=10.0))


def test_the_shipped_settings_are_gentle_then_firm():
    assert 0 < CONFIG['grasp_force_first'] < CONFIG['grasp_force'] == 40.0
    assert CONFIG['grasp_speed'] <= CONFIG['gripper_speed']
    assert 0 < CONFIG['grasp_settle'] <= 1.0


# ------------------------------------------------------------------------------------- what closing did to the hand
def closing_mission(tmp_path, after, fingers=(0.0078, 0.0076), yaw_deg=0.0):
    m = fake.mission()
    m.grasp_pick = np.array([0.40, 0.00, 0.0])                                     # where the hand was when it was about to close
    pose = np.eye(4)
    rotation = np.array([[np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg)), 0], [np.sin(np.radians(yaw_deg)), -np.cos(np.radians(yaw_deg)), 0], [0, 0, -1]])
    pose[:3, :3], pose[:3, 3] = rotation, after
    m.mover = types.SimpleNamespace(hand_pose=lambda: pose, finger_width=lambda: 0.0155,
                                    finger_positions={'fr3_finger_joint1': fingers[0], 'fr3_finger_joint2': fingers[1]})
    m.c = dict(m.c, auto_log=str(tmp_path / 'auto.json'))
    return m


def test_the_shift_of_the_hand_while_closing_is_recorded_across_and_along(tmp_path):
    m = closing_mission(tmp_path, after=[0.4004, -0.0008, 0.0])
    m.check_grasp('after closing')
    assert m.closing['along_mm'] == pytest.approx(0.4, abs=0.01) and m.closing['across_mm'] == pytest.approx(0.8, abs=0.01)
    assert m.closing['shift_base_mm'] == pytest.approx([0.4, -0.8], abs=0.01)
    assert m.closing['fingers_mm'] == [7.8, 7.6] and m.closing['finger_difference_mm'] == pytest.approx(0.2)
    assert any('Closing: the hand moved +0.80 mm across the brick and +0.40 mm along it' in line for line in m.said) if hasattr(m, 'said') else True
    m.failure, m.stage_pick, m.place_trace, m.blocked, m.grasp_widths = None, None, {}, [], {}
    m.place = None
    m.save_log()
    assert json.loads((tmp_path / 'auto.json').read_text())[-1]['closing']['across_mm'] == pytest.approx(0.8, abs=0.01)


def test_the_shift_follows_the_hand_when_it_is_turned(tmp_path):
    """Hand turned 90 deg: its 'across' is the base x direction."""
    m = closing_mission(tmp_path, after=[0.4010, 0.0, 0.0], yaw_deg=90.0)
    m.check_grasp('after closing')
    assert m.closing['across_mm'] == pytest.approx(1.0, abs=0.01) and m.closing['along_mm'] == pytest.approx(0.0, abs=0.01)


def test_only_the_closing_is_recorded_not_the_lift(tmp_path):
    m = closing_mission(tmp_path, after=[0.4004, -0.0008, 0.0])
    m.check_grasp('after the lift')
    assert getattr(m, 'closing', None) is None


def test_a_diagnostic_that_fails_never_stops_a_pick(tmp_path):
    m = closing_mission(tmp_path, after=[0.4004, -0.0008, 0.0])
    m.mover.hand_pose = lambda: (_ for _ in ()).throw(RuntimeError('tf lost'))
    m.check_grasp('after closing')                                                  # no exception: the pick goes on
    assert any('not recorded' in line for line in m.said)
