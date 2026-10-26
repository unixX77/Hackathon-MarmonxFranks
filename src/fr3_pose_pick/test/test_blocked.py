"""A hand that is stopped on its way down never closes (pick) or lets go (place): it lifts back and asks for a correction."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('pp_blocked', ROOT / 'scripts' / 'pose_pick.py')
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))


def mission(arrivals, inspect=True, place_inspect=True):
    """A Mission without ROS. `arrivals`: the (dx, dy, dz) error in mm of the hand after each *lowering* move, in order;
    every other move arrives perfectly. Returns the mission and the log of what it did."""
    log, remaining = [], list(arrivals)
    m = pp.Mission.__new__(pp.Mission)
    m.c = dict(CONFIG)
    m.plan = pp.plan_pick(CONFIG, 0.40, 0.036)
    m.place = {'offset': (0.0, 0.07), 'xy': None, 'z': None, 'is_tcp': False, 'yaw': None}
    m.place_nudge, m.nudge, m.blocked = np.zeros(4), np.zeros(4), []
    m.inspect, m.place_inspect, m.on_placed = inspect, place_inspect, None
    m.say = lambda text: log.append(('say', text))
    m.spin = lambda seconds: None
    state = {'hand': np.zeros(3)}

    def move(label, xyz, yaw=None):
        error = np.zeros(3)
        if label.startswith('Lower') and remaining:
            error = np.array(remaining.pop(0)) / 1000.0
        state['hand'] = np.array(xyz) + error
        log.append(('move', label, [round(v, 4) for v in xyz]))
        return list(xyz)
    m.mover = types.SimpleNamespace(move=move, hand_pose=lambda: np.block([[np.eye(3), state['hand'][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    m.correct_offset = lambda: log.append(('correct_pick',))
    m.correct_place = lambda: log.append(('correct_place',))
    return m, log


# ------------------------------------------------------------------------------ detecting it
def test_a_normal_arrival_is_not_blocked():
    m, log = mission([(-1.6, 0.8, -0.4)])                                   # the usual 1-2 mm of tracking error, hand a hair low
    m.go('Lower to the brick', [0.4, 0.04, 0.0], watch_blocked=True)


def test_the_run_from_the_log_would_have_been_blocked():
    """The real run: 'arrived: 4.9 mm sideways (dx=+4.5 dy=+2.0), +5.3 mm in height' - and then it closed the hand."""
    m, _ = mission([(4.5, 2.0, 5.3)])
    with pytest.raises(pp.Blocked, match='something is in the way') as stopped:
        m.go('Lower to the brick', [0.4, 0.04, 0.0], watch_blocked=True)
    assert stopped.value.info['short_mm'] == pytest.approx(5.3) and stopped.value.info['sideways_mm'] == pytest.approx(4.9, abs=0.05)


@pytest.mark.parametrize('arrival', [(0.0, 0.0, 2.5), (3.8, 0.0, 0.0), (0.0, 3.6, -1.0)])
def test_either_stopped_short_or_pushed_aside_counts(arrival):
    m, _ = mission([arrival])
    with pytest.raises(pp.Blocked):
        m.go('Lower to the brick', [0.4, 0.04, 0.0], watch_blocked=True)


def test_other_moves_only_warn_as_before():
    m, log = mission([])
    m.mover.move = lambda label, xyz, yaw=None: (_ for _ in ()).throw(AssertionError) if False else list(xyz)
    state = np.array([0.4045, 0.04, 0.0053])                               # 4.5 mm off, 5.3 mm high - but not a lowering
    m.mover.hand_pose = lambda: np.block([[np.eye(3), state[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    m.go('Hover above the brick', [0.4, 0.04, 0.0])                        # no watch_blocked: a warning, no exception
    assert any('WARNING' in e[1] for e in log if e[0] == 'say')


# ----------------------------------------------------------------------------- the pick recovers
def test_a_blocked_lowering_never_closes_lifts_back_and_asks_again():
    m, log = mission([(4.5, 2.0, 5.3), (-1.6, 0.8, -0.4)])
    m.lower_to_brick()
    steps = [e[1] if e[0] == 'move' else e[0] for e in log if e[0] in ('move', 'correct_pick')]
    assert steps == ['Lower to the brick', 'Lift back to the inspection height', 'correct_pick', 'Lower to the brick']
    lifted = [e for e in log if e[0] == 'move' and e[1].startswith('Lift back')][0]
    assert lifted[2] == pytest.approx(m.plan['inspect'], abs=1e-4)
    assert len(m.blocked) == 1 and m.blocked[0]['short_mm'] == pytest.approx(5.3)


def test_blocked_without_the_inspection_stop_stops_the_run_with_the_hand_open():
    m, log = mission([(0.0, 0.0, 4.0)], inspect=False)
    with pytest.raises(RuntimeError, match='NOT closed'):
        m.lower_to_brick()
    assert not any(e[0] == 'correct_pick' for e in log)


def test_three_blocked_tries_stop_the_run():
    m, log = mission([(0, 0, 4.0), (0, 0, 4.0), (0, 0, 4.0)])
    with pytest.raises(RuntimeError, match='NOT closed'):
        m.lower_to_brick()
    assert sum(1 for e in log if e[0] == 'correct_pick') == 2 and len(m.blocked) == 3


# ----------------------------------------------------------------------------- the place recovers
def test_a_blocked_release_lowering_never_lets_go_in_the_air():
    m, log = mission([(0.0, 0.0, 4.0), (0.0, 0.0, -0.3)])
    where = m.place_plan()
    result = m.lower_for_release(where)
    steps = [e[1] if e[0] == 'move' else e[0] for e in log if e[0] in ('move', 'correct_place')]
    assert steps == ['Lower the brick to the place pose', 'Lift back to the place inspection height', 'correct_place',
                     'Lower the brick to the place pose']
    assert result['release'] == pytest.approx(where['release'])


def test_blocked_release_without_the_place_stop_keeps_the_brick():
    m, _ = mission([(0.0, 0.0, 4.0)], place_inspect=False)
    with pytest.raises(RuntimeError, match='NOT released'):
        m.lower_for_release(m.place_plan())


# ------------------------------------------------------------------------ the optional wide opening
def test_pick_open_width_must_be_between_the_open_width_and_8_cm():
    pp.check_pick_open_width(CONFIG)                                       # not set: fine
    pp.check_pick_open_width(dict(CONFIG, pick_open_width=0.026))
    for bad in (0.010, 0.09, 'x', True, 26):
        with pytest.raises(ValueError, match='pick_open_width'):
            pp.check_pick_open_width(dict(CONFIG, pick_open_width=bad))


def test_the_hand_opens_to_the_requested_width():
    pytest.importorskip('franka_msgs')
    spec2 = importlib.util.spec_from_file_location('sm_blocked', ROOT / 'scripts' / 'smooth_motion.py')
    sm = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(sm)
    sent = []

    class Client:
        def send_goal_async(self, goal):
            sent.append(round(goal.width, 4))
            return types.SimpleNamespace(accepted=True, get_result_async=lambda: 'result')

    mover = sm.Mover.__new__(sm.Mover)
    mover.c = dict(CONFIG)
    mover.ask_step = lambda label, detail: None
    mover.move_client = mover.grasp_client = Client()
    calls = []

    def wait(future, timeout):
        calls.append(future)
        if future == 'result':
            return types.SimpleNamespace(status=4, result=types.SimpleNamespace(success=True))
        return future
    mover.wait = wait
    mover.hand()                                                           # default: open_width
    mover.hand(width=0.026)                                                # the wider pick opening
    mover.hand(close=True)                                                 # grasp_width
    assert sent == [round(CONFIG['open_width'], 4), 0.026, round(CONFIG['grasp_width'], 4), round(CONFIG['grasp_width'], 4)]   # the close is gentle, then firm
