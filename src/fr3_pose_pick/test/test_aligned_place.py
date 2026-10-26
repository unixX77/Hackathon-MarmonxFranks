"""The autonomous place by MEASURED position: align over the place, press in, shift and retry when the brick sits on the studs.
A fake arm lands with a different small error for each kind of move, like the real one (numbers from the 16:23 run)."""
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('auto_aligned', ROOT / 'scripts' / 'pose_pick_auto.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)
pp = auto.pp
import smooth_motion as sm  # noqa: E402  (pure helpers, no ROS needed)

CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))
MM = 0.001
STUD = 0.0017            # a stud stands this high: a misaligned brick stops this far above its seat
ENGAGE = 0.0005          # the brick only drops onto its studs within this (x, y)

# where the arm lands relative to the point it was sent to, per kind of move (mm): the pick, and a different place
PICK_ERRORS = {'Hover above the brick': (-2.0, 1.1, -0.7), 'Come down to the inspection height': (-1.8, 1.5, -0.4),
               'Lower to the brick': (-0.6, 1.1, 0.7), 'Lift the brick': (-0.4, 0.5, -1.0)}
# the new place comes down in the same two steps as the pick, so it lands like the pick does (within a few tenths of a mm)
# Here the stage-to-fine descent drags the hand 2.2 mm in x, where the pick's drag was 1.2 mm in x: the 17:23 palletizing run
# lost 2.5 mm that way. Only the second alignment, at the fine stop, brings the brick back onto its seat.
PLACE_ERRORS = {'Carry the brick above the place pose': (-0.5, 0.4, -1.0),
                'Lower the brick to the alignment height': (-1.7, 1.4, -0.5),
                'Align the brick over its place': (-1.8, 1.5, -0.4),
                'Lower the brick to the fine height': (0.5, 1.4, -0.3),
                'Align the brick over its place (fine)': (0.4, 1.3, -0.3),
                'Lower the brick to the place pose': (0.5, 1.4, 0.2),
                'Lift and shift to try again': (0.0, 0.0, 0.0), 'Back off straight up': (-0.3, 0.7, -0.6)}
# what the OLD open-loop place did in the 16:23 run: one 10 cm line from the hover, landing (+0.8, +0.3) mm off
OPEN_LOOP_ERRORS = dict(PLACE_ERRORS, **{'Lower the brick to the place pose': (0.8, 0.3, 0.2)})


class Arm:
    def __init__(self, errors, extra=None):
        self.errors, self.extra = dict(errors), extra or {}
        self.pos = np.array([0.33, -0.05, 0.22])
        self.seat = None                      # (x, y, z) hand position at which the brick is seated, truth
        self.log, self.opened, self.detached = [], 0, 0

    def corrected(self, xyz, yaw):
        ox, oy = sm.hand_frame_to_base(CONFIG['tcp_correction_hand'], yaw)
        return [xyz[0] + ox, xyz[1] + oy, xyz[2]]

    def move(self, label, xyz, yaw=None):
        command = np.array(self.corrected(xyz, 0.0 if yaw is None else yaw))
        error = np.array(self.errors.get(label, (0.0, 0.0, 0.0))) * MM + np.array(self.extra.get(label, (0.0, 0.0, 0.0))) * MM
        landed = command + error
        if label == 'Lower the brick to the place pose' and self.seat is not None:
            miss = math.hypot(landed[0] - self.seat[0], landed[1] - self.seat[1])
            if miss > ENGAGE:
                landed[2] = command[2] + STUD + 0.001          # resting on the studs, 1.7 mm up (the real run: +2.7 mm)
        self.pos = landed
        self.log.append((label, [round(float(v), 5) for v in command]))
        return [float(v) for v in command]

    def hand_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])           # hand pointing down, yaw 0
        pose[:3, 3] = self.pos
        return pose

    def hand(self, close=False, width=None):
        self.opened += 0 if close else 1

    def detach_block(self):
        self.detached += 1


def mission(errors=PLACE_ERRORS, extra=None, place=None, config=None):
    """A mission that has just closed the hand on a brick in layer 2, hand at the pick's measured pose."""
    m = auto.AutoMission.__new__(auto.AutoMission)
    m.c = dict(CONFIG, **(config or {}))
    m.plan = pp.plan_pick(m.c, 0.44, -0.099, pp.layer_centre_z(m.c, 2), 0.0)
    m.place = place or {'offset': (0.0, 0.0), 'xy': None, 'z': 'pick', 'is_tcp': False, 'yaw': None}
    m.place_nudge, m.nudge, m.blocked, m.grasp_widths, m.given = np.zeros(4), np.zeros(4), [], {}, {}
    m.failure, m.place_inspect, m.inspect, m.on_placed, m.place_trace = None, False, False, None, {}
    m.stage_pick = m.grasp_pick = m.truth_target = m.release_hand = None
    m.said = []
    m.say = m.said.append
    m.spin = lambda seconds: None
    m.show_target = lambda: None
    m.mover = Arm(PICK_ERRORS)
    m.go('Hover above the brick', m.plan['above'])
    m.go('Come down to the inspection height', m.plan['inspect'])
    m.go('Lower to the brick', m.plan['grip'])
    m.mover.seat = m.mover.pos.copy()                         # truth: the studs are where the hand was when it closed
    m.mover.errors, m.mover.extra = dict(errors), extra or {}
    return m


# --------------------------------------------------------------------------------------------- the references
def test_the_pick_records_where_the_hand_really_was_not_where_it_was_sent():
    m = mission()
    assert m.stage_pick[:2] == pytest.approx(np.array(m.mover.corrected(m.plan['inspect'], 0.0)[:2]) + np.array([-1.8, 1.5]) * MM)
    assert m.grasp_pick[:2] == pytest.approx(m.mover.seat[:2])
    drift = m.grasp_pick[:2] - m.stage_pick[:2]
    assert drift * 1000 == pytest.approx([1.2, -0.4], abs=0.01)              # the descent drags the hand 1.2 mm in x


def test_only_the_two_pick_moves_are_recorded():
    m = mission()
    before = m.stage_pick.copy()
    m.go('Align the brick over its place', m.plan['inspect'])
    assert m.stage_pick == pytest.approx(before)


# ----------------------------------------------------------------------------------------------- the target
def test_where_the_hand_must_be_at_the_release():
    m = mission()
    assert m.place_hand_target() == pytest.approx(m.grasp_pick[:2])                                      # the same pose
    m.place = {'offset': (0.0, 0.05), 'xy': None, 'z': None, 'is_tcp': False, 'yaw': None}
    assert m.place_hand_target() == pytest.approx(m.grasp_pick[:2] + [0.0, 0.05])                          # 5 cm to the left
    m.truth_target = (0.40, 0.10)
    assert m.place_hand_target() == pytest.approx([0.40, 0.10])                                           # a measured row slot
    m.truth_target, m.grasp_pick = None, None
    with pytest.raises(RuntimeError, match='no measured grasp pose'):
        m.place_hand_target()


def test_aiming_makes_the_commanded_hand_point_the_target():
    m = mission()
    where = m.place_plan()
    target = np.array([0.4412, -0.0963])
    where = m.aim_plan(where, target)
    commanded = m.mover.corrected(where['grip'], m.plan['yaw'])
    assert commanded[:2] == pytest.approx(target)
    assert where['release'][:2] == where['grip'][:2] == where['inspect'][:2] == where['above'][:2]


# -------------------------------------------------------------------------------------------- the whole place
def test_the_brick_goes_back_in_where_the_open_loop_run_put_it_on_the_studs():
    """The brick returns to the measured grasp pose within 0.5 mm, in the studs on the first press."""
    m = mission()
    m.place_brick()
    target = m.grasp_pick[:2]
    assert np.hypot(*(m.release_hand[:2] - target)) < 0.0005
    assert m.mover.opened == 1 and m.mover.detached == 1 and not m.blocked
    assert m.place_trace['search_tries'] == 0 and m.place_trace['coarse_left_mm'] <= 1.0 and m.place_trace['fine_left_mm'] <= 0.3
    labels = [entry[0] for entry in m.mover.log]
    steps = [label for i, label in enumerate(labels) if i == 0 or label != labels[i - 1]]     # repeated corrections counted once
    assert steps[-7:] == ['Carry the brick above the place pose', 'Lower the brick to the alignment height',
                          'Align the brick over its place', 'Lower the brick to the fine height',
                          'Align the brick over its place (fine)', 'Lower the brick to the place pose', 'Back off straight up']
    fine = [entry for entry in m.mover.log if entry[0] == 'Lower the brick to the fine height'][0]
    assert fine[1][2] == pytest.approx(m.place_plan()['release'][2] + CONFIG['place_fine_height'])
    # the same two-step approach as the pick: the alignment height is the pick's inspection height
    align = [entry for entry in m.mover.log if entry[0] == 'Lower the brick to the alignment height'][0]
    assert align[1][2] == pytest.approx(m.plan['inspect'][2])


def test_the_second_alignment_is_what_saves_a_place_where_the_long_descent_drags_differently():
    """17:23: the 4 cm stop was aligned to 0.14 mm, the descent dragged the hand 2.5 mm, the brick sat on the studs."""
    m = mission(config={'place_fine_tries': 0, 'place_fine_tolerance': 1.0, 'place_search_tries': 0})
    with pytest.raises(RuntimeError, match='NOT released'):
        m.place_brick()
    assert m.blocked and m.mover.opened == 0
    m = mission()                                                            # the same arm with the fine alignment
    m.place_brick()
    assert np.hypot(*(m.release_hand[:2] - m.place_hand_target())) < 0.0005 and not m.blocked


def test_without_the_alignment_the_same_run_sits_on_top_of_the_studs():
    """The control: no alignment, no search (the old behaviour) -> blocked 2.7 mm up, as in the log."""
    m = mission(errors=OPEN_LOOP_ERRORS, config={'place_align_tries': 0, 'place_align_tolerance': 1.0, 'place_fine_tries': 0,
                                                  'place_fine_tolerance': 1.0, 'place_search_tries': 0})
    with pytest.raises(RuntimeError, match='NOT released'):
        m.place_brick()
    assert m.blocked and m.blocked[0]['short_mm'] == pytest.approx(2.7, abs=0.05)
    assert m.mover.opened == 0                                              # the hand did not let go in the air


def test_a_brick_on_top_of_the_studs_is_shifted_and_pressed_again():
    # skip the alignment and leave 0.9 mm of error in x: the first try (and the +0.7 mm one) sit on the studs, the -0.7 mm one drops in
    m = mission(errors=dict(PLACE_ERRORS, **{'Lower the brick to the place pose': (0.9, 0.0, 0.0)}),
                config={'place_align_tries': 0, 'place_align_tolerance': 1.0, 'place_fine_tries': 0, 'place_fine_tolerance': 1.0})
    m.place_brick()
    assert m.place_trace['search_tries'] == 2 and len(m.blocked) == 2
    assert np.hypot(*(m.release_hand[:2] - m.mover.seat[:2])) <= ENGAGE
    assert [entry[0] for entry in m.mover.log].count('Lift and shift to try again') == 2
    assert m.mover.opened == 1


def test_when_nothing_works_the_brick_stays_in_the_hand():
    m = mission(extra={'Lower the brick to the place pose': (0.0, 4.0, 0.0)},
                config={'place_align_tries': 0, 'place_align_tolerance': 1.0, 'place_fine_tries': 0, 'place_fine_tolerance': 1.0})
    with pytest.raises(RuntimeError, match=r'NOT released.*after 7 tries'):
        m.place_brick()
    assert len(m.blocked) == 7 and m.mover.opened == 0 and m.mover.detached == 0


def test_an_alignment_stop_far_from_the_target_stops_the_run():
    m = mission(extra={'Lower the brick to the alignment height': (5.0, 0.0, 0.0)}, config={'place_align_max': 0.003})
    with pytest.raises(RuntimeError, match=r'place_align_max.*NOT released'):
        m.place_brick()
    assert m.mover.opened == 0


def test_the_trace_goes_into_the_log(tmp_path):
    import json
    m = mission()
    m.place_brick()
    m.c['auto_log'] = str(tmp_path / 'auto.json')
    m.save_log()
    entry = json.loads((tmp_path / 'auto.json').read_text())[-1]
    assert entry['place']['search_tries'] == 0 and entry['pick_grasp_xy'] == pytest.approx([round(float(v), 5) for v in m.grasp_pick[:2]])
    assert entry['place']['drift_mm'] == pytest.approx([1.2, -0.4], abs=0.01)
