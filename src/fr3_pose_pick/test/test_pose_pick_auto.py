"""The autonomous script: no questions, a countdown instead of GO, its own log, and a stop-and-explain failure path."""
import builtins
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('pp_auto', ROOT / 'scripts' / 'pose_pick_auto.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)
pp = auto.pp
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))


# -------------------------------------------------------------------------------------------- the countdown
def test_the_go_question_becomes_a_countdown(monkeypatch, capsys):
    waited = []
    monkeypatch.setattr(auto.time, 'sleep', waited.append)
    answer = auto.countdown_input(4)('Hold the e-stop, clear the workspace. Type GO to start: ')
    assert answer == 'GO' and waited == [1.0] * 4
    out = capsys.readouterr().out
    assert 'starts in 4 s' in out and 'Ctrl+C or the e-stop cancels' in out and '1...' in out


@pytest.mark.parametrize('prompt', ['  correction> ', '  place correction> ',
                                    '  Enter = do it | a = run the rest automatically | q = stop here: ', ''])
def test_no_other_question_is_ever_answered(monkeypatch, prompt):
    monkeypatch.setattr(auto.time, 'sleep', lambda s: None)
    with pytest.raises(RuntimeError, match='never asks questions'):
        auto.countdown_input(5)(prompt)


# ------------------------------------------------------------------------------------------- the arguments
def test_countdown_option_is_taken_out_and_checked():
    same = auto.SAME_POSE
    assert auto.parse_own(['--config', 'c.yaml', '--pose', '0.4', '0.1']) == (5, ['--config', 'c.yaml', '--pose', '0.4', '0.1'] + same)
    assert auto.parse_own(['--countdown', '8', '--pose', '0.4', '0.1']) == (8, ['--pose', '0.4', '0.1'] + same)
    assert auto.parse_own(['--pose', '0.4', '0.1', '--countdown=10'])[0] == 10


def test_with_no_place_given_the_brick_goes_back_where_it_was_picked():
    _, rest = auto.parse_own(['--camera-xyz', '0', '0', '0.3'])
    assert rest[-5:] == ['--place-offset', '0', '0', '--place-layers', 'same']            # the same pose and the same layer
    _, rest = auto.parse_own(['--camera-xyz', '0', '0', '0.3', '--place-layers', '2'])
    assert rest[-3:] == ['--place-offset', '0', '0'] and '2' in rest and 'same' not in rest   # a layer you gave is kept
    for given in (['--place', '0.4', '0.2'], ['--place-offset', '0', '0.05'], ['--place-camera-xyz', '0', '0', '0.3'],
                  ['--pallet']):
        _, rest = auto.parse_own(['--camera-xyz', '0', '0', '0.3'] + given)
        assert rest == ['--camera-xyz', '0', '0', '0.3'] + given                          # an explicit place is never touched


def test_the_same_pose_place_really_is_the_pick_pose():
    """Offset 0 0 with layers 'same' through the real planning: same x, y, angle AND height, whatever layer the brick
    was in (the first autonomous run took a brick from layer 2 and tried to put it on the table)."""
    for pick_z in (None, pp.layer_centre_z(CONFIG, 2)):
        plan = pp.plan_pick(CONFIG, 0.4565, 0.0664, pick_z, 0.0)
        place = {'offset': (0.0, 0.0), 'xy': None, 'z': 'pick', 'is_tcp': False, 'yaw': None}
        where = pp.resolve_place(CONFIG, plan, place)
        assert where['grip'] == plan['grip'] and where['centre'] == plan['centre'] and where['yaw'] == plan['yaw']
        assert where['release'][:2] == plan['grip'][:2]
        assert where['release'][2] == pytest.approx(plan['grip'][2] + CONFIG['place_margin'])
        pp.check_offset_place(CONFIG, plan, place)
    on_table = pp.resolve_place(CONFIG, plan, dict(place, z=None))                        # without 'same': always the table
    assert on_table['centre'][2] == pytest.approx(pp.plan_pick(CONFIG, 0.4565, 0.0664)['centre'][2])
    assert on_table['centre'][2] < plan['centre'][2] - 0.009                           # valid for every allowed correction
    for bad in (['--countdown', '2'], ['--countdown', '61'], ['--countdown', 'x'], ['--countdown'], ['--countdown=0']):
        with pytest.raises(ValueError, match='countdown'):
            auto.parse_own(bad)


@pytest.mark.parametrize('flag', ['--step', '--inspect', '--place-inspect'])
def test_options_that_bring_a_question_back_are_refused(flag):
    with pytest.raises(ValueError, match='never asks'):
        auto.parse_own(['--pose', '0.4', '0.1', flag])


def test_main_runs_pose_pick_with_questions_off_and_puts_everything_back(monkeypatch):
    seen = {}

    def fake_main():
        seen['argv'] = list(sys.argv)
        seen['mission'] = pp.Mission
        seen['input'] = builtins.input
        return 0
    original_input, original_mission, original_argv = builtins.input, pp.Mission, list(sys.argv)
    monkeypatch.setattr(pp, 'main', fake_main)
    assert auto.main(['--config', 'c.yaml', '--camera-xyz', '0.0', '0.0', '0.3', '--countdown', '7']) == 0
    assert seen['argv'][1:] == ['--config', 'c.yaml', '--camera-xyz', '0.0', '0.0', '0.3', '--place-offset', '0', '0',
                                '--place-layers', 'same', '--no-inspect', '--no-step']
    assert seen['mission'] is auto.AutoMission
    assert seen['input'] is not original_input
    assert builtins.input is original_input and pp.Mission is original_mission and sys.argv == original_argv   # restored


def test_main_reports_bad_arguments_without_starting_anything(monkeypatch, capsys):
    monkeypatch.setattr(pp, 'main', lambda: pytest.fail('must not start'))
    assert auto.main(['--pose', '0.4', '0.1', '--step']) == 2
    assert auto.main(['--countdown', '1']) == 2
    assert 'ERROR' in capsys.readouterr().err


def test_a_failure_stops_with_code_1_and_an_explanation(monkeypatch, capsys):
    def failing():
        raise RuntimeError('Lower to the brick: the hand stopped +4.3 mm above its target')
    monkeypatch.setattr(pp, 'main', failing)
    assert auto.main(['--pose', '0.4', '0.1']) == 1
    err = capsys.readouterr().err
    assert 'AUTONOMOUS RUN STOPPED' in err and '+4.3 mm' in err and 'not opened or closed automatically' in err and 'still there' in err


def test_ctrl_c_and_a_cancelled_countdown_are_not_failures(monkeypatch, capsys):
    def interrupted():
        raise KeyboardInterrupt
    monkeypatch.setattr(pp, 'main', interrupted)
    assert auto.main(['--pose', '0.4', '0.1']) == 130
    assert 'CANCELLED' in capsys.readouterr().out

    def cancelled():
        raise SystemExit('Cancelled; nothing was moved.')
    monkeypatch.setattr(pp, 'main', cancelled)
    with pytest.raises(SystemExit, match='nothing was moved'):
        auto.main(['--pose', '0.4', '0.1'])


def test_help_prints_the_description_and_starts_nothing(monkeypatch, capsys):
    monkeypatch.setattr(pp, 'main', lambda: pytest.fail('must not start'))
    assert auto.main(['--help']) == 0
    assert 'AUTONOMOUS' in capsys.readouterr().out


# ----------------------------------------------------------------------------------------- mission and log
def mission(tmp_path, place=True):
    m = auto.AutoMission.__new__(auto.AutoMission)
    m.c = dict(CONFIG, auto_log=str(tmp_path / 'auto.json'), offsets_log=str(tmp_path / 'offsets.json'))
    m.plan = pp.plan_pick(CONFIG, 0.40, 0.036)
    m.place = {'offset': (0.0, 0.07), 'xy': None, 'z': None, 'is_tcp': False, 'yaw': None} if place else None
    m.place_nudge, m.nudge = np.zeros(4), np.zeros(4)
    m.blocked, m.given, m.failure, m.grasp_widths = [], {'camera_xyz': [0.0, 0.0, 0.3]}, None, {}
    m.say = lambda text: None
    return m


def test_a_run_is_logged_in_its_own_file_and_never_in_the_offsets_log(tmp_path):
    m = mission(tmp_path)
    m.save_log()
    m.save_log()
    entries = json.loads((tmp_path / 'auto.json').read_text())
    assert len(entries) == 2 and entries[0]['autonomous'] is True and entries[0]['result'] == 'done'
    assert entries[0]['failure'] is None and entries[0]['grasp_force'] == CONFIG['grasp_force']
    assert entries[0]['tcp_correction_used'] == CONFIG['tcp_correction_hand']
    assert entries[0]['place_xyz'][2] == pytest.approx(m.plan['grip'][2] + CONFIG['place_margin'])
    assert not (tmp_path / 'offsets.json').exists()                          # measure_grasp_offset.py reads that one


def test_a_stopped_run_is_logged_with_the_reason_and_the_error_still_reaches_the_caller(tmp_path, monkeypatch):
    m = mission(tmp_path, place=False)
    monkeypatch.setattr(pp.Mission, 'run', lambda self: (_ for _ in ()).throw(pp.Blocked('stopped 4.3 mm short', {})))
    with pytest.raises(pp.Blocked, match='4.3 mm'):
        m.run()
    entry = json.loads((tmp_path / 'auto.json').read_text())[-1]
    assert entry['result'] == 'stopped' and 'Blocked: stopped 4.3 mm short' in entry['failure'] and 'place_xyz' not in entry


def test_a_log_that_cannot_be_written_does_not_hide_the_real_problem(tmp_path, monkeypatch, capsys):
    m = mission(tmp_path)
    m.c['auto_log'] = str(tmp_path / 'missing_dir' / 'auto.json')
    monkeypatch.setattr(pp.Mission, 'run', lambda self: (_ for _ in ()).throw(RuntimeError('the real problem')))
    with pytest.raises(RuntimeError, match='the real problem'):
        m.run()
    assert 'could not write the log' in capsys.readouterr().out


def test_a_corrupt_log_file_is_started_again(tmp_path):
    (tmp_path / 'auto.json').write_text('not json')
    m = mission(tmp_path)
    m.save_log()
    assert len(json.loads((tmp_path / 'auto.json').read_text())) == 1


def test_the_interactive_script_is_unchanged_and_still_asks():
    """pose_pick.py keeps its questions: the GO prompt and the two correction prompts are still in it."""
    text = (ROOT / 'scripts' / 'pose_pick.py').read_text()
    assert "input('Hold the e-stop, clear the workspace. Type GO to start: ')" in text
    assert "input('  correction> ')" in text and "input('  place correction> ')" in text
    assert 'pose_pick_auto' not in text


def test_the_script_is_installed_and_executable():
    assert 'scripts/pose_pick_auto.py' in (ROOT / 'CMakeLists.txt').read_text()
    assert (ROOT / 'scripts' / 'pose_pick_auto.py').stat().st_mode & 0o111
