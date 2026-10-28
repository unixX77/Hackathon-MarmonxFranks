"""record_pose.py: reading the hand pose, refusing a moving arm, the poses file, and whether a pose can be the bin."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('rec', ROOT / 'scripts' / 'record_pose.py')
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))


def hand(x, y, z, yaw_deg=0.0, tilt_deg=0.0):
    """A hand pose: pointing down (Rx(pi)), turned by yaw, then tilted."""
    rotation = Rotation.from_euler('z', yaw_deg, degrees=True) * Rotation.from_euler('x', 180.0 + tilt_deg, degrees=True)
    pose = np.eye(4)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = [x, y, z]
    return pose


def test_the_average_of_still_readings_is_recorded_with_the_yaw_and_the_tilt():
    samples = [hand(0.3519 + 0.00002 * i, -0.1058, 0.1831, yaw_deg=0.22) for i in range(-3, 4)]
    record = rec.pose_record(samples)
    assert record['xyz'] == pytest.approx([0.3519, -0.1058, 0.1831], abs=1e-5)
    assert record['yaw_deg'] == pytest.approx(0.22, abs=0.01) and record['tilt_deg'] == pytest.approx(0.0, abs=0.01)
    assert record['spread_mm'] == pytest.approx(0.12, abs=0.01) and len(record['recorded']) == 19


def test_the_tilt_is_how_far_the_hand_is_from_pointing_down():
    assert rec.pose_record([hand(0.4, 0.0, 0.2, tilt_deg=7.0)] * 3)['tilt_deg'] == pytest.approx(7.0, abs=0.01)
    assert rec.pose_record([hand(0.4, 0.0, 0.2, yaw_deg=-40.0)] * 3)['yaw_deg'] == pytest.approx(-40.0, abs=0.01)


def test_a_moving_arm_is_refused():
    moving = [hand(0.35 + 0.0004 * i, -0.1, 0.18) for i in range(20)]                 # 7.6 mm over the reading
    with pytest.raises(ValueError, match='The arm is moving'):
        rec.pose_record(moving)


def test_a_pose_is_saved_next_to_the_config_and_keeps_the_others(tmp_path):
    path = rec.poses_path(tmp_path / 'pose_pick.yaml')
    assert path == tmp_path / 'poses.yaml' and rec.load_poses(path) == {}
    first = rec.pose_record([hand(0.35, -0.1, 0.18)] * 3)
    second = rec.pose_record([hand(0.4, 0.2, 0.3)] * 3)
    rec.save_pose(path, 'bin', first)
    rec.save_pose(path, 'camera', second)
    assert set(rec.load_poses(path)) == {'bin', 'camera'}
    rec.save_pose(path, 'bin', second)                                                  # recording again replaces it
    assert rec.load_poses(path)['bin']['xyz'] == second['xyz'] and rec.load_poses(path)['camera']['xyz'] == second['xyz']


def test_a_damaged_poses_file_is_an_error_not_an_empty_list(tmp_path):
    path = tmp_path / 'poses.yaml'
    for text in ('bin: [1, 2', 'bin: 5', 'bin:\n  yaw_deg: 3\n'):
        path.write_text(text)
        with pytest.raises(ValueError, match='damaged'):
            rec.load_poses(path)


@pytest.mark.parametrize('record, message', [
    ({'xyz': [0.35, -0.1, 0.18], 'tilt_deg': 0.7}, None),
    ({'xyz': [0.35, -0.1, 0.18], 'tilt_deg': 25.0}, 'tilted'),
    ({'xyz': [0.9, 0.3, 0.18], 'tilt_deg': 0.0}, 'out of reach'),
    ({'xyz': [-0.2, 0.1, 0.18], 'tilt_deg': 0.0}, 'out of reach'),
    ({'xyz': [0.35, -0.1, -0.015], 'tilt_deg': 0.0}, 'too close to the table'),
    ({'xyz': [0.35, -0.1, 0.70], 'tilt_deg': 0.0}, 'ceiling'),
])
def test_what_can_be_a_bin(record, message):
    if message is None:
        rec.check_bin_pose(CONFIG, record)
    else:
        with pytest.raises(ValueError, match=message):
            rec.check_bin_pose(CONFIG, record)


def test_a_pose_can_be_nudged_and_the_measurement_and_the_nudges_are_kept():
    record = rec.pose_record([hand(0.35189, -0.10578, 0.18307)] * 3)
    once = rec.shift_pose(record, 0.0, -0.002)
    assert once['xyz'] == pytest.approx([0.35189, -0.10778, 0.18307]) and once['xyz_recorded'] == record['xyz']
    twice = rec.shift_pose(once, 0.001, 0.0, 0.0005)
    assert twice['xyz'] == pytest.approx([0.35289, -0.10778, 0.18357]) and twice['xyz_recorded'] == record['xyz']      # still the first measurement
    assert [(n['dx'], n['dy'], n['dz']) for n in twice['nudges']] == [(0.0, -0.002, 0.0), (0.001, 0.0, 0.0005)]
    summed = np.array(twice['xyz_recorded']) + np.sum([[n['dx'], n['dy'], n['dz']] for n in twice['nudges']], axis=0)
    assert twice['xyz'] == pytest.approx(list(summed), abs=1e-5)                       # xyz = the measurement + the nudges, always
    assert record['xyz'] == [0.35189, -0.10578, 0.18307] and 'nudges' not in record    # the original record is not changed


def test_a_big_shift_is_refused():
    record = rec.pose_record([hand(0.35, -0.1, 0.18)] * 3)
    with pytest.raises(ValueError, match='at most 2 cm'):
        rec.shift_pose(record, 0.0, -0.05)
    with pytest.raises(ValueError, match='at most 2 cm'):
        rec.shift_pose(record, 0.0, -2.0)                                              # millimetres typed as metres


def test_shift_from_the_command_line(tmp_path, capsys):
    config = tmp_path / 'pose_pick.yaml'
    config.write_text(yaml.safe_dump(CONFIG))
    rec.save_pose(rec.poses_path(config), 'bin', rec.pose_record([hand(0.35189, -0.10578, 0.18307)] * 3))
    assert rec.main(['bin', '--shift', '0', '-0.002', '--config', str(config)]) == 0
    assert 'dy=-2.0 mm' in capsys.readouterr().out
    saved = rec.load_poses(rec.poses_path(config))['bin']
    assert saved['xyz'] == pytest.approx([0.35189, -0.10778, 0.18307]) and len(saved['nudges']) == 1
    assert rec.main(['nope', '--shift', '0', '0.001', '--config', str(config)]) == 1                 # no such pose
    assert rec.main(['bin', '--shift', '0.5', '--config', str(config)]) == 1                         # one number
    assert rec.main(['bin', '--shift', '0', '0.5', '--config', str(config)]) == 1                    # 50 cm
    # a nudge that would make the bin unusable is refused and nothing is saved
    assert rec.main(['bin', '--shift', '0', '0', '-0.019', '--config', str(config)]) == 0            # still above the table: fine
    assert rec.main(['bin', '--shift', '0', '0', '-0.02', '--config', str(config)]) == 0
    assert rec.main(['bin', '--shift', '0', '0', '-0.02', '--config', str(config)]) == 0
    before = rec.load_poses(rec.poses_path(config))['bin']['xyz']
    for _ in range(10):                                                                           # walk it down to the table
        if rec.main(['bin', '--shift', '0', '0', '-0.02', '--config', str(config)]) == 1:
            break
    assert rec.load_poses(rec.poses_path(config))['bin']['xyz'][2] >= CONFIG['table_top_z'] + 0.03
    assert rec.load_poses(rec.poses_path(config))['bin']['xyz'][2] < before[2]


def test_show_and_delete_need_no_robot(tmp_path, capsys):
    config = tmp_path / 'pose_pick.yaml'
    assert rec.main(['--show', '--config', str(config)]) == 0 and 'No poses recorded' in capsys.readouterr().out
    rec.save_pose(rec.poses_path(config), 'bin', rec.pose_record([hand(0.35, -0.1, 0.18)] * 3))
    assert rec.main(['--show', '--config', str(config)]) == 0
    assert 'bin: x=0.3500 y=-0.1000 z=0.1800' in capsys.readouterr().out
    assert rec.main(['--delete', 'bin', '--config', str(config)]) == 0 and rec.load_poses(rec.poses_path(config)) == {}
    assert rec.main(['--delete', 'bin', '--config', str(config)]) == 1


def test_the_recorded_bin_is_installed_and_executable():
    assert 'scripts/record_pose.py' in (ROOT / 'CMakeLists.txt').read_text()
    assert (ROOT / 'scripts' / 'record_pose.py').stat().st_mode & 0o111
