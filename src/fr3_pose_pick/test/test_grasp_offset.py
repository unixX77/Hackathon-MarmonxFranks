"""Grasp centring: tcp_correction_hand, the hand-frame maths, and the measuring helper."""
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sm = load('smooth_motion')
mg = load('measure_grasp_offset')
CONFIG_PATH = ROOT / 'config' / 'pose_pick.yaml'


@pytest.mark.parametrize('yaw_deg', [-170, -90, -30, 0, 25, 90, 135])
def test_hand_frame_vector_matches_the_real_hand_rotation(yaw_deg):
    yaw = math.radians(yaw_deg)
    rotation = Rotation.from_quat(sm.tool_down_quaternion(yaw)).as_matrix()      # the hand the arm is told to hold
    offset = (0.003, -0.002)
    assert sm.hand_frame_to_base(offset, yaw) == pytest.approx((rotation[:2, :2] @ np.array(offset)).tolist(), abs=1e-12)


def test_at_yaw_zero_along_is_base_x_and_across_is_minus_base_y():
    assert sm.hand_frame_to_base((0.002, 0.0), 0.0) == pytest.approx((0.002, 0.0))
    assert sm.hand_frame_to_base((0.0, 0.002), 0.0) == pytest.approx((0.0, -0.002))
    assert sm.hand_frame_to_base((0.002, 0.0), math.pi / 2) == pytest.approx((0.0, 0.002))      # turned 90 deg: along = base y


@pytest.mark.parametrize('yaw_deg', [-80, 0, 33, 90, 150])
def test_base_to_hand_is_the_inverse(yaw_deg):
    yaw = math.radians(yaw_deg)
    base = sm.hand_frame_to_base((0.0017, -0.0028), yaw)
    assert mg.base_to_hand(*base, yaw) == pytest.approx((0.0017, -0.0028))


def run(true_offset, used, yaw_deg, noise=(0.0, 0.0)):
    """What pose_pick.py would log if the true centring offset were `true_offset` while `used` was in force:
    the operator types the base-frame correction that brings the fingers to the centre."""
    needed = (true_offset[0] - used[0] + noise[0], true_offset[1] - used[1] + noise[1])
    dx, dy = sm.hand_frame_to_base(needed, math.radians(yaw_deg))
    return {'correction_dx_dy_m': [dx, dy], 'correction_yaw_deg': 0.0, 'hand_yaw_deg': yaw_deg,
            'tcp_correction_used': list(used)}


def test_a_known_offset_is_recovered_whatever_the_brick_angle():
    true = (0.0025, -0.0012)
    entries = [run(true, (0.0, 0.0), yaw) for yaw in (0, 30, -45, 90, 160)]
    result = mg.summarise(entries)
    assert result['n'] == 5
    assert result['mean'] == pytest.approx(true, abs=1e-12)
    assert result['std'] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_measuring_again_with_a_correction_already_in_force_refines_it():
    true = (0.0025, -0.0012)
    entries = [run(true, (0.0020, -0.0010), yaw, noise) for yaw, noise in ((0, (0.0003, 0.0)), (20, (-0.0003, 0.0)))]
    assert mg.summarise(entries)['mean'] == pytest.approx(true, abs=1e-9)         # noise averages out; used value is added back


def test_runs_logged_without_a_hand_yaw_are_ignored():
    old = {'correction_dx_dy_m': [0.005, 0.0], 'correction_yaw_deg': 0.0}
    assert mg.implied_correction(old) is None
    assert mg.summarise([old]) is None
    assert mg.summarise([old, run((0.001, 0.0), (0.0, 0.0), 0)])['n'] == 1


def test_config_line_is_written_in_place_and_the_old_file_kept(tmp_path):
    config = tmp_path / 'c.yaml'
    config.write_text('a: 1\ntcp_correction_hand: [0.0, 0.0]\nb: 2\n')
    assert mg.apply_to_config(config, (0.0025, -0.0012))
    assert yaml.safe_load(config.read_text()) == {'a': 1, 'tcp_correction_hand': [0.0025, -0.0012], 'b': 2}
    assert (tmp_path / 'c.yaml.bak').read_text().startswith('a: 1\ntcp_correction_hand: [0.0, 0.0]')
    bare = tmp_path / 'd.yaml'
    bare.write_text('a: 1\n')
    assert not mg.apply_to_config(bare, (0.001, 0.0))                             # nothing to replace: touch nothing
    assert bare.read_text() == 'a: 1\n'


def test_the_command_line_end_to_end(tmp_path):
    log = tmp_path / 'log.json'
    log.write_text(json.dumps([run((0.002, 0.001), (0.0, 0.0), yaw) for yaw in (0, 15, -20)]))
    config = tmp_path / 'c.yaml'
    config.write_text('tcp_correction_hand: [0.0, 0.0]\n')
    out = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts' / 'measure_grasp_offset.py'), '--log', str(log),
                          '--apply', str(config)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 'tcp_correction_hand: [0.0020, 0.0010]' in out.stdout
    assert yaml.safe_load(config.read_text())['tcp_correction_hand'] == [0.002, 0.001]
    empty = tmp_path / 'empty.json'
    empty.write_text('[]')
    bad = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts' / 'measure_grasp_offset.py'), '--log', str(empty)],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and 'No usable runs' in bad.stderr


def test_the_stored_correction_is_valid_and_small():
    """The value in the shipped config must pass the same check the script runs at start (two numbers, under 10 mm)."""
    config = yaml.safe_load(open(CONFIG_PATH, encoding='utf-8'))
    value = config['tcp_correction_hand']
    assert len(value) == 2 and max(abs(value[0]), abs(value[1])) <= 0.010
    pose_pick = load('pose_pick')
    pose_pick.check_tcp_correction(config)
    assert value == pytest.approx([-0.0100, 0.0], abs=1e-9)          # the stored real measurement


def test_move_commands_the_corrected_point_and_returns_it():
    """Mover.move with a fake MoveIt: the straight line must be requested to the corrected point."""
    pytest.importorskip('moveit_msgs')
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    requests, executed = [], []
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.0] * 7),
                                          JointTrajectoryPoint(positions=[0.05] + [0.0] * 6)]
    response = types.SimpleNamespace(fraction=1.0, error_code=types.SimpleNamespace(val=1), solution=trajectory)
    mover = sm.Mover.__new__(sm.Mover)                       # no ROS node: only the planning path is exercised
    mover.c = {'base_frame': 'fr3_link0', 'ee_frame': 'fr3_hand_tcp', 'tcp_correction_hand': [0.002, 0.001]}
    mover.scale = 0.05
    mover.node = types.SimpleNamespace(get_logger=lambda: types.SimpleNamespace(info=lambda message: None))
    mover.path_client = types.SimpleNamespace(call_async=lambda request: requests.append(request) or response)
    mover.wait = lambda future, timeout: future
    start = np.eye(4)
    start[:3, :3] = Rotation.from_quat(sm.tool_down_quaternion(0.0)).as_matrix()
    start[:3, 3] = [0.30, 0.0, 0.30]
    mover.hand_pose = lambda: start
    mover.ask_step = lambda label, detail: None
    mover.execute = lambda label, trajectory: executed.append(label)
    commanded = mover.move('Test', [0.40, 0.10, 0.05], yaw=0.0)
    sent = requests[0].waypoints[0].position
    # hand yaw 0: along = +x, across = -y  ->  x + 0.002, y - 0.001
    assert (sent.x, sent.y, sent.z) == pytest.approx((0.402, 0.099, 0.05))
    assert commanded == pytest.approx([0.402, 0.099, 0.05])
    assert executed == ['Test']
    mover.c['tcp_correction_hand'] = None                    # no correction configured: exactly the old behaviour
    assert mover.move('Test', [0.40, 0.10, 0.05], yaw=0.0) == pytest.approx([0.40, 0.10, 0.05])
