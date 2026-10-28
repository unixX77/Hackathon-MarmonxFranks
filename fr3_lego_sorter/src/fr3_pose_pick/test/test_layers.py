"""Automatic brick height for camera points: on the table, on other bricks, or anywhere else."""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cf = load('camera_frame')
pp = load('pose_pick')
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))
H = CONFIG['block_size'][2]                    # one layer
TABLE = CONFIG['table_top_z']
BIAS = CONFIG['camera_height_bias']            # the camera sees a top face this much too high
SHIFT = CONFIG.get('camera_offset_base') or [0.0, 0.0]     # the base-frame pick correction added to every camera point
HAND = cf.pose_matrix([0.40, 0.05, 0.30],
                      (Rotation.from_euler('z', 25, degrees=True) * Rotation.from_euler('x', 180, degrees=True)).as_quat())
HAND_CAM = cf.pose_matrix([0.05, -0.04, -0.045], Rotation.from_euler('z', 90, degrees=True).as_quat())


def seen(true_top_z, error=0.0, x=0.46, y=0.12):
    """The camera-frame point for a top face at true_top_z, as the real camera reports it (BIAS too high, plus error)."""
    base = np.array([x, y, true_top_z + BIAS + error])
    return (np.linalg.inv(HAND @ HAND_CAM) @ np.append(base, 1.0))[:3]


def resolve(true_top_z, layers=None, error=0.0, use_measured_z=False, info=None):
    return pp.resolve_camera_target(CONFIG, HAND, HAND_CAM, seen(true_top_z, error), None, 'x', use_measured_z, 'Pick',
                                    layers, info)


def test_layer_heights():
    assert pp.layer_centre_z(CONFIG, 1) == pytest.approx(TABLE + H / 2)            # on the table: the old default
    assert pp.layer_top_z(CONFIG, 1) == pytest.approx(TABLE + H)
    assert pp.layer_centre_z(CONFIG, 3) == pytest.approx(TABLE + 2.5 * H)
    assert pp.layer_top_z(CONFIG, 3) - pp.layer_top_z(CONFIG, 2) == pytest.approx(H)
    for bad in (0, -1, 1.5, True, 'a'):
        with pytest.raises(ValueError):
            pp.layer_centre_z(CONFIG, bad)


def test_nearest_layer():
    n, residual, levels = pp.nearest_layer(CONFIG, TABLE + 2 * H + 0.003)
    assert n == 2 and residual == pytest.approx(0.003) and levels == pytest.approx(2.3125)
    assert pp.nearest_layer(CONFIG, TABLE - 0.004)[0] == 1                        # never below layer 1


def test_layers_argument():
    assert pp.layers_arg('3') == 3 and pp.layers_arg('Auto') == 'auto'
    for bad in ('0', '11', 'x', '1.5'):
        with pytest.raises(argparse.ArgumentTypeError):
            pp.layers_arg(bad)


def test_grasp_height_follows_the_layer():
    one = pp.plan_pick(CONFIG, 0.40, 0.0)
    three = pp.plan_pick(CONFIG, 0.40, 0.0, z=pp.layer_centre_z(CONFIG, 3))
    assert three['grip'][2] - one['grip'][2] == pytest.approx(2 * H)
    assert three['above'][2] - one['above'][2] == pytest.approx(2 * H)


def test_the_camera_bias_is_off():
    """The +3.7 mm bias of the first runs was the hand-eye tilt, which differs across the pile; sort_pile.py takes the layer
    from the pipeline's height above the table instead (--layers), so the base-frame height is not corrected."""
    assert BIAS == 0.0


@pytest.mark.parametrize('layers', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('error', [-0.002, 0.0, 0.002])
def test_every_layer_is_found_automatically_and_the_bias_and_noise_are_removed(layers, error):
    info = {}
    x, y, centre_z, yaw = resolve(pp.layer_top_z(CONFIG, layers), error=error, info=info)
    assert (x, y) == pytest.approx((0.46 + SHIFT[0], 0.12 + SHIFT[1]), abs=1e-9)
    if layers == 1:
        assert centre_z is None                                                    # a brick on the table: the old path
    else:
        assert centre_z == pytest.approx(pp.layer_centre_z(CONFIG, layers))        # exact layer height, noise gone
    assert info['layer_used'] == layers
    assert info['top_z_measured_m'] == pytest.approx(pp.layer_top_z(CONFIG, layers) + BIAS + error)


def test_a_height_that_is_not_a_layer_is_used_as_measured():
    """A brick on something that is not a whole brick high: no refusal, the corrected measurement is used."""
    true_top = TABLE + H + 0.015                                                   # 15 mm above a brick on the table
    info = {}
    x, y, centre_z, _ = resolve(true_top, info=info)
    assert centre_z == pytest.approx(true_top - H / 2)
    assert info['layer_used'] is None and info['layers_above_table'] == pytest.approx((H + 0.015) / H)


def test_the_case_that_was_refused_on_the_real_robot_now_works():
    """The real log: camera top face at z=+0.005, a 'table check +15.4 mm'. After the 3.7 mm camera calibration that is
    2.22 layers, within 3 mm of layer 2, so the brick is picked as a layer-2 brick (it used to be refused)."""
    measured_top = 0.0050
    config, bias = dict(CONFIG, camera_height_bias=0.0037), 0.0037          # the calibration of that run
    point = (np.linalg.inv(HAND @ HAND_CAM) @ np.append(np.array([0.378, 0.0117, measured_top]), 1.0))[:3]
    info = {}
    _, _, centre_z, _ = pp.resolve_camera_target(config, HAND, HAND_CAM, point, None, 'x', False, 'Pick', None, info)
    assert info['layer_used'] == 2 and centre_z == pytest.approx(pp.layer_centre_z(config, 2))
    assert info['layers_above_table'] == pytest.approx((measured_top - bias - TABLE) / H)


def test_use_measured_z_never_snaps():
    true_top = pp.layer_top_z(CONFIG, 2) + 0.002                                   # 2 mm above layer 2
    assert resolve(true_top)[2] == pytest.approx(pp.layer_centre_z(CONFIG, 2))                   # snapped
    assert resolve(true_top, use_measured_z=True)[2] == pytest.approx(true_top - H / 2)         # as measured


def test_impossible_heights_are_refused():
    with pytest.raises(ValueError, match='BELOW a brick on the table'):
        resolve(TABLE - 0.010)
    with pytest.raises(ValueError, match='BELOW a brick on the table'):
        resolve(0.0, error=-0.03)                                                  # the capture pose was surely wrong
    with pytest.raises(ValueError, match='max_brick_top_above_table'):
        resolve(TABLE + 0.15)


def test_a_stated_layer_is_checked_strictly():
    measured = pp.layer_top_z(CONFIG, 3)
    assert resolve(measured, 3)[2] == pytest.approx(pp.layer_centre_z(CONFIG, 3))
    with pytest.raises(ValueError, match='where layer 2 should be'):
        resolve(measured, 2)                                                       # 9.6 mm off
    with pytest.raises(ValueError, match='where layer 5 should be'):
        resolve(measured, 5)
    assert resolve(measured, 'auto')[2] == pytest.approx(pp.layer_centre_z(CONFIG, 3))      # 'auto' is the default


def test_place_on_a_stack_by_offset_and_by_height():
    pick = pp.plan_pick(CONFIG, 0.396, 0.036, yaw=0.0)
    on_two = pp.layer_centre_z(CONFIG, 3)                                           # lands on top of two bricks
    place = {'offset': (0.0, 0.05), 'xy': None, 'z': on_two, 'is_tcp': False, 'yaw': None}
    where = pp.resolve_place(CONFIG, pick, place)
    assert where['centre'][:2] == pytest.approx([0.396, 0.086])
    assert where['centre'][2] == pytest.approx(on_two)
    low = pp.resolve_place(CONFIG, pick, dict(place, z=None))
    assert where['release'][2] - low['release'][2] == pytest.approx(2 * H)         # released 2 layers higher


def test_height_correction_at_the_inspection_stop():
    step, yaw_step = CONFIG['max_nudge_step'], CONFIG['max_nudge_yaw_step_deg']
    assert pp.parse_nudge('z 0.002', step, yaw_step) == ('nudge', 0.0, 0.0, 0.0, 0.002)
    assert pp.parse_nudge('x 0.001 z -0.003 yaw 2', step, yaw_step) == ('nudge', 0.001, 0.0, 2.0, -0.003)
    with pytest.raises(ValueError, match='z'):
        pp.parse_nudge('z 0.02', step, yaw_step)                                   # 20 mm in one go: refused
    plan = pp.plan_pick(CONFIG, 0.40, 0.0)
    before = {k: plan[k][2] for k in ('centre', 'grip', 'inspect', 'above')}
    pp.apply_nudge(plan, 0.0, 0.0, 0.0, 0.002)
    for key, z in before.items():
        assert plan[key][2] == pytest.approx(z + 0.002)
    assert plan['grip'][:2] == pytest.approx([0.40, 0.0])


def test_the_teammates_angle_convention_is_fixed_once_in_the_config():
    assert pp.convert_camera_angle(CONFIG, 80.0) == 80.0                       # default: same convention, nothing changes
    assert CONFIG['camera_angle_sign'] == 1 and CONFIG['camera_angle_offset_deg'] == 0.0
    ccw = dict(CONFIG, camera_angle_sign=-1)                                   # his angle is counter-clockwise positive
    assert pp.convert_camera_angle(ccw, 10.0) == -10.0
    short = dict(CONFIG, camera_angle_offset_deg=90.0)                          # his angle is the SHORT side
    assert pp.convert_camera_angle(short, 0.0) == 90.0
    both = dict(CONFIG, camera_angle_sign=-1, camera_angle_offset_deg=90.0)     # e.g. counter-clockwise from vertical
    assert pp.convert_camera_angle(both, 10.0) == 80.0
    for bad in (dict(camera_angle_sign=2), dict(camera_angle_sign=True), dict(camera_angle_sign=0),
                dict(camera_angle_offset_deg='x'), dict(camera_angle_offset_deg=1000), dict(camera_angle_offset_deg=float('nan'))):
        with pytest.raises(ValueError, match='camera_angle'):
            pp.convert_camera_angle(dict(CONFIG, **bad), 0.0)


def test_a_converted_angle_gives_the_same_base_yaw_as_the_angle_in_the_tool_convention():
    cf = load('camera_frame')
    hand = cf.pose_matrix([0.40, 0.0, 0.30], [1.0, 0.0, 0.0, 0.0])
    hand_cam = cf.pose_matrix([0.05, -0.04, -0.045], Rotation.from_euler('z', 90, degrees=True).as_quat())
    ccw = dict(CONFIG, camera_angle_sign=-1)
    his_angle = -80.0                                                          # counter-clockwise positive: this is 80 clockwise
    a = cf.camera_yaw_to_base(hand, hand_cam, cf.image_angle_to_quat(pp.convert_camera_angle(ccw, his_angle)), 'x')[0]
    b = cf.camera_yaw_to_base(hand, hand_cam, cf.image_angle_to_quat(80.0), 'x')[0]
    assert a == pytest.approx(b, abs=1e-9)
