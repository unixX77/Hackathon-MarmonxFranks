"""Bricks lying on other bricks: --layers N|auto and --place-layers."""
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
HAND = cf.pose_matrix([0.40, 0.05, 0.30],
                      (Rotation.from_euler('z', 25, degrees=True) * Rotation.from_euler('x', 180, degrees=True)).as_quat())
HAND_CAM = cf.pose_matrix([0.05, -0.04, -0.045], Rotation.from_euler('z', 90, degrees=True).as_quat())


def camera_point_of(base_point):
    return (np.linalg.inv(HAND @ HAND_CAM) @ np.append(base_point, 1.0))[:3]


def seen_at(top_z, x=0.46, y=0.12):
    """The camera-frame point a top face at base height top_z would have."""
    return camera_point_of(np.array([x, y, top_z]))


def resolve(top_z, layers, **kw):
    return pp.resolve_camera_target(CONFIG, HAND, HAND_CAM, seen_at(top_z), None, 'x', False, 'Pick', layers, **kw)


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


def test_the_15_mm_case_is_refused_as_a_single_brick_and_names_the_layers():
    top = TABLE + H + 0.0154                                                       # what the camera gave: +15.4 mm
    with pytest.raises(ValueError, match='where a brick on the table should be') as problem:
        resolve(top, None)
    assert '2.6' in str(problem.value) and '--layers 3' in str(problem.value)      # 2.6 layers: it says which count to try


def test_a_stacked_brick_uses_the_exact_layer_height_not_the_noisy_measurement():
    for layers in (2, 3, 4):
        measured = pp.layer_top_z(CONFIG, layers) + 0.004                          # camera 4 mm too high, as seen before
        x, y, centre_z, yaw = resolve(measured, layers)
        assert (x, y) == pytest.approx((0.46, 0.12), abs=1e-9)
        assert centre_z == pytest.approx(pp.layer_centre_z(CONFIG, layers))        # snapped, the 4 mm is gone
        assert yaw is None


def test_layer_one_is_the_table_path():
    assert resolve(TABLE + H + 0.003, 1)[2] is None                                # same as the default: table height


def test_wrong_layer_count_is_refused():
    measured = pp.layer_top_z(CONFIG, 3)
    with pytest.raises(ValueError, match='where layer 2 should be'):
        resolve(measured, 2)                                                       # 9.6 mm off
    with pytest.raises(ValueError, match='where layer 5 should be'):
        resolve(measured, 5)


def test_auto_layers_picks_the_clear_cases_and_refuses_the_ambiguous_ones():
    assert resolve(pp.layer_top_z(CONFIG, 3) + 0.002, 'auto')[2] == pytest.approx(pp.layer_centre_z(CONFIG, 3))
    assert resolve(pp.layer_top_z(CONFIG, 1) - 0.002, 'auto')[2] is None
    between = TABLE + 2.5 * H                                                      # halfway between layers 2 and 3
    with pytest.raises(ValueError, match='too close to call') as problem:
        resolve(between, 'auto')
    assert '--layers' in str(problem.value) and '2.50' in str(problem.value)
    with pytest.raises(ValueError, match='too close to call'):
        resolve(pp.layer_top_z(CONFIG, 2) + 0.004, 'auto')                         # 4 mm off one layer: over the 3 mm limit


def test_place_on_a_stack_by_offset_and_by_height():
    pick = pp.plan_pick(CONFIG, 0.396, 0.036, yaw=0.0)
    on_two = pp.layer_centre_z(CONFIG, 3)                                           # lands on top of two bricks
    place = {'offset': (0.0, 0.05), 'xy': None, 'z': on_two, 'is_tcp': False, 'yaw': None}
    where = pp.resolve_place(CONFIG, pick, place)
    assert where['centre'][:2] == pytest.approx([0.396, 0.086])
    assert where['centre'][2] == pytest.approx(on_two)
    low = pp.resolve_place(CONFIG, pick, dict(place, z=None))
    assert where['release'][2] - low['release'][2] == pytest.approx(2 * H)         # released 2 layers higher
