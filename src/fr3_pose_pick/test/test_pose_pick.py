"""Pure-logic checks for pose_pick: grasp heights, workspace limits, yaw handling."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pp', ROOT / 'scripts' / 'pose_pick.py')
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
CONFIG = yaml.safe_load(open(ROOT / 'config' / 'pose_pick.yaml', encoding='utf-8'))


def test_brick_on_table_defaults():
    plan = pp.plan_pick(CONFIG, 0.45, 0.10)
    top, h = CONFIG['table_top_z'], CONFIG['block_size'][2]
    assert plan['centre'][2] == pytest.approx(top + h / 2)
    assert plan['grip'][2] == pytest.approx(top + CONFIG['grasp_tcp_height'])
    assert plan['above'][2] == pytest.approx(plan['grip'][2] + CONFIG['clearance'])
    assert plan['block_in_tcp'][2] == pytest.approx(CONFIG['grasp_tcp_height'] - h / 2)


def test_explicit_centre_z_and_tcp_z_agree():
    by_centre = pp.plan_pick(CONFIG, 0.4, 0.0, z=-0.0152)
    by_tcp = pp.plan_pick(CONFIG, 0.4, 0.0, z=by_centre['grip'][2], pose_is_tcp=True)
    assert by_tcp['centre'][2] == pytest.approx(by_centre['centre'][2])


@pytest.mark.parametrize('x, y', [(0.1, 0.0), (0.7, 0.0), (0.4, 0.5), (0.4, -0.5)])
def test_pose_outside_workspace_is_refused(x, y):
    with pytest.raises(ValueError, match='outside the allowed window'):
        pp.plan_pick(CONFIG, x, y)


def test_fingers_never_planned_into_the_table():
    with pytest.raises(ValueError):
        pp.plan_pick(CONFIG, 0.4, 0.0, z=CONFIG['table_top_z'] - 0.01)
    with pytest.raises(ValueError, match='below the table'):
        pp.plan_pick(CONFIG, 0.4, 0.0, z=CONFIG['table_top_z'] + 0.005, pose_is_tcp=True)
    strict = dict(CONFIG, min_tcp_above_table=0.025)       # fingertip rule on its own
    with pytest.raises(ValueError, match='fingertips'):
        pp.plan_pick(strict, 0.4, 0.0)


def test_hover_too_high_and_missing_tcp_z():
    with pytest.raises(ValueError, match='max_hover_z'):
        pp.plan_pick(CONFIG, 0.4, 0.0, z=0.6)
    with pytest.raises(ValueError, match='needs a z'):
        pp.plan_pick(CONFIG, 0.4, 0.0, pose_is_tcp=True)


def test_quaternion_yaw_and_line_wrap():
    yaw, tilt = pp.yaw_from_quaternion((0, 0, math.sin(0.3), math.cos(0.3)))      # 0.6 rad about z
    assert yaw == pytest.approx(0.6) and tilt == pytest.approx(0, abs=1e-6)
    _, tilt = pp.yaw_from_quaternion((math.sin(0.2), 0, 0, math.cos(0.2)))
    assert tilt == pytest.approx(math.degrees(0.4))
    assert pp.wrap_line_angle(math.radians(200)) == pytest.approx(math.radians(20))
    assert pp.wrap_line_angle(math.radians(-100)) == pytest.approx(math.radians(80))
    assert pp.wrap_line_angle(math.pi / 2) == pytest.approx(math.pi / 2)
    with pytest.raises(ValueError):
        pp.yaw_from_quaternion((0, 0, 0, 0))


def test_task_config_has_what_the_mover_needs():
    cfg = pp.task_config(CONFIG, pp.plan_pick(CONFIG, 0.45, 0.10))
    for key in ('pick', 'place', 'block_size', 'table_center', 'table_size', 'grasp_width', 'open_width',
                'velocity_scale', 'base_frame', 'ee_frame'):
        assert key in cfg


def test_inspection_height_sits_between_grasp_and_hover():
    plan = pp.plan_pick(CONFIG, 0.45, 0.10)
    assert plan['grip'][2] < plan['inspect'][2] < plan['above'][2]
    assert plan['inspect'][2] - plan['grip'][2] == pytest.approx(CONFIG['inspect_height'])
    with pytest.raises(ValueError, match='inspect_height'):
        pp.plan_pick(dict(CONFIG, inspect_height=0.2), 0.45, 0.10)


def test_correction_commands_and_limits():
    step, yaw_step = CONFIG['max_nudge_step'], CONFIG['max_nudge_yaw_step_deg']
    assert pp.parse_nudge('ok', step, yaw_step) == ('ok',)
    assert pp.parse_nudge('Q', step, yaw_step) == ('quit',)
    assert pp.parse_nudge('x 0.002', step, yaw_step) == ('nudge', 0.002, 0.0, 0.0, 0.0)
    assert pp.parse_nudge('y -0.001, yaw 5', step, yaw_step) == ('nudge', 0.0, -0.001, 5.0, 0.0)
    for bad in ('', 'x', 'w 0.001', 'z 0.02', 'x abc', 'x 0.05', 'yaw 40', 'y 0.011'):
        with pytest.raises(ValueError):
            pp.parse_nudge(bad, step, yaw_step)


def test_apply_nudge_moves_xy_and_yaw_but_not_heights():
    plan = pp.plan_pick(CONFIG, 0.45, 0.10, yaw=math.radians(10))
    heights = [plan[k][2] for k in ('centre', 'grip', 'inspect', 'above')]
    pp.apply_nudge(plan, 0.003, -0.002, 5)
    for key in ('centre', 'grip', 'inspect', 'above'):
        assert plan[key][:2] == pytest.approx([0.453, 0.098])
    assert [plan[k][2] for k in ('centre', 'grip', 'inspect', 'above')] == heights
    assert math.degrees(plan['yaw']) == pytest.approx(15)


def test_hand_tilt():
    from scipy.spatial.transform import Rotation
    down = Rotation.from_euler('x', math.pi)
    assert pp.hand_tilt_deg(down.as_matrix()) == pytest.approx(0, abs=1e-6)
    assert pp.hand_tilt_deg((Rotation.from_euler('y', 30, degrees=True) * down).as_matrix()) == pytest.approx(30)
    assert pp.hand_tilt_deg(Rotation.identity().as_matrix()) == pytest.approx(180)      # pointing up


def test_base_pad_sits_between_robot_base_and_table():
    c = CONFIG
    size, centre = c['base_pad_size'], c['base_pad_center']
    assert size[2] == pytest.approx(0.02)                                   # 2 cm tall
    assert centre[2] + size[2] / 2 == pytest.approx(0.0)                    # top face = underside of the base
    assert centre[2] - size[2] / 2 == pytest.approx(c['table_top_z'])       # bottom face rests on the table
    # footprint of the FR3 base (link0 collision mesh): x -0.1528..0.0701, y -0.0944..0.0938
    assert (size[0], size[1]) == pytest.approx((0.223, 0.188), abs=0.002)
    assert centre[0] - size[0] / 2 == pytest.approx(-0.1528, abs=0.002)
    assert centre[0] + size[0] / 2 == pytest.approx(0.0701, abs=0.002)


def test_place_plan_stops_above_the_table_by_the_margin():
    plan = pp.plan_place(CONFIG, 0.40, -0.15, math.radians(30))
    grip = CONFIG['table_top_z'] + CONFIG['grasp_tcp_height']
    assert plan['grip'][2] == pytest.approx(grip)
    assert plan['release'][2] == pytest.approx(grip + CONFIG['place_margin'])
    assert plan['release'][:2] == plan['grip'][:2] == [0.40, -0.15]
    assert plan['yaw'] == pytest.approx(math.radians(30))


def test_place_pose_gets_the_same_checks_as_a_pick():
    with pytest.raises(ValueError, match='outside the allowed window'):
        pp.plan_place(CONFIG, 0.40, 0.45, 0.0)
    pp.plan_place(dict(CONFIG, place_margin=-0.003), 0.40, 0.0, 0.0)       # pressed onto the studs, up to place_max_press: fine
    with pytest.raises(ValueError, match='place_margin'):
        pp.plan_place(dict(CONFIG, place_margin=-0.0031), 0.40, 0.0, 0.0)   # more than place_max_press (3 mm): refused
    with pytest.raises(ValueError, match='place_margin'):
        pp.plan_place(dict(CONFIG, place_margin=-0.002, place_max_press=0.001), 0.40, 0.0, 0.0)


def test_place_at_a_given_height_for_stacking():
    one_brick_up = CONFIG['table_top_z'] + CONFIG['block_size'][2] * 1.5
    plan = pp.plan_place(CONFIG, 0.40, 0.0, 0.0, z=one_brick_up)
    assert plan['centre'][2] == pytest.approx(one_brick_up)
    low = pp.plan_place(CONFIG, 0.40, 0.0, 0.0)
    assert plan['release'][2] - low['release'][2] == pytest.approx(CONFIG['block_size'][2])


def test_place_height_can_be_the_hand_height():
    by_tcp = pp.plan_place(CONFIG, 0.40, 0.0, 0.0, z=0.05, pose_is_tcp=True)
    assert by_tcp['grip'][2] == pytest.approx(0.05)
    with pytest.raises(ValueError, match='needs a z'):
        pp.plan_place(CONFIG, 0.40, 0.0, 0.0, pose_is_tcp=True)


def test_place_below_the_table_is_refused():
    with pytest.raises(ValueError, match='below the table'):
        pp.plan_place(CONFIG, 0.40, 0.0, 0.0, z=CONFIG['table_top_z'] - 0.01)


def test_no_detection_or_simulated_brick_in_the_pose_package():
    # camera-frame NUMBERS may be typed in and converted, but nothing here looks at an image or a fake brick
    for name in ('pose_pick.py', 'camera_frame.py'):
        source = (ROOT / 'scripts' / name).read_text()
        for word in ('cv2', 'sensor_msgs.msg import Image', 'cv_bridge', 'sim_brick', 'sim_world'):
            assert word not in source


def test_two_robot_stacks_in_one_domain_are_detected():
    one_stack = ['move_group', 'controller_manager', 'robot_state_publisher', 'rviz2', 'sim_gripper']
    assert pp.duplicate_stack_nodes(one_stack) == []
    both = one_stack + ['move_group', 'controller_manager', 'robot_state_publisher']
    assert pp.duplicate_stack_nodes(both) == ['controller_manager', 'move_group', 'robot_state_publisher']
    assert pp.duplicate_stack_nodes(one_stack + ['move_group']) == ['move_group']


def test_place_offset_follows_the_pick_position():
    pick = pp.plan_pick(CONFIG, 0.396, 0.036, yaw=math.radians(10))
    place = {'offset': (0.0, 0.05), 'xy': None, 'z': None, 'is_tcp': False, 'yaw': None}
    where = pp.resolve_place(CONFIG, pick, place)
    assert where['centre'][:2] == pytest.approx([0.396, 0.086])
    assert where['centre'][2] == pytest.approx(CONFIG['table_top_z'] + CONFIG['block_size'][2] / 2)   # on the table
    assert where['yaw'] == pytest.approx(math.radians(10))                    # keeps the picked angle
    pp.apply_nudge(pick, 0.004, -0.002, 5)                                    # inspection correction moves the place too
    again = pp.resolve_place(CONFIG, pick, place)
    assert again['centre'][:2] == pytest.approx([0.400, 0.084])
    assert again['yaw'] == pytest.approx(math.radians(15))
    assert pp.resolve_place(CONFIG, pick, dict(place, yaw=0.3))['yaw'] == pytest.approx(0.3)


def test_absolute_place_is_unchanged_by_the_pick():
    pick = pp.plan_pick(CONFIG, 0.40, 0.0)
    place = {'offset': None, 'xy': (0.35, -0.2), 'z': None, 'is_tcp': False, 'yaw': math.radians(20)}
    assert pp.resolve_place(CONFIG, pick, place)['centre'][:2] == pytest.approx([0.35, -0.2])


def test_offset_place_is_refused_up_front_if_a_correction_could_push_it_out_of_the_workspace():
    place = {'offset': (0.0, 0.05), 'xy': None, 'z': None, 'is_tcp': False, 'yaw': None}
    pp.check_offset_place(CONFIG, pp.plan_pick(CONFIG, 0.396, 0.036), place)            # fine
    edge = pp.plan_pick(CONFIG, 0.45, 0.25)       # place y=0.30 is still allowed, but not with a +30 mm correction
    with pytest.raises(ValueError, match='corrected by up to'):
        pp.check_offset_place(CONFIG, edge, place)
    with pytest.raises(ValueError):
        pp.check_offset_place(CONFIG, pp.plan_pick(CONFIG, 0.30, 0.0), {**place, 'offset': (-0.05, 0.0)})


def test_hand_widths_are_fixed_in_the_config():
    assert CONFIG['open_width'] == pytest.approx(0.018)
    assert CONFIG['grasp_width'] == pytest.approx(0.015)
    assert CONFIG['grasp_width'] < CONFIG['block_size'][1] < CONFIG['open_width']    # opens wider than the brick, closes on it


def test_jaw_clearance_note_only_when_tight():
    note = pp.jaw_clearance_note(CONFIG)
    assert note and '18 mm' in note and '1.0 mm' in note
    assert pp.jaw_clearance_note(dict(CONFIG, open_width=0.025)) is None


def test_config_check_accepts_the_fixed_widths_and_still_rejects_a_hand_that_is_too_narrow():
    pytest.importorskip('rclpy')                                  # pick_place imports ROS
    spec = importlib.util.spec_from_file_location('pick_place_for_test', ROOT / 'scripts' / 'pick_place.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = pp.task_config(CONFIG, pp.plan_pick(CONFIG, 0.40, 0.0))
    assert module.validate(cfg)                                                       # 18 mm / 15 mm pass
    with pytest.raises(ValueError, match='more than the brick short side'):
        module.validate(dict(cfg, open_width=0.0165))                                 # only 0.5 mm more than the brick
    with pytest.raises(ValueError, match='more than the brick short side'):
        module.validate({k: v for k, v in cfg.items() if k != 'min_open_clearance'})  # without the setting: the 5 mm rule


def test_the_grasp_force_is_strong_enough_for_the_studs_and_capped_below_the_hand_rating():
    """A seated 2x4 needs about 10-25 N to pull off the baseplate; the hand holds 2 x friction x force. Never above 50 N (rated 70)."""
    pytest.importorskip('rclpy')                                  # pick_place imports ROS
    spec = importlib.util.spec_from_file_location('pick_place_force', ROOT / 'scripts' / 'pick_place.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = pp.task_config(dict(CONFIG), pp.plan_pick(CONFIG, 0.40, 0.0))
    assert 30.0 <= CONFIG['grasp_force'] <= 50.0
    assert module.validate(cfg)
    assert module.validate(dict(cfg, grasp_force=50.0))
    for bad in (0.0, -5.0, 50.1, 70.0, '40', True):
        with pytest.raises(ValueError, match='grasp_force'):
            module.validate(dict(cfg, grasp_force=bad))


def test_a_camera_point_with_a_missing_decimal_point_is_explained():
    """The real typo: --camera-xyz 0.087685 0.042182 0260966 (z = 260966 m)."""
    pp.check_camera_xyz('--camera-xyz', [0.087685, 0.042182, 0.260966])
    pp.check_camera_xyz('--camera-xyz', [-0.30, 0.10, 1.2])                   # far but possible
    with pytest.raises(ValueError, match=r'METRES.*Did you mean 0\.260966\?'):
        pp.check_camera_xyz('--camera-xyz', [0.087685, 0.042182, 260966.0])
    with pytest.raises(ValueError, match='millimetres'):
        pp.check_camera_xyz('--place-camera-xyz', [88.0, 42.0, 261.0])
    for bad in ([0.1, 0.1, float('nan')], [0.1, float('inf'), 0.3], [3.0, 0.0, 0.3]):
        with pytest.raises(ValueError):
            pp.check_camera_xyz('--camera-xyz', bad)


def test_the_typo_is_caught_before_the_config_or_ROS_are_touched(monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'argv', ['pose_pick.py', '--config', str(ROOT / 'config' / 'pose_pick.yaml'),
                                      '--camera-xyz', '0.087685', '0.042182', '260966', '--camera-angle', '-87.1'])
    with pytest.raises(ValueError, match='Did you mean 0.260966'):
        pp.main()


def test_tcp_correction_must_be_two_small_numbers_in_metres():
    pp.check_tcp_correction({})                                          # absent: fine
    pp.check_tcp_correction({'tcp_correction_hand': [0.0025, -0.0012]})
    for bad in ([0.0025], [0.0025, 0.0, 0.0], [2.5, 1.0], [0.03, 0.0], ['a', 0.0], [float('nan'), 0.0], 0.002):
        with pytest.raises(ValueError, match='tcp_correction_hand'):
            pp.check_tcp_correction({'tcp_correction_hand': bad})
