"""Offline checks with actual Humble messages; no robot/node/action connection."""
import sys
import json
import math
import copy
import tempfile
from contextlib import nullcontext
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE / 'pick_place_stepwise.py').exists() else HERE.parent / 'scripts'))
import pick_place_stepwise as s
from moveit_msgs.msg import AllowedCollisionMatrix, RobotTrajectory, RobotState, MoveItErrorCodes
from moveit_msgs.action import MoveGroup
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Pose
from pick_place import block_center, block_object, rotate_vector


def trajectory(value=0.):
    t = RobotTrajectory()
    t.joint_trajectory.joint_names = s.ARM_JOINTS
    t.joint_trajectory.points = [JointTrajectoryPoint(positions=[value] * 7)]
    return t


def result():
    r = MoveGroup.Result()
    r.error_code.val = MoveItErrorCodes.SUCCESS
    r.planned_trajectory = trajectory()
    return r


def task(hardware=False):
    t = object.__new__(s.StepwiseTask)
    t.node = MagicMock()
    t.hardware = hardware
    t.execute = False
    t.c = dict(pick=[.45, 0., .1], place=[.45, .2, .1], clearance=.1,
               block_size=[.03, .015, .02], table_center=[.4, 0., 0.],
               table_size=[.6, .6, .05], velocity_scale=.03, acceleration_scale=.03,
               open_width=.08, grasp_width=.015)
    t.joints = dict.fromkeys(s.ARM_JOINTS, 0.)
    t.home = dict(t.joints)
    t.acm = AllowedCollisionMatrix()
    t.arm = object()
    t.executor = object()
    t.commands = []
    t.vertical_starts = {}
    t.label = ''
    t.preview = MagicMock()
    t.vertical = MagicMock()
    t.vertical.plan.side_effect = lambda *a, **kw: (trajectory(), RobotState())
    t.check_endpoint = MagicMock()
    t.capture_grasp_transform = MagicMock()
    return t


class Stages(unittest.TestCase):
    def test_stop_after_grasp_never_prepares_lift(self):
        t = task()
        with patch.object(s.Task, 'action', return_value=result()), patch.object(s, 'confirm') as prompts:
            t.run(stop_after=3)
        self.assertEqual(prompts.call_count, 3)
        self.assertEqual(t.vertical.plan.call_count, 1)  # descent only
        t.capture_grasp_transform.assert_called_once()

    def test_nine_prompts_and_no_execution_in_default_mode(self):
        t = task()
        stages = []
        def accept(label, physical):
            self.assertFalse(physical)
            stages.append([kind for kind, _, _ in t.commands])
        with patch.object(s.Task, 'action', return_value=result()) as actions:
            with patch.object(s, 'confirm', side_effect=accept) as prompts:
                t.run()
        self.assertEqual(prompts.call_count, 9)
        self.assertEqual(stages, [['hand', 'arm'], ['arm'], ['hand'], ['arm'],
                                  ['arm'], ['arm'], ['hand'], ['arm'], ['arm']])
        self.assertEqual(actions.call_count, 3)
        self.assertEqual(t.vertical.plan.call_count, 4)
        self.assertEqual([call.kwargs for call in t.vertical.plan.call_args_list],
                         [dict(carrying=False, placed=False), dict(carrying=True, placed=False),
                          dict(carrying=True, placed=False), dict(carrying=False, placed=True)])
        self.assertEqual([len(call.args[0].trajectory) for call in t.preview.publish.call_args_list],
                         [1, 1, 1, 1, 1, 1, 1])
        for call in actions.call_args_list:
            self.assertIs(call.args[0], t.arm)
            self.assertTrue(call.args[1].planning_options.plan_only)
        # Home uses joint goals, and keeps the placed block/table in the scene.
        goal = actions.call_args.args[1]
        self.assertEqual([j.joint_name for j in goal.request.goal_constraints[0].joint_constraints],
                         s.ARM_JOINTS)
        self.assertEqual({o.id for o in goal.planning_options.planning_scene_diff.world.collision_objects},
                         {'fr3_demo_table', s.BLOCK})

    def test_hardware_still_plans_before_prompt_and_operator_can_stop(self):
        t = task(True)
        q = dict(t.joints)
        events = []
        t.settled_snapshot = MagicMock(return_value=q)
        t.execute_commands = MagicMock(side_effect=lambda: events.append('execute'))
        def accept(label, hardware):
            events.append('prompt')
            self.assertTrue(hardware)
            if label.startswith('2/'):
                raise s.OperatorStop()
        with patch.object(s.Task, 'action', return_value=result()) as actions:
            with patch.object(s, 'confirm', side_effect=accept):
                with self.assertRaises(s.OperatorStop):
                    t.run()
        self.assertEqual(events, ['prompt', 'execute', 'prompt'])
        for call in actions.call_args_list:
            self.assertTrue(call.args[1].planning_options.plan_only)

    def test_transfer_uses_attached_block_without_redundant_world_removal(self):
        t = task()
        with patch.object(s.Task, 'action', return_value=result()) as actions:
            t.prepare_stage(5)
        goal = actions.call_args.args[1]
        scene = goal.planning_options.planning_scene_diff
        self.assertFalse(any(obj.id == s.BLOCK for obj in scene.world.collision_objects))
        self.assertTrue(any(obj.object.id == s.BLOCK for obj in scene.robot_state.attached_collision_objects))

    def test_failed_descent_plan_never_prompts_or_executes_descent(self):
        t = task(True)
        t.settled_snapshot = MagicMock(return_value=dict(t.joints))
        t.execute_commands = MagicMock()
        t.vertical.plan.side_effect = RuntimeError('no plan')
        with patch.object(s.Task, 'action', return_value=result()):
            with patch.object(s, 'confirm') as prompt:
                with self.assertRaisesRegex(RuntimeError, 'no plan'):
                    t.run()
        self.assertEqual(prompt.call_count, 1)  # only the approach was approved
        t.execute_commands.assert_called_once()

    def test_preview_captures_virtual_grasp_but_hardware_waits_for_real_grasp(self):
        for hardware in (False, True):
            t = task(hardware)
            t.prepare_stage(3)
            if hardware:
                t.capture_grasp_transform.assert_not_called()
            else:
                t.capture_grasp_transform.assert_called_once_with(t.joints)

    def test_arm_moved_during_prompt_prevents_execution(self):
        t = task(True)
        changed = dict.fromkeys(s.ARM_JOINTS, .2)
        t.settled_snapshot = MagicMock(side_effect=[dict(t.joints), changed])
        t.execute_commands = MagicMock()
        with patch.object(s.Task, 'action', return_value=result()):
            with patch.object(s, 'confirm'):
                with self.assertRaisesRegex(RuntimeError, 'Arm moved'):
                    t.run()
        t.execute_commands.assert_not_called()

    def test_trajectory_start_guard(self):
        s.check_start(dict.fromkeys(s.ARM_JOINTS, 0.), trajectory())
        with self.assertRaisesRegex(RuntimeError, 'no longer matches'):
            s.check_start(dict.fromkeys(s.ARM_JOINTS, .1), trajectory())
        with self.assertRaises(RuntimeError):
            s.check_start({}, trajectory())

    def test_failed_hand_action_prevents_following_arm_execution(self):
        t = task(True)
        t.execute = True
        t.commands = [('hand', 'Grasp', True), ('arm', 'Lift', trajectory())]
        t.action = MagicMock()
        with patch.object(s.Task, 'hand', side_effect=RuntimeError('grasp failed')):
            with self.assertRaisesRegex(RuntimeError, 'grasp failed'):
                t.execute_commands()
        t.action.assert_not_called()

    def test_execution_uses_approved_trajectory_without_replanning(self):
        t = task(True)
        approved = trajectory()
        t.commands = [('arm', 'Transfer', approved)]
        t.settled_snapshot = MagicMock(return_value=dict(t.joints))
        t.action = MagicMock(return_value=SimpleNamespace(
            error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS)))
        t.execute_commands()
        t.action.assert_called_once()
        self.assertIs(t.action.call_args.args[0], t.executor)
        self.assertEqual(t.action.call_args.args[1].trajectory, approved)
        t.check_endpoint.assert_called_once_with(approved, t.joints)

    def test_each_hardware_stage_uses_fresh_measured_joints(self):
        t = task(True)
        # Accepted tracking offset from the preceding stage must seed the next plan.
        measured = dict.fromkeys(s.ARM_JOINTS, .004)
        t.settled_snapshot = MagicMock(side_effect=lambda: dict(measured))
        t.execute_commands = MagicMock()
        starts = []
        original = t.prepare_stage
        def prepare(stage):
            starts.append(dict(t.joints))
            original(stage)
        t.prepare_stage = prepare
        with patch.object(s.Task, 'action', return_value=result()), patch.object(s, 'confirm'):
            t.run()
        self.assertEqual(starts, [measured]*9)


class CameraInput(unittest.TestCase):
    def test_camera_mode_ignores_old_pose_file_and_fixes_orientation(self):
        c = s.load_config(HERE.parent / 'config/task.yaml',
                          '/does/not/exist/taught.json', camera_mode=True)
        self.assertEqual(c['tcp_orientation_xyzw'], [1., 0., 0., 0.])
        self.assertNotIn('approach_seed_joints', c)
        self.assertTrue(c['camera_ik'])

    def test_camera_ik_uses_current_joints_without_old_posture_bound(self):
        t = task()
        t.c = s.load_config(HERE.parent / 'config/task.yaml', camera_mode=True)
        t.action = MagicMock(return_value=result())
        t.motion('Approach', [.3, 0., .1])
        goal = t.action.call_args.args[1]
        orientation = goal.request.goal_constraints[0].orientation_constraints[0].orientation
        self.assertEqual([orientation.x, orientation.y, orientation.z, orientation.w], [1., 0., 0., 0.])
        t.vertical.scoped_scene.side_effect = lambda *a: nullcontext(RobotState())
        t.vertical.state.return_value = RobotState()
        responses = []
        for angle in (1.2, 1.0, 1.3, 1.4, 1.1):
            state = RobotState()
            state.joint_state.name = s.ARM_JOINTS
            state.joint_state.position = [angle] * 7
            responses.append(SimpleNamespace(solution=state,
                error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS)))
        t.vertical.call.side_effect = responses
        t.seed_pose_goal(goal, False)
        request = t.vertical.call.call_args.args[1]
        self.assertEqual(request.ik_request.constraints.joint_constraints, [])
        self.assertTrue(request.ik_request.avoid_collisions)
        self.assertEqual(t.vertical.state.call_args.args[2], t.joints)
        self.assertEqual(goal.request.goal_constraints[0].joint_constraints[0].position, 1.0)
        t.vertical.call.side_effect = None
        t.vertical.call.return_value = SimpleNamespace(
            error_code=SimpleNamespace(val=MoveItErrorCodes.NO_IK_SOLUTION))
        with self.assertRaisesRegex(RuntimeError, 'No old taught-posture restriction'):
            t.seed_pose_goal(goal, False)

    def test_supplied_color_and_depth_extrinsics_and_top_surface_offset(self):
        path = HERE.parent / 'config/manual_camera_target.yaml'
        for frame, translation in (
                ('camera_color_optical_frame', [.058336, .012374, .06075]),
                ('camera_depth_optical_frame', [.04773, .022981, .06075])):
            with tempfile.TemporaryDirectory() as directory:
                data = s.yaml.safe_load(path.read_text())
                data['camera_frame'] = frame
                data['point_to_block_center_world_m'] = [0., 0., -.00475]
                data['handeye_file'] = str((path.parent / data['handeye_file']).resolve())
                temporary = Path(directory) / 'target.yaml'
                temporary.write_text(s.yaml.safe_dump(data))
                target = s.load_camera_target(temporary)
            c = s.load_config(HERE.parent / 'config/task.yaml')
            # Flange rotated +90 degrees about Z: tests both chained rotations
            # and the translation direction without using implementation math.
            q = [0., 0., math.sqrt(.5), math.sqrt(.5)]
            actual = s.camera_pick_config(c, target, [.3, .2, .5], q)
            x, y, z = target['xyz_m']
            expected = [.3 - translation[1] - (x-y)/math.sqrt(2),
                        .2 + translation[0] - (x+y)/math.sqrt(2),
                        .5 + translation[2] + z - .00475]
            for got, want in zip(block_center(actual), expected):
                self.assertAlmostEqual(got, want, places=6)
            self.assertEqual(actual['place'], c['place'])
            self.assertEqual(actual['tcp_orientation_xyzw'], c['tcp_orientation_xyzw'])
            self.assertEqual(actual['block_center_offset_world'], c['block_center_offset_world'])
            self.assertNotEqual(actual['pick'], c['pick'])

    def test_invalid_frame_nonfinite_values_and_table_penetration_rejected(self):
        path = HERE.parent / 'config/manual_camera_target.yaml'
        original = s.yaml.safe_load(path.read_text())
        original['handeye_file'] = str((path.parent / original['handeye_file']).resolve())
        for field, value in (('camera_frame', 'camera_link'), ('xyz_m', [0., 0., float('nan')]),
                             ('xyz_m', [0., 0., -1.]),
                             ('point_to_block_center_world_m', [0., float('inf'), 0.])):
            with tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory) / 'target.yaml'
                temporary.write_text(s.yaml.safe_dump(dict(original, **{field: value})))
                with self.assertRaises(ValueError):
                    s.load_camera_target(temporary)
        target = s.load_camera_target(path)
        c = s.load_config(HERE.parent / 'config/task.yaml')
        with self.assertRaisesRegex(ValueError, 'intersect the table.*block bottom Z=.*table top Z=.*penetration='):
            s.camera_pick_config(c, target, [0., 0., -2.], [0., 0., 0., 1.])

    def test_resolves_using_flange_fk_once_and_rejects_wrong_frame(self):
        t = task()
        t.c = s.load_config(HERE.parent / 'config/task.yaml')
        t.scene_client = object()
        t.settled_snapshot = MagicMock(return_value=dict(t.joints))
        t.vertical.state.return_value = RobotState()
        pose = s.GetPositionFK.Response()
        pose.error_code.val = MoveItErrorCodes.SUCCESS
        pose.fk_link_names = ['fr3_link8']
        from geometry_msgs.msg import PoseStamped
        stamped = PoseStamped()
        stamped.header.frame_id = s.BASE
        stamped.pose.position.z = .5
        stamped.pose.orientation.w = 1.
        pose.pose_stamped = [stamped]
        scene = SimpleNamespace(scene=SimpleNamespace(robot_state=RobotState()))
        target = s.load_camera_target(HERE.parent / 'config/manual_camera_target.yaml')
        t.vertical.call.side_effect = [scene, pose]
        t.use_camera_target(target)
        request = t.vertical.call.call_args.args[1]
        self.assertEqual(request.fk_link_names, ['fr3_link8'])
        self.assertEqual(request.header.frame_id, s.BASE)
        stamped.header.frame_id = 'wrong_frame'
        t.vertical.call.side_effect = [scene, pose]
        with self.assertRaisesRegex(RuntimeError, 'unexpected frame'):
            t.use_camera_target(target)


class TaughtConfig(unittest.TestCase):
    config_path = HERE.parent / 'config/task.yaml'

    def test_config_separates_low_tcp_from_upright_block_centre(self):
        c = s.load_config(self.config_path)
        self.assertEqual(c['block_size'], [.0315, .0155, .0095])
        top = c['table_center'][2] + c['table_size'][2]/2
        centre = block_center(c)
        self.assertAlmostEqual(centre[2] - c['block_size'][2]/2, top)
        self.assertNotEqual(centre[2], c['pick'][2])
        self.assertAlmostEqual(c['place'][1]-c['pick'][1], .05)
        obj = block_object(c, carrying=True)
        p = obj.primitive_poses[0]
        local = [p.position.x, p.position.y, p.position.z]
        reconstructed = rotate_vector(c['tcp_orientation_xyzw'], local)
        for got, expected in zip(reconstructed, c['block_center_offset_world']):
            self.assertAlmostEqual(got, expected)
        # Local box axes rotated back into world must be axis aligned.
        inv = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        for axis in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]):
            reconstructed_axis = rotate_vector(c['tcp_orientation_xyzw'], rotate_vector(inv, axis))
            for got, expected in zip(reconstructed_axis, axis):
                self.assertAlmostEqual(got, expected)

    def test_json_overrides_targets_without_needing_place_joints(self):
        c = s.load_config(self.config_path)
        pick = {'xyz_m': [.4, .1, .01], 'quaternion_xyzw': [1., 0., 0., 0.]}
        place = {'xyz_m': [.4, .15, .01], 'quaternion_xyzw': [1., 0., 0., 0.], 'arm_joints_rad': None}
        data = {'frame_id': s.BASE, 'tcp_link': s.TCP, 'poses': {'pick': pick, 'place': place}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poses.json'
            path.write_text(json.dumps(data))
            loaded = s.load_config(self.config_path, path)
            self.assertEqual(loaded['pick'], pick['xyz_m'])
            self.assertEqual(loaded['place'], place['xyz_m'])
            self.assertEqual(loaded['tcp_orientation_xyzw'], pick['quaternion_xyzw'])
            self.assertAlmostEqual(block_center(loaded)[2], block_center(c)[2])
            data['frame_id'] = 'wrong_frame'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'fr3_link0'):
                s.load_config(self.config_path, path)
            data['frame_id'] = s.BASE
            place['quaternion_xyzw'] = [0., 0., 0., 1.]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'same pick/place orientation'):
                s.load_config(self.config_path, path)

    def test_taught_orientation_is_sent_to_pose_planner(self):
        t = task()
        t.c = s.load_config(self.config_path)
        with patch.object(s.Task, 'action', return_value=result()) as actions, patch.object(t, 'seed_pose_goal'):
            t.prepare_stage(1)
        orientation = actions.call_args.args[1].request.goal_constraints[0].orientation_constraints[0].orientation
        self.assertEqual([orientation.x, orientation.y, orientation.z, orientation.w], t.c['tcp_orientation_xyzw'])

    def test_pose_goal_ik_keeps_collision_checks_and_rejects_large_posture_changes(self):
        t = task()
        t.c = s.load_config(self.config_path)
        t.action = MagicMock(return_value=result())
        t.motion('Approach', [.3, 0., .1])
        goal = t.action.call_args.args[1]
        t.vertical.scoped_scene.side_effect = lambda *a: nullcontext(RobotState())
        t.vertical.state.return_value = RobotState()
        solved = RobotState()
        solved.joint_state.name = s.ARM_JOINTS
        solved.joint_state.position = [t.c['approach_seed_joints'][n] + .1 for n in s.ARM_JOINTS]
        t.vertical.call.return_value = SimpleNamespace(solution=solved,
            error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS))
        t.seed_pose_goal(goal, False)
        req = t.vertical.call.call_args.args[1]
        self.assertTrue(req.ik_request.avoid_collisions)
        self.assertEqual(len(req.ik_request.constraints.joint_constraints), 7)
        self.assertTrue(all(j.tolerance_above == .75 and j.tolerance_below == .75
                            for j in req.ik_request.constraints.joint_constraints))
        self.assertEqual(len(goal.request.goal_constraints[0].joint_constraints), 7)
        self.assertEqual(len(goal.request.goal_constraints[0].position_constraints), 1)
        solved.joint_state.position[0] += 1.
        with self.assertRaisesRegex(RuntimeError, 'Closest returned solution changes fr3_joint1'):
            t.seed_pose_goal(goal, False)
        unsuitable = copy.deepcopy(t.vertical.call.return_value)
        solved.joint_state.position[0] -= 1.
        acceptable = copy.deepcopy(t.vertical.call.return_value)
        t.vertical.call.side_effect = [unsuitable, acceptable]
        t.seed_pose_goal(goal, False)
        self.assertEqual(goal.request.goal_constraints[0].joint_constraints[0].position,
                         solved.joint_state.position[0])

    def test_grasp_matches_operator_tested_command(self):
        t = task(True)
        t.execute = True
        t.c = s.load_config(self.config_path)
        t.grasp = object()
        t.action = MagicMock(return_value=SimpleNamespace(success=True))
        s.Task.hand(t, close=True)
        client, goal = t.action.call_args.args
        self.assertIs(client, t.grasp)
        self.assertEqual((goal.width, goal.speed, goal.force, goal.epsilon.inner, goal.epsilon.outer),
                         (.015, .01, 10., .002, .002))

    def test_grasp_attachment_uses_measured_tcp_without_shifting_world_block(self):
        t = task(True)
        t.c = s.load_config(self.config_path)
        t.scene_client = object()
        t.settled_snapshot = MagicMock(return_value=dict(t.joints))
        t.vertical.call.return_value = SimpleNamespace(scene=SimpleNamespace(robot_state=RobotState()))
        measured = Pose()
        measured.position.x, measured.position.y = t.c['pick'][:2]
        measured.position.z = t.c['pick'][2] - .001414
        q = t.c['tcp_orientation_xyzw']
        measured.orientation.x, measured.orientation.y, measured.orientation.z, measured.orientation.w = q
        t.vertical.fk.return_value = measured
        before = block_center(t.c)
        s.StepwiseTask.capture_grasp_transform(t)
        attachment = block_object(t.c, carrying=True).primitive_poses[0]
        relative = [attachment.position.x, attachment.position.y, attachment.position.z]
        world = rotate_vector(q, relative)
        xyz = [measured.position.x, measured.position.y, measured.position.z]
        for got, expected in zip([a+b for a, b in zip(xyz, world)], before):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(block_center(t.c), before)
        target = t.placement_target()
        offset = rotate_vector(t.c['carrying_orientation_xyzw'], relative)
        placed_centre = [a+b for a, b in zip(target, offset)]
        destination = block_center(t.c, placed=True)
        for got, expected in zip(placed_centre[:2], destination[:2]):
            self.assertAlmostEqual(got, expected)
        top = t.c['table_center'][2] + t.c['table_size'][2]/2
        self.assertAlmostEqual(placed_centre[2]-t.c['block_size'][2]/2,
                               top+t.c['place_release_clearance_m'])

    def test_virtual_grasp_transform_does_not_read_physical_home_joints(self):
        t = task()
        t.c = s.load_config(self.config_path)
        t.scene_client = object()
        t.settled_snapshot = MagicMock(side_effect=AssertionError('must use virtual joints'))
        t.vertical.call.return_value = SimpleNamespace(scene=SimpleNamespace(robot_state=RobotState()))
        virtual = dict.fromkeys(s.ARM_JOINTS, .1)
        p = Pose()
        p.orientation.w = 1.
        t.vertical.fk.return_value = p
        s.StepwiseTask.capture_grasp_transform(t, virtual)
        t.settled_snapshot.assert_not_called()
        self.assertEqual(t.vertical.state.call_args.args[2], virtual)

    def test_endpoint_acceptance_warns_at_1_4_mm_but_rejects_over_2_mm(self):
        for error, acceptable in ((.001414, True), (.0021, False)):
            t = task(True)
            t.c = s.load_config(self.config_path)
            t.preview_start = RobotState()
            expected, actual = Pose(), Pose()
            expected.orientation.w = actual.orientation.w = 1.
            expected.position.z, actual.position.z = .1, .1-error
            t.vertical.fk.side_effect = [expected, actual]
            if acceptable:
                s.StepwiseTask.check_endpoint(t, trajectory(), t.joints)
                t.node.get_logger.return_value.warning.assert_called_once()
            else:
                with self.assertRaisesRegex(RuntimeError, 'Outside endpoint tolerance'):
                    s.StepwiseTask.check_endpoint(t, trajectory(), t.joints)

    def test_lateral_teaching_allowance_accepts_observed_approach_but_is_bounded(self):
        for error, acceptable in ((.001091, True), (.0021, False)):
            t = task(True)
            t.c = s.load_config(self.config_path)
            t.preview_start = RobotState()
            expected, actual = Pose(), Pose()
            expected.orientation.w = actual.orientation.w = 1.
            actual.position.x = error
            actual.position.z = -.000405
            t.vertical.fk.side_effect = [expected, actual]
            if acceptable:
                s.StepwiseTask.check_endpoint(t, trajectory(), t.joints)
                t.node.get_logger.return_value.warning.assert_called_once()
            else:
                with self.assertRaisesRegex(RuntimeError, 'Outside endpoint tolerance'):
                    s.StepwiseTask.check_endpoint(t, trajectory(), t.joints)


if __name__ == '__main__':
    unittest.main()
