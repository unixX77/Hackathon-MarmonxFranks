"""Offline checks; never connect to a robot."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cartesian_jog as j
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, PlanningScene, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


def pose():
    p = Pose()
    p.position.z = .3
    p.orientation.x = 1.
    p.orientation.w = 0.
    return p


def task(execute):
    t = object.__new__(j.JogTask)
    t.hardware = execute
    t.execute = execute
    t.endpoint_z_tolerance_m = .002
    t.node = MagicMock()
    t.joints = dict.fromkeys(j.ARM, 0.)
    t.measured_pose = MagicMock(return_value=pose())
    t.vertical = MagicMock()
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = j.ARM
    trajectory.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.]*7)]
    t.vertical.plan.return_value = (trajectory, RobotState())
    t.preview = MagicMock()
    t.executor = object()
    t.settled_snapshot = MagicMock(return_value=dict(t.joints))
    t.action = MagicMock()
    return t


class JogChecks(unittest.TestCase):
    def test_hardware_detection_checks_arm_component(self):
        def urdf(plugin):
            return ('<robot><ros2_control><hardware><plugin>' + plugin
                    + '</plugin></hardware><joint name="fr3_joint1"/></ros2_control></robot>')
        self.assertTrue(j.is_mock_arm(urdf('mock_components/GenericSystem')))
        self.assertFalse(j.is_mock_arm(urdf('franka_hardware/FrankaHardwareInterface')))
        for invalid in (urdf('unknown/Driver'), '<robot/>'):
            with self.assertRaises(RuntimeError):
                j.is_mock_arm(invalid)

    def test_fake_execution_is_not_preview_only(self):
        t = task(True)
        t.hardware = False
        t.vertical.revalidate.side_effect = RuntimeError('execution path reached')
        with patch.object(j, 'confirm') as confirm:
            with self.assertRaisesRegex(RuntimeError, 'execution path reached'):
                t.jog(-.005)
        self.assertTrue(confirm.call_args.args[1])

    def test_scene_geometry_requirement_is_only_for_real_execution(self):
        planner = object.__new__(j.ScenePlanner)
        planner.task = SimpleNamespace(hardware=False, scene_client=object(),
                                       table_object_id='table', require_camera=MagicMock())
        planner.call = MagicMock(return_value=SimpleNamespace(scene=PlanningScene()))
        with planner.scoped_scene():
            pass
        planner.task.require_camera.assert_not_called()
        planner.task.hardware = True
        with self.assertRaisesRegex(RuntimeError, 'table collision object'):
            with planner.scoped_scene():
                pass
        scene = PlanningScene()
        scene.world.collision_objects = [j.box('table', [2., 2., .04], [0., 0., -.04])]
        planner.call.return_value = SimpleNamespace(scene=scene)
        with planner.scoped_scene():
            pass
        planner.task.require_camera.assert_not_called()

    def test_missing_table_is_added_without_camera_or_board(self):
        t = task(False)
        t.table_object_id = 'fr3_demo_table'
        t.scene_client = MagicMock()
        t.wait = MagicMock(side_effect=[SimpleNamespace(scene=PlanningScene()),
                                       SimpleNamespace(success=True)])
        config = Path(__file__).resolve().parents[1] / 'config/cartesian_jog.yaml'
        t.ensure_table(config)
        diff = t.node.create_client.return_value.call_async.call_args.args[0].scene
        self.assertTrue(diff.is_diff)
        self.assertTrue(diff.robot_state.is_diff)
        self.assertEqual(len(diff.world.collision_objects), 1)
        table = diff.world.collision_objects[0]
        self.assertEqual(table.id, t.table_object_id)
        self.assertEqual(table.header.frame_id, j.BASE)
        self.assertAlmostEqual(table.primitive_poses[0].position.z
                               + table.primitives[0].dimensions[2] / 2, -.020)
        self.assertEqual(list(diff.robot_state.attached_collision_objects), [])
        t.action.assert_not_called()

    def test_existing_table_is_not_replaced(self):
        t = task(False)
        t.table_object_id = 'table'
        t.scene_client = MagicMock()
        scene = PlanningScene()
        scene.world.collision_objects = [j.box('table', [1., 1., .1], [0., 0., -.1])]
        t.wait = MagicMock(return_value=SimpleNamespace(scene=scene))
        t.ensure_table(Path('/unused/config.yaml'))
        t.node.create_client.assert_not_called()

    def test_invalid_table_or_failed_apply_aborts(self):
        t = task(False)
        t.table_object_id = 'table'
        t.scene_client = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'jog.yaml'
            config.write_text('table_collision_center: [0, 0, -0.04]\ntable_collision_size: [2, 2, -1]\n')
            t.wait = MagicMock(return_value=SimpleNamespace(scene=PlanningScene()))
            with self.assertRaisesRegex(ValueError, 'dimensions must be positive'):
                t.ensure_table(config)
            t.node.create_client.assert_not_called()
        t.wait = MagicMock(side_effect=[SimpleNamespace(scene=PlanningScene()),
                                       SimpleNamespace(success=False)])
        config = Path(__file__).resolve().parents[1] / 'config/cartesian_jog.yaml'
        with self.assertRaisesRegex(RuntimeError, 'Could not add configured table'):
            t.ensure_table(config)
        t.action.assert_not_called()

    def test_step_limits(self):
        for value in ('nan', 'inf', '0', '-0.0001', '.021', '-1'):
            with self.assertRaises(ValueError):
                j.parse_step(value)
        self.assertEqual(j.parse_step('-.005'), -.005)

    def test_saved_pose_has_orientation_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poses.json'
            joints = dict.fromkeys(j.ARM, .1)
            j.record_pose(path, 'pick', pose(), joints)
            data = json.loads(path.read_text())
            self.assertEqual(data['poses']['pick']['quaternion_xyzw'], [1., 0., 0., 0.])
            self.assertEqual(data['poses']['pick']['xyz_m'], [0., 0., .3])
            with self.assertRaises(ValueError):
                j.record_pose(path, 'pick', pose(), joints)
            j.record_pose(path, 'place', pose(), joints)
            self.assertEqual(set(json.loads(path.read_text())['poses']), {'pick', 'place'})

    def test_state_keeps_all_attachments_without_mutating_original(self):
        planner = object.__new__(j.ScenePlanner)
        state = RobotState()
        for name in ('handeye_camera', 'fr3_demo_block'):
            attachment = AttachedCollisionObject(link_name=j.TCP)
            attachment.object.id = name
            state.attached_collision_objects.append(attachment)
        before = copy.deepcopy(state)
        merged = planner.state(state, False, dict.fromkeys(j.ARM, .1))
        self.assertEqual(merged.attached_collision_objects, state.attached_collision_objects)
        self.assertEqual(state, before)

    def test_preview_never_executes_and_only_changes_target_z(self):
        t = task(False)
        with patch.object(j, 'confirm'):
            t.jog(-.005)
        self.assertEqual(t.vertical.plan.call_args.args[0], [0., 0., .295])
        t.action.assert_not_called()

    def test_failed_scene_revalidation_prevents_execution(self):
        t = task(True)
        t.vertical.revalidate.side_effect = RuntimeError('changed scene')
        with patch.object(j, 'confirm'):
            with self.assertRaisesRegex(RuntimeError, 'changed scene'):
                t.jog(-.005)
        t.action.assert_not_called()

    def test_changed_start_prevents_execution(self):
        t = task(True)
        t.settled_snapshot.return_value['fr3_joint1'] = .1
        with patch.object(j, 'confirm'):
            with self.assertRaisesRegex(RuntimeError, 'no longer matches'):
                t.jog(-.005)
        t.action.assert_not_called()

    def test_executes_exact_preview_after_guards(self):
        t = task(True)
        t.vertical.limits.line_tolerance_m = .001
        t.vertical.limits.orientation_tolerance_deg = .5
        end = pose()
        end.position.z = .295
        t.measured_pose.side_effect = [pose(), end]
        t.action.return_value = SimpleNamespace(error_code=SimpleNamespace(val=j.MoveItErrorCodes.SUCCESS))
        with patch.object(j, 'confirm'):
            t.jog(-.005)
        t.vertical.revalidate.assert_called_once()
        t.vertical.check_execution_start.assert_called_once()
        self.assertEqual(t.action.call_args.args[1].trajectory, t.vertical.plan.return_value[0])

    def test_successful_action_with_wrong_endpoint_reports_measured_errors(self):
        t = task(True)
        t.vertical.limits.line_tolerance_m = .001
        t.vertical.limits.orientation_tolerance_deg = .5
        t.action.return_value = SimpleNamespace(error_code=SimpleNamespace(val=j.MoveItErrorCodes.SUCCESS))
        # Controller reports success but the measured arm did not move.
        with patch.object(j, 'confirm'):
            with self.assertRaisesRegex(RuntimeError, r'Z error=\+5\.000 mm'):
                t.jog(-.005)
        t.action.assert_called_once()
        report = t.node.get_logger.return_value.info.call_args.args[0]
        self.assertIn('measured dXYZ=[+0.000, +0.000, +0.000] mm', report)
        self.assertIn('max final joint error=', report)

    def test_teaching_accepts_observed_z_offset_with_warning(self):
        t = task(True)
        t.vertical.limits.line_tolerance_m = .001
        t.vertical.limits.orientation_tolerance_deg = .5
        end = pose()
        end.position.x = -.000701
        end.position.y = -.000009
        end.position.z -= .006414
        t.measured_pose.side_effect = [pose(), end]
        t.action.return_value = SimpleNamespace(error_code=SimpleNamespace(val=j.MoveItErrorCodes.SUCCESS))
        with patch.object(j, 'confirm'):
            t.jog(-.005)
        t.node.get_logger.return_value.warning.assert_called_once()
        self.assertEqual(t.vertical.limits.line_tolerance_m, .001)

    def test_teaching_z_allowance_does_not_relax_lateral_check(self):
        t = task(True)
        t.vertical.limits.line_tolerance_m = .001
        t.vertical.limits.orientation_tolerance_deg = .5
        end = pose()
        end.position.x = .0015
        end.position.z = .295
        t.measured_pose.side_effect = [pose(), end]
        t.action.return_value = SimpleNamespace(error_code=SimpleNamespace(val=j.MoveItErrorCodes.SUCCESS))
        with patch.object(j, 'confirm'):
            with self.assertRaisesRegex(RuntimeError, 'outside tolerance'):
                t.jog(-.005)


if __name__ == '__main__':
    unittest.main()
