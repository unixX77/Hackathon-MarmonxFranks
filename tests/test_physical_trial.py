"""Physical-trial preflight and stage behavior with no hardware connections."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import sys
import numpy as np
import yaml

from lego_sorter.core import ROOT,load_config,Store,Cancelled
from lego_sorter.physical_trial import boundary_from_corners,build_trial_config,validate_trial,PhysicalStageGate,correct_trial_observation
from lego_sorter.regions import source_pixel_mask


class PhysicalTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'runtime',prefix='test-physical-preflight-')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.base=yaml.safe_load((ROOT/'src/lego_sorter/config/hardware.yaml').read_text())
        points={'source_corner_a':[.3,-.04,.2],'source_corner_b':[.48,.04,.2],
                'place':[.4,-.25,.2],'discard':[.7,.25,.2],
                'observation':[.4,0.,.4],'grasp':[.4,0.,-.008]}
        self.teaching=dict(destinations={name:dict(base_frame='fr3_link0',tcp_frame='fr3_hand_tcp',
            xy_m=xyz[:2],captured_tcp_xyz_m=xyz,captured_tcp_quaternion_xyzw=[1.,0.,0.,0.],
            observation_pose_xyz_yaw=[*xyz,0.]) for name,xyz in points.items()})

    def profile(self):
        return build_trial_config(self.base,self.teaching,self.path,3,-20.)

    def test_profile_is_physical_but_preserves_uncommissioned_flags_and_explicit_scope(self):
        original=copy.deepcopy(self.base);c=self.profile();validate_trial(c)
        self.assertEqual(self.base,original)
        self.assertEqual(c['mode'],'hardware');self.assertFalse(c['hardware_validated'])
        self.assertFalse(c['geometry_confirmed']);self.assertEqual(c['source_mode'],'detected')
        self.assertEqual(c['pick_correction_tool_m'],[-.02,0.])
        self.assertEqual(c['tcp_correction_hand'],[0.,0.])
        self.assertAlmostEqual(c['grasp_tcp_height'],.012)
        self.assertFalse(c['destination_clearance_check'])

    def test_scoped_profile_does_not_enable_normal_hardware_startup(self):
        c=self.profile();p=self.path/'config.yaml';p.write_text(yaml.safe_dump(c))
        with self.assertRaisesRegex(ValueError,'not been commissioned'):
            load_config(p)
        self.assertEqual(load_config(p,supervised_trial=True)['mode'],'hardware')

    def test_existing_grasp_geometry_needs_no_new_teaching(self):
        del self.teaching['destinations']['grasp']
        c=self.profile();validate_trial(c)
        self.assertAlmostEqual(c['grasp_tcp_height'],.012)
        self.assertEqual(c['grasp_height_source'],'hardware.yaml:grasp_tcp_height')
        self.assertNotIn('grasp',c['teaching_evidence'])
        with self.assertRaisesRegex(ValueError,'1 through 30'):
            build_trial_config(self.base,self.teaching,self.path,31)

    def test_optional_grasp_teaching_overrides_config_but_keeps_limits(self):
        self.teaching['destinations']['grasp']['captured_tcp_xyz_m'][2]=-.007
        c=self.profile();validate_trial(c)
        self.assertAlmostEqual(c['grasp_tcp_height'],.013)
        self.assertEqual(c['grasp_height_source'],'teaching:grasp')
        self.teaching['destinations']['grasp']['captured_tcp_xyz_m'][2]=.09
        with self.assertRaisesRegex(ValueError,'table-clearance'):self.profile()

    def test_invalid_configured_grasp_height_is_rejected_without_teaching(self):
        del self.teaching['destinations']['grasp']
        for height in (.009,.041,float('nan')):
            self.base['grasp_tcp_height']=height
            with self.assertRaisesRegex(ValueError,'table-clearance'):self.profile()

    def test_boundary_is_axis_aligned_and_independent_of_corner_order(self):
        self.assertEqual(boundary_from_corners([.48,.04],[.3,-.04]),self.profile()['source_boundary'])
        with self.assertRaises(ValueError):boundary_from_corners([.3,0.],[.301,.2])

    def test_explicit_pick_correction_is_applied_once_before_neighbour_selection(self):
        c=self.profile();original=dict(bricks=[dict(centre=[.4,0.,-.0152],top=[.4,0.,-.0104],yaw=0.)])
        corrected=correct_trial_observation(original,c,0.)
        self.assertAlmostEqual(corrected['bricks'][0]['centre'][0],.38)
        self.assertEqual(original['bricks'][0]['centre'][0],.4)
        self.assertEqual(corrected['bricks'][0]['centre'][2],-.0152)
        with self.assertRaisesRegex(ValueError,'already applied'):
            correct_trial_observation(corrected,c,0.)

    def test_destination_source_region_overlap_is_not_allowed(self):
        c=self.profile();c['destinations']['selected']['origin']=[.4,0.,-.02]
        with self.assertRaisesRegex(ValueError,'source search region'):validate_trial(c)

    def test_no_destination_clearance_allocation_against_assumed_grid(self):
        c=self.profile();c['mode']='mock'
        c['destinations']['selected'].update(rows=1,columns=1,origin=[*c['source']['origin']])
        store=Store(c);self.addCleanup(store.close)
        slot=store.reserve(dict(colour='red',confidence='high'))
        self.assertEqual(slot['state'],'RESERVED')

    def test_pixel_roi_uses_base_coordinates_and_excludes_invalid_depth(self):
        info=dict(aligned_depth_intrinsics=dict(fx=1.,fy=1.,ppx=1.,ppy=1.,distortion_coeffs=[]),depth_scale_m=.001)
        depth=np.full((3,3),1000,np.uint16)
        mask=source_pixel_mask(depth,info,np.eye(4),dict(x=[-.1,.1],y=[-.1,.1]))
        self.assertEqual(int(mask.sum()),1);self.assertTrue(mask[1,1])
        depth[1,1]=0
        self.assertFalse(source_pixel_mask(depth,info,np.eye(4),dict(x=[-.1,.1],y=[-.1,.1])).any())

    def test_outside_masks_are_removed_before_top_layer_selection(self):
        import importlib
        sys.path.insert(0,str(ROOT/'src/lego_sorter/lego_sorter/vendor'))
        try:
            module=importlib.import_module('top_layer_bricks')
            depth_module=importlib.import_module('top_layer_depth')
            args=module.build_parser().parse_args([str(self.path),'--no-show','--brick-size-mm','32','16','9.6','--top-min-support','1'])
            args.block_height_mm=9.6
            depth=np.full((60,80),1000,np.uint16)
            masks=np.zeros((2,60,80),bool)
            masks[0,25:35,15:35]=True;masks[1,25:35,55:75]=True
            depth[masks[0]]=990;depth[masks[1]]=970
            info=dict(aligned_depth_intrinsics=dict(width=80,height=60,fx=600.,fy=600.,ppx=40.,ppy=30.,
                distortion_coeffs=[],K=[[600.,0.,40.],[0.,600.,30.],[0.,0.,1.]]),depth_scale_m=.001)
            args.source_roi_mask=source_pixel_mask(depth,info,np.eye(4),dict(x=[-.08,0.],y=[-.05,.05]))
            accepted,rejected=module.geometry_mask_filter(depth,info,args)(masks,np.array([.9,.95]))
            self.assertEqual(accepted.tolist(),[0])
            self.assertIn('outside source boundary',rejected[0]['reason'])
            instance=np.zeros(depth.shape,int);instance[masks[0]]=1
            _,report=depth_module.filter_top(instance,depth,info,args)
            self.assertAlmostEqual(report['top_height_m'],.01,places=3)
        finally:sys.path.pop(0)

    def test_physical_gate_never_approves_on_quit_or_changed_joints(self):
        feedback=SimpleNamespace(stationary=Mock(),fresh=Mock(return_value=np.zeros(7)))
        gate=PhysicalStageGate(feedback,Mock(),reader=lambda _: 'q')
        gate.stop=Mock();gate.log=Mock()
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(Cancelled):gate('LOWER','EMPTY')
        gate.stop.assert_called_once();gate.log.assert_not_called()
        feedback.fresh.side_effect=[np.zeros(7),np.full(7,.01)]
        gate.reader=lambda _:'';gate.stop.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(Cancelled,'Arm moved'):
            gate('LOWER','EMPTY')
        gate.log.assert_not_called()

    def test_detected_source_has_no_grid_boxes_or_four_mm_matching(self):
        try:
            from lego_sorter.hardware import Hardware
            from moveit_msgs.msg import CollisionObject
        except ImportError:self.skipTest('Source ROS for collision-object checks')
        backend=object.__new__(Hardware);backend.c=self.profile()
        backend.scene=dict(removed=[],placed={});backend.mover=Mock();backend.scene_client=Mock()
        backend.mover.wait.return_value=SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(
            collision_objects=[CollisionObject(id='source/0/0/0'),CollisionObject(id='other_fixture')])) )
        backend.apply=Mock();backend.box=Mock()
        target=dict(id='observation-brick',centre=[.45,.02,-.0152])
        backend.prepare(target)
        backend.box.assert_not_called();self.assertIsNone(backend.target_id)
        self.assertIs(backend.target_source,target)
        objects=backend.apply.call_args.args[0]
        self.assertEqual({o.id for o in objects},{'lego_table','source/0/0/0'})
        self.assertEqual(next(o for o in objects if o.id.startswith('source/')).operation,CollisionObject.REMOVE)

    def test_runner_stops_at_requested_limit_with_only_test_doubles(self):
        try:
            import rclpy
            import tf2_ros
        except ImportError:self.skipTest('Source ROS for runner import')
        sys.path.insert(0,str(ROOT/'scripts'))
        try:from run_one_layer_hardware import run
        finally:sys.path.pop(0)
        from lego_sorter.workflow import MockWorld
        c=self.profile();c['pick_correction_tool_m']=[0.,0.]
        fixture=copy.deepcopy(c);fixture['source'].update(rows=1,columns=5,yaw_deg=0)
        world=MockWorld(fixture)
        feedback=SimpleNamespace(stationary=Mock(),fresh=Mock(return_value=np.zeros(7)))
        pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[.4,0.,.3]
        backend=SimpleNamespace(ready=Mock(),stationary=Mock(),feedback=feedback,
            mover=SimpleNamespace(health=Mock(),hand_pose=Mock(return_value=pose)),step=world.step)
        worker=SimpleNamespace(start=Mock(),close=Mock(),observe=lambda epoch,lookup:world.observe(epoch))
        path=self.path/'config.yaml';path.write_text(yaml.safe_dump(c))
        with patch('rclpy.init'),patch('rclpy.create_node',return_value=Mock()), \
             patch('rclpy.shutdown'),patch('rclpy.ok',return_value=False), \
             patch('rclpy.executors.MultiThreadedExecutor'),patch('tf2_ros.Buffer'), \
             patch('tf2_ros.TransformListener'),patch('lego_sorter.hardware.Hardware',return_value=backend), \
             patch('lego_sorter.perception.SamProcess',return_value=worker), \
             patch('builtins.input',return_value=''),contextlib.redirect_stdout(io.StringIO()):
            code=run(c,path)
        self.assertEqual(code,0)
        mission=json.loads((self.path/'mission.json').read_text())
        self.assertEqual((mission['status'],mission['completed']),('TRIAL_COMPLETE',3))
        self.assertEqual(len(world.read()['bricks']),2)
        self.assertIsNone(world.read()['held'])

    def test_fk_speed_stretch_and_full_observation_quaternion_with_fake_services(self):
        try:
            from lego_sorter.hardware import SafeMover
            from moveit_msgs.msg import RobotTrajectory
            from geometry_msgs.msg import PoseStamped
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError:self.skipTest('Source ROS for message construction')
        mover=object.__new__(SafeMover);mover.c=self.profile();mover.node=Mock()
        mover.wait=Mock();mover.node.create_client.return_value.wait_for_service.return_value=True
        trajectory=RobotTrajectory();trajectory.joint_trajectory.joint_names=[f'fr3_joint{i}' for i in range(1,8)]
        for t,x in [(0,0.),(1,.02)]:
            p=JointTrajectoryPoint(positions=[x]*7,velocities=[.01]*7)
            p.time_from_start.sec=t;trajectory.joint_trajectory.points.append(p)
        poses=[]
        for x in [0.,.02]:
            p=PoseStamped();p.pose.position.x=x
            poses.append(SimpleNamespace(error_code=SimpleNamespace(val=1),pose_stamped=[p]))
        mover.wait.side_effect=poses
        mover.limit_tcp_speed(trajectory)
        final=trajectory.joint_trajectory.points[-1]
        self.assertEqual(final.time_from_start.sec,2)
        self.assertAlmostEqual(final.velocities[0],.005)

        from lego_sorter.core import matrix
        mover.scale=.02;mover.feedback=SimpleNamespace(fresh=lambda:np.zeros(7))
        mover.path_client=Mock();mover.wait=Mock(return_value=SimpleNamespace(
            error_code=SimpleNamespace(val=1),fraction=1.,solution=trajectory))
        mover.execute=Mock();mover.settle=Mock()
        q=[.8,.6,0.,0.];xyz=[.4,0.,.4]
        mover.hand_pose=Mock(return_value=matrix(xyz,q))
        mover.move_pose('OBSERVE_MOVE',xyz,q)
        request=mover.path_client.call_async.call_args.args[0]
        self.assertTrue(request.avoid_collisions)
        self.assertAlmostEqual(request.waypoints[0].orientation.x,.8)
        self.assertAlmostEqual(request.waypoints[0].orientation.y,.6)
        mover.execute.assert_called_once()
