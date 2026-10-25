import copy
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from lego_sorter.core import (ROOT,load_config,matrix,rigid,calibration,adapt_detection,layout,bounds,
    supported,select_brick,Store,Manipulator,Cancelled,atomic)
from lego_sorter.workflow import MockWorld,LocalTransport,Coordinator
from lego_sorter.perception import empty_depth


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'runtime',prefix='test-')
        self.c=load_config(ROOT/'src/lego_sorter/config/sorter.yaml'); self.c['runtime']=self.temp.name
    def tearDown(self): self.temp.cleanup()
    def machine(self):
        w=MockWorld(self.c); m=Manipulator(self.c,w); m.reconcile_empty('Mock physical inspection'); return w,m,LocalTransport(m,w)
    def observed(self):
        w,m,t=self.machine(); t.observation_pose(); o=t.observe(); b=select_brick(o['bricks'],self.c); return w,m,t,o,b
    def test_full_sixty_brick_two_layer_loop(self):
        w,m,t=self.machine(); s=Store(self.c); e=Coordinator(self.c,s,t); e.control('start')
        for _ in range(65):
            if not e.tick(): break
        self.assertEqual(e.data['status'],'COMPLETE'); self.assertEqual(e.data['completed'],60)
        self.assertEqual(len(w.read()['bricks']),0); self.assertEqual(m.state['hand'],'EMPTY')
        self.assertEqual(sum(x['state']=='PLACED' for x in s.slots),60)
        self.assertEqual(sum(x['state']=='PLACED' and x['destination']=='discard' for x in s.slots),24)
        self.assertFalse(any(x['state'] in ('RESERVED','UNCERTAIN') for x in s.slots))
        self.assertTrue(any(x['state']=='PLACED' and x['layer']>1 for x in s.slots))
    def test_geometry_30_per_layer_gap_and_rotation(self):
        s=layout(self.c['source'],self.c['block_size'],'s'); self.assertEqual(len(s),60)
        self.assertAlmostEqual(s[5]['centre'][1]-s[0]['centre'][1],.032)
        self.assertAlmostEqual(s[1]['centre'][0]-s[0]['centre'][0],.032)
        self.assertAlmostEqual(s[30]['yaw']-s[0]['yaw'],math.pi/2)
        a=np.array(s[0]['centre'][:2])-self.c['source']['origin'][:2]
        b=np.array(s[30]['centre'][:2])-self.c['source']['origin'][:2]
        np.testing.assert_allclose(b,[-a[1],a[0]],atol=1e-12)
    def test_explicit_layer_origin_override(self):
        spec=copy.deepcopy(self.c['source']); spec['layer_overrides']={'1':{'origin':[.4,.1,-.02],'yaw_deg':0}}
        s=layout(spec,self.c['block_size'],'s'); self.assertAlmostEqual(s[30]['centre'][0]-s[0]['centre'][0],.03)
    def test_support_requires_placed_underneath(self):
        s=layout(self.c['destinations']['red'],self.c['block_size'],'red')
        self.assertFalse(supported(s[1],s,.5)); s[0]['state']='PLACED'; self.assertTrue(supported(s[1],s,.5))
        self.assertFalse(supported(s[1],s,.51))
    def test_clearance_blocks_neighbor_across_jaws(self):
        _,_,_,_,b=self.observed(); b=copy.deepcopy(b); other=copy.deepcopy(b); other['id']='other'; other['centre'][1]+=.017
        self.assertIsNone(select_brick([b,other],self.c))
    def test_capture_transform_not_current_hand_and_no_double_height(self):
        self.c['camera_height_bias']=.002
        bp=matrix([.35,0,.4],[0,0,0,1]); pc=matrix([0,0,0],[1,0,0,0])
        metadata={'brick_size_mm':[32,16,9.6],'bricks':[dict(id=1,status='brick',centre_cam_m=[0,0,.4],box_centre_cam_m=[0,0,.4048],box_rotation_cam=np.diag([1,-1,-1]).tolist(),box_size_m=[.032,.016,.0096],colour='red',quality='good',colour_confidence='high')]}
        b=adapt_detection(metadata,'obs',bp,pc,self.c)[0]
        self.assertAlmostEqual(b['top'][2],-.002); self.assertAlmostEqual(b['centre'][2],-.0068)
        self.assertEqual(b['frame'],self.c['base_frame']); self.assertAlmostEqual(b['yaw'],0)
        metadata['brick_size_mm']=[30,15,10]
        with self.assertRaises(ValueError): adapt_detection(metadata,'obs',bp,pc,self.c)
    def test_calibration_parent_mismatch_no_identity_fallback(self):
        p=Path(self.temp.name)/'calibration.yaml'; self.c['calibration_file']=str(p)
        p.write_text(yaml.safe_dump(dict(parent_frame='fr3_link8',child_frame=self.c['camera_frame'],status='passed_quality_gates',T_ee_camera=np.eye(4).tolist())))
        with self.assertRaises(ValueError): calibration(self.c)
        with self.assertRaises(ValueError): rigid(np.diag([1,1,-1,1]))
    def test_modular_schema_rejected(self):
        with self.assertRaises(ValueError): adapt_detection({'detections':[]},'x',np.eye(4),np.eye(4),self.c)
    def test_atomic_reservation_restart_uncertain(self):
        _,_,_,_,b=self.observed(); s=Store(self.c); slot=s.reserve(b)
        with self.assertRaises(RuntimeError): s.reserve(b)
        with self.assertRaises(ValueError): s.transition(slot['id'],'PLACED','')
        s.fd.close(); restarted=Store(self.c)
        self.assertEqual(next(x for x in restarted.slots if x['id']==slot['id'])['state'],'UNCERTAIN')
    def test_single_colour_and_unknown_discard(self):
        self.c['policy']['mode']='single'; _,_,_,_,b=self.observed(); s=Store(self.c)
        b.update(colour='red',confidence='high'); slot=s.reserve(b); self.assertEqual(slot['destination'],'selected'); s.transition(slot['id'],'EMPTY')
        b['confidence']='low'; self.assertEqual(s.reserve(b)['destination'],'discard')
    def test_alternating_rule_advances_only_on_commit(self):
        self.c['policy']['mode']='alternating'; _,_,_,_,b=self.observed(); s=Store(self.c)
        b.update(colour='red',confidence='high'); a=s.reserve(b); self.assertEqual(a['destination'],'diverse')
        s.transition(a['id'],'PLACED','mock verified'); b['colour']='red'; a=s.reserve(b); self.assertEqual(a['destination'],'discard')
        s.transition(a['id'],'EMPTY'); b['colour']='dark_red'; self.assertEqual(s.reserve(b)['destination'],'diverse')
    def test_capacity_fail_closed(self):
        self.c['destinations']['red']['layers']=1; _,_,_,_,b=self.observed(); s=Store(self.c); b.update(colour='red',confidence='high')
        a=s.reserve(b); s.transition(a['id'],'PLACED','mock')
        with self.assertRaises(RuntimeError): s.reserve(b)
    def test_stale_after_camera_move_and_ttl(self):
        _,m,t,o,b=self.observed(); t.observation_pose()
        self.assertFalse(m.run('pick',b,o['epoch'])['success'])
        o=t.observe(); b=select_brick(o['bricks'],self.c); m.observation['stamp_ns']=1
        self.assertFalse(m.run('pick',b,o['epoch'])['success'])
    def test_observation_lease_excludes_motion(self):
        _,m,_=self.machine(); token,epoch=m.capture_begin()
        self.assertEqual(m.run('observe_pose')['outcome'],'BUSY'); m.capture_end(token,'obs',time.time_ns())
    def test_pick_holds_then_place_empty(self):
        _,m,t,o,b=self.observed(); result=t.pick(b,o); self.assertTrue(result['success']); self.assertEqual(m.state['hand'],'HOLDING')
        self.assertFalse(t.pick(b,o)['success']); s=Store(self.c); slot=s.reserve(b)
        self.assertTrue(t.place(slot,result['held_id'])['success']); self.assertEqual(m.state['hand'],'EMPTY')
        self.assertFalse(t.place(slot,result['held_id'])['success'])
    def test_failure_before_grasp_no_open_after_failure(self):
        self.c['mock_fail_stage']='LOWER'; w,m,t,o,b=self.observed(); before=w.read()['steps']; r=t.pick(b,o)
        self.assertEqual(r['outcome'],'FAILURE_BEFORE_GRASP'); self.assertEqual(m.state['hand'],'EMPTY'); self.assertTrue(m.cancel.is_set())
        self.assertEqual(w.read()['steps']-before,2)
    def test_grasp_uncertainty_never_opens_or_restarts(self):
        self.c['mock_fail_stage']='GRASP'; w,m,t,o,b=self.observed(); r=t.pick(b,o)
        self.assertEqual(r['hand_state'],'GRASP_UNCERTAIN'); steps=w.read()['steps']
        self.assertFalse(t.pick(b,o)['success']); self.assertEqual(w.read()['steps'],steps)
        with self.assertRaises(RuntimeError): t.resume()
    def test_lift_failure_preserves_held_context(self):
        self.c['mock_fail_stage']='LIFT'; _,m,t,o,b=self.observed(); r=t.pick(b,o)
        self.assertEqual(r['hand_state'],'HOLDING'); self.assertTrue(r['held_id'])
    def test_release_uncertainty_reserves_destination(self):
        self.c['mock_fail_stage']='RELEASE'; w,m,t=self.machine(); s=Store(self.c); e=Coordinator(self.c,s,t); e.control('start'); e.tick()
        self.assertEqual(e.data['status'],'FAILED'); self.assertEqual(m.state['hand'],'RELEASE_UNCERTAIN')
        self.assertEqual(sum(x['state']=='UNCERTAIN' for x in s.slots),1)
    def test_cancel_between_grasp_and_lift(self):
        w,m,t,o,b=self.observed()
        def feedback(stage,hand):
            if stage=='LIFT': m.stop()
        r=m.run('pick',b,o['epoch'],feedback)
        self.assertEqual(r['outcome'],'OPERATOR_STOP'); self.assertEqual(m.state['hand'],'HOLDING'); self.assertIsNotNone(w.read()['held'])
    def test_pause_resume_observes_afresh(self):
        w,m,t=self.machine(); s=Store(self.c); e=Coordinator(self.c,s,t); e.control('start'); e.control('pause'); e.tick()
        self.assertEqual(e.data['status'],'PAUSED'); self.assertEqual(e.data['completed'],0)
        e.control('resume'); e.tick(); self.assertEqual(e.data['completed'],1)
    def test_empty_detections_inconclusive(self):
        _,m,t=self.machine(); t.observe=lambda:dict(id='x',epoch=0,bricks=[],empty_verified=False,evidence='occluded')
        s=Store(self.c); e=Coordinator(self.c,s,t); e.control('start'); e.tick(); self.assertEqual(e.data['status'],'FAILED')
    def test_placement_requires_verification(self):
        w,m,t=self.machine(); t.verify=lambda slot:(False,'occluded')
        s=Store(self.c); e=Coordinator(self.c,s,t); e.control('start'); e.tick()
        self.assertEqual(m.state['hand'],'EMPTY'); self.assertTrue(any(x['state']=='UNCERTAIN' for x in s.slots))
        with self.assertRaises(RuntimeError): e.control('resume')
    def test_crash_active_hand_requires_reconciliation(self):
        w,m,t=self.machine(); m.state.update(active=True,hand='EMPTY'); m.save(); m.close(); restarted=Manipulator(self.c,w)
        self.assertEqual(restarted.state['hand'],'UNRECONCILED'); self.assertFalse(restarted.run('observe_pose')['success'])
    def test_process_owner_exclusion(self):
        w,m,t=self.machine()
        with self.assertRaises(BlockingIOError): Manipulator(self.c,w)
    def test_stop_from_paused_has_explicit_stopped_status(self):
        w,m,t=self.machine(); s=Store(self.c); e=Coordinator(self.c,s,t)
        e.control('start'); e.control('pause'); e.tick(); e.control('stop')
        self.assertEqual(e.data['status'],'STOPPED'); self.assertTrue(m.cancel.is_set())
        e.control('resume'); self.assertFalse(m.cancel.is_set())
    def test_unknown_goal_dimensions_refused_before_open(self):
        w,m,t,o,b=self.observed(); b['dimensions']=[.03,.015,.01]; steps=w.read()['steps']
        self.assertFalse(t.pick(b,o)['success']); self.assertEqual(w.read()['steps'],steps)
    def test_empty_depth_with_full_coverage_and_occlusion(self):
        info={'aligned_depth_intrinsics':{'K':[[400,0,320],[0,400,240],[0,0,1]]},'depth_scale_m':.001}
        t=matrix([.37,0,.48],[1,0,0,0]); depth=np.full((480,640),500,dtype=np.uint16)
        self.assertTrue(empty_depth(depth,info,t,self.c)[0]); depth[200:250,300:350]=480
        self.assertFalse(empty_depth(depth,info,t,self.c)[0]); depth[:]=0; self.assertFalse(empty_depth(depth,info,t,self.c)[0])


if __name__=='__main__': unittest.main()
