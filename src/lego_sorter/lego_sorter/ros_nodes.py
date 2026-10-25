"""Three persistent ROS nodes. Worker threads never spin an executor."""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer,ActionClient,GoalResponse,CancelResponse
from rclpy.task import Future
from rclpy.time import Time
from tf2_ros import Buffer,TransformListener
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped,TransformStamped
from lego_sort_interfaces.msg import Brick,Observation
from lego_sort_interfaces.srv import ObservePile,MotionControl,TaskControl,VerifyPlacement
from lego_sort_interfaces.action import PickBrick,PlaceBrick
from .core import load_config, matrix, Store, Manipulator, atomic, inside
from .hardware import wait_future,Hardware
from .workflow import MockWorld,ReplayWorld,Coordinator
from .perception import SamProcess


def pose(centre,quat,frame):
    p=PoseStamped(); p.header.frame_id=frame
    p.pose.position.x,p.pose.position.y,p.pose.position.z=map(float,centre)
    p.pose.orientation.x,p.pose.orientation.y,p.pose.orientation.z,p.pose.orientation.w=map(float,quat)
    return p


def brick_msg(b):
    msg=Brick(); msg.observation_id=b['observation_id']; msg.local_id=b['id']
    q=b.get('quaternion',Rotation.from_euler('z',b['yaw']).as_quat())
    msg.top_face=pose(b['top'],q,b['frame']); msg.object_pose=pose(b['centre'],q,b['frame'])
    msg.dimensions.x,msg.dimensions.y,msg.dimensions.z=map(float,b['dimensions'])
    msg.colour=b['colour']; msg.quality=b['quality']; msg.colour_confidence=b['confidence']
    msg.colour_distance=float(b.get('colour_distance',0)); msg.colour_margin=float(b.get('colour_margin',0)); return msg


def xyz(p): return [p.x,p.y,p.z]
def quat(q): return [q.x,q.y,q.z,q.w]
def brick_dict(msg):
    q=quat(msg.object_pose.pose.orientation); r=Rotation.from_quat(q).as_matrix()
    return dict(id=msg.local_id,observation_id=msg.observation_id,frame=msg.object_pose.header.frame_id,
        top=xyz(msg.top_face.pose.position),centre=xyz(msg.object_pose.pose.position),quaternion=q,
        yaw=math.atan2(r[1,0],r[0,0]),dimensions=xyz(msg.dimensions),colour=msg.colour,quality=msg.quality,
        confidence=msg.colour_confidence,colour_distance=msg.colour_distance,colour_margin=msg.colour_margin)


def transform(t,parent,child,stamp):
    msg=TransformStamped(); msg.header.frame_id=parent; msg.child_frame_id=child; msg.header.stamp=stamp
    msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z=map(float,t[:3,3])
    q=Rotation.from_matrix(t[:3,:3]).as_quat()
    msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z,msg.transform.rotation.w=map(float,q)
    return msg


class Base(Node):
    def __init__(self,name):
        super().__init__(name)
        # Baseline Mover creates clients without callback_group; make their default reentrant.
        self._default_callback_group=ReentrantCallbackGroup()
        self.declare_parameter('config','/home/tanay/Documents/Hackathon/project-planner/src/lego_sorter/config/sorter.yaml')
        self.config_path=self.get_parameter('config').value; self.c=load_config(self.config_path)
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=3)
    async def worker(self,fn,*args):
        future=Future()
        def run():
            try: future.set_result(fn(*args))
            except Exception as e: future.set_exception(e)
        self.pool.submit(run)
        return await future
    def client_call(self,client,request,timeout=20):
        if not client.wait_for_service(timeout_sec=5): raise RuntimeError('Service unavailable: '+client.srv_name)
        return wait_future(client.call_async(request),timeout)
    def destroy_node(self):
        self.pool.shutdown(wait=False,cancel_futures=True); super().destroy_node()


class ManipulationNode(Base):
    def __init__(self):
        super().__init__('lego_manipulation'); self.tf=Buffer(); self.listener=TransformListener(self.tf,self)
        self.backend=Hardware(self,self.c,self.tf) if self.c['mode']=='hardware' else MockWorld(self.c)
        self.machine=Manipulator(self.c,self.backend)
        if self.c['mode']!='hardware' and not self.backend.read()['held']:
            self.machine.reconcile_empty('SIMULATED world confirms empty hand')
        self.admission=threading.Lock(); self.admitted=False
        self.pick_server=ActionServer(self,PickBrick,'pick_brick',self.pick_execute,goal_callback=self.pick_goal,cancel_callback=self.cancel_goal)
        self.place_server=ActionServer(self,PlaceBrick,'place_brick',self.place_execute,goal_callback=self.place_goal,cancel_callback=self.cancel_goal)
        self.control=self.create_service(MotionControl,'motion_control',self.control_execute)
        self.publisher=self.create_publisher(String,'manipulation_status',10)
        self.timer=self.create_timer(.2,self.publish)
    def publish(self): self.publisher.publish(String(data=json.dumps(dict(self.machine.state,inspection_waiting=getattr(self.backend,'inspection_waiting',False)))))
    def admit(self,hand):
        with self.admission:
            if self.admitted or self.machine.lock.locked() or self.machine.state['hand']!=hand or self.machine.cancel.is_set(): return GoalResponse.REJECT
            self.admitted=True; return GoalResponse.ACCEPT
    def pick_goal(self,request):
        o=self.machine.observation
        if not o or request.brick.observation_id!=o['id'] or request.motion_epoch!=o['epoch']: return GoalResponse.REJECT
        return self.admit('EMPTY')
    def place_goal(self,request):
        held=self.machine.state.get('held') or {}
        if request.engage_studs or request.held_id!=held.get('held_id'): return GoalResponse.REJECT
        return self.admit('HOLDING')
    def cancel_goal(self,handle): self.machine.stop(); return CancelResponse.ACCEPT
    def action_run(self,handle,kind,target,epoch,action):
        try:
            def feedback(stage,hand): handle.publish_feedback(action.Feedback(stage=stage,hand_state=hand))
            result=self.machine.run(kind,target,epoch,feedback)
            if handle.is_cancel_requested: handle.canceled()
            elif result['success']: handle.succeed()
            else: handle.abort()
            return action.Result(**result)
        finally:
            with self.admission: self.admitted=False
    async def pick_execute(self,handle):
        try: b=brick_dict(handle.request.brick)
        except Exception as e:
            with self.admission: self.admitted=False
            handle.abort(); return PickBrick.Result(success=False,outcome='INVALID_GOAL',detail=str(e),hand_state=self.machine.state['hand'])
        return await self.worker(self.action_run,handle,'pick',b,handle.request.motion_epoch,PickBrick)
    async def place_execute(self,handle):
        req=handle.request
        try:
            if req.object_pose.header.frame_id!=self.c['base_frame']: raise ValueError('Place must be in base frame')
            q=quat(req.object_pose.pose.orientation); r=matrix(xyz(req.object_pose.pose.position),q)[:3,:3]
            if r[2,2]<.999: raise ValueError('Placement must be level')
            slots=json.loads((Path(self.c['runtime'])/'slots.json').read_text())['slots']
            reserved=next(s for s in slots if s['id']==req.slot_id and s['state']=='RESERVED')
            if not np.allclose(reserved['centre'],xyz(req.object_pose.pose.position),atol=1e-6) or abs(math.sin(reserved['yaw']-math.atan2(r[1,0],r[0,0])))>1e-6:
                raise ValueError('Place target does not match reserved slot')
            b=dict(id=req.slot_id,held_id=req.held_id,centre=xyz(req.object_pose.pose.position),yaw=math.atan2(r[1,0],r[0,0]),dimensions=xyz(req.dimensions),engage_studs=req.engage_studs)
        except Exception as e:
            with self.admission: self.admitted=False
            handle.abort(); return PlaceBrick.Result(success=False,outcome='INVALID_GOAL',detail=str(e),hand_state=self.machine.state['hand'])
        return await self.worker(self.action_run,handle,'place',b,None,PlaceBrick)
    async def control_execute(self,req,res):
        def run():
            m=self.machine
            if req.command=='status': pass
            elif req.command=='stop': m.stop()
            elif req.command=='approve_inspection':
                if not hasattr(self.backend,'approve_inspection'): raise RuntimeError('No hardware inspection pending')
                self.backend.approve_inspection()
            elif req.command=='reconcile_empty': m.reconcile_empty(req.token)
            elif req.command=='resume':
                m.resume()
            elif req.command=='observe_pose':
                r=m.run('observe_pose')
                if not r['success']: raise RuntimeError(r['detail'])
            elif req.command=='capture_begin':
                res.token,res.motion_epoch=m.capture_begin()
            elif req.command=='capture_end':
                stamp=req.capture_stamp.sec*10**9+req.capture_stamp.nanosec
                m.capture_end(req.token,req.observation_id,stamp)
            else: raise ValueError('Unknown motion command')
            res.success=True; res.hand_state=m.state['hand']; res.motion_epoch=m.state['epoch']; return res
        try: return await self.worker(run)
        except Exception as e: res.success=False; res.detail=str(e); res.hand_state=self.machine.state['hand']; return res


class PerceptionNode(Base):
    def __init__(self):
        super().__init__('lego_perception'); self.tf=Buffer(); self.listener=TransformListener(self.tf,self)
        self.motion=self.create_client(MotionControl,'motion_control'); self.busy=threading.Lock()
        self.backend=SamProcess(self.c,self.config_path) if self.c['mode']=='hardware' else ReplayWorld(self.c) if self.c['mode']=='replay' else MockWorld(self.c)
        self.service=self.create_service(ObservePile,'observe_pile',self.execute)
        self.verify_service=self.create_service(VerifyPlacement,'verify_placement',self.verify)
        # Model loading happens before snapshot requests, on its own worker; never inside a ROS callback.
        self.initialized=self.pool.submit(self.backend.start) if self.c['mode']=='hardware' else None
    def lookup(self,stamp_ns):
        from rclpy.duration import Duration
        # Capture time, not latest TF. Hardware requires system ROS time.
        if self.get_parameter('use_sim_time').value: raise ValueError('Hardware capture requires system time')
        t=self.tf.lookup_transform(self.c['base_frame'],self.c['calibration_parent'],Time(nanoseconds=stamp_ns),timeout=Duration(seconds=2)).transform
        return matrix(xyz(t.translation),quat(t.rotation))
    async def execute(self,req,res):
        if not self.busy.acquire(False): res.detail='Snapshot already in progress'; return res
        def run():
            lease=None; observation=None
            try:
                if self.initialized is not None:
                    if not self.initialized.done(): raise RuntimeError('SAM model loading; retry later')
                    self.initialized.result()
                lease=self.client_call(self.motion,MotionControl.Request(command='capture_begin'))
                if not lease.success: raise RuntimeError(lease.detail)
                observation=self.backend.observe(lease.motion_epoch,self.lookup) if self.c['mode']=='hardware' else self.backend.observe(lease.motion_epoch)
                msg=Observation(); msg.id=observation['id']; msg.capture_stamp.sec=observation['stamp_ns']//10**9; msg.capture_stamp.nanosec=observation['stamp_ns']%10**9
                msg.camera_frame=self.c['camera_frame']; msg.motion_epoch=observation['epoch']; msg.bricks=[brick_msg(b) for b in observation['bricks']]
                for b in msg.bricks: b.top_face.header.stamp=msg.capture_stamp; b.object_pose.header.stamp=msg.capture_stamp
                msg.source_empty_verified=observation['empty_verified']; msg.evidence=observation['evidence']; msg.diagnostics_path=observation['diagnostics']
                if observation.get('base_parent') is not None:
                    msg.base_from_parent=transform(observation['base_parent'],self.c['base_frame'],self.c['calibration_parent'],msg.capture_stamp)
                    msg.parent_from_camera=transform(observation['parent_camera'],self.c['calibration_parent'],self.c['camera_frame'],msg.capture_stamp)
                res.observation=msg; res.success=True
            finally:
                if lease and lease.success:
                    end=MotionControl.Request(command='capture_end',token=lease.token,observation_id=observation['id'] if observation else '')
                    if observation:
                        end.capture_stamp.sec=observation['stamp_ns']//10**9; end.capture_stamp.nanosec=observation['stamp_ns']%10**9
                    done=self.client_call(self.motion,end)
                    if not done.success: raise RuntimeError('Capture stationarity/lease failure: '+done.detail)
            return res
        try: return await self.worker(run)
        except Exception as e: res.success=False; res.detail=str(e); return res
        finally: self.busy.release()
    def receipt_path(self,transaction): return inside(Path(self.c['runtime'])/'receipts'/f'{hashlib.sha256(transaction.encode()).hexdigest()}.json')
    async def verify(self,req,res):
        if self.c['mode']!='hardware':
            w=self.backend.read(); res.verified=req.slot_id in w['placed']; res.evidence='SIMULATED world occupancy'; return res
        if self.c['verification']!='operator': res.evidence='Hardware verification must be explicitly configured as operator'; return res
        p=self.receipt_path(req.transaction_id)
        if p.exists():
            receipt=json.loads(p.read_text())
            res.verified=receipt['slot_id']==req.slot_id and receipt['transaction_id']==req.transaction_id and bool(receipt['evidence'])
            res.evidence=receipt['evidence']
        else: res.evidence='Awaiting operator placement verification'
        return res
    def destroy_node(self):
        if isinstance(self.backend,SamProcess): self.backend.close()
        super().destroy_node()


class RosTransport:
    def __init__(self,node):
        self.node=node; self.motion=node.create_client(MotionControl,'motion_control'); self.obs=node.create_client(ObservePile,'observe_pile')
        self.verifier=node.create_client(VerifyPlacement,'verify_placement')
        self.picker=ActionClient(node,PickBrick,'pick_brick'); self.placer=ActionClient(node,PlaceBrick,'place_brick')
        self.active=None; self.stopped=threading.Event(); self.stop_future=None
    def control(self,command,token=''):
        r=self.node.client_call(self.motion,MotionControl.Request(command=command,token=token),240)
        if not r.success: raise RuntimeError(r.detail)
        return r
    def observation_pose(self): self.control('observe_pose'); return dict(success=True)
    def observe(self):
        r=self.node.client_call(self.obs,ObservePile.Request(purpose='source'),self.node.c['snapshot_timeout_s']+30)
        if not r.success: raise RuntimeError(r.detail)
        o=r.observation
        return dict(id=o.id,epoch=o.motion_epoch,stamp_ns=o.capture_stamp.sec*10**9+o.capture_stamp.nanosec,
                    bricks=[brick_dict(b) for b in o.bricks],empty_verified=o.source_empty_verified,evidence=o.evidence)
    def action(self,client,goal):
        if self.stopped.is_set(): raise RuntimeError('Stopped')
        if not client.wait_for_server(timeout_sec=5): raise RuntimeError('Manipulation action unavailable')
        pending=client.send_goal_async(goal)
        try: handle=wait_future(pending,10)
        except TimeoutError:
            pending.add_done_callback(lambda f:f.result().cancel_goal_async() if f.result() and f.result().accepted else None)
            self.stop(); raise
        if not handle.accepted: raise RuntimeError('Manipulation goal rejected')
        self.active=handle
        if self.stopped.is_set(): handle.cancel_goal_async()
        try:
            result=wait_future(handle.get_result_async(),600).result
            return {k:getattr(result,k) for k in ['success','outcome','detail','hand_state','held_id']}
        except Exception:
            handle.cancel_goal_async(); self.stop(); raise
        finally: self.active=None
    def pick(self,b,o): return self.action(self.picker,PickBrick.Goal(brick=brick_msg(b),motion_epoch=o['epoch']))
    def place(self,s,held):
        g=PlaceBrick.Goal(); g.held_id=held; g.slot_id=s['id']; g.object_pose=pose(s['centre'],Rotation.from_euler('z',s['yaw']).as_quat(),self.node.c['base_frame'])
        g.dimensions.x,g.dimensions.y,g.dimensions.z=map(float,s['dimensions']); g.engage_studs=False
        return self.action(self.placer,g)
    def verify(self,s):
        deadline=time.monotonic()+self.node.c['verification_timeout_s']
        self.node.engine.log.event('verification_required',slot=s['id'],transaction=s['transaction'])
        while True:
            r=self.node.client_call(self.verifier,VerifyPlacement.Request(slot_id=s['id'],transaction_id=s['transaction']))
            if r.verified: return True,r.evidence
            if self.stopped.wait(.2) or time.monotonic()>deadline: return False,r.evidence
    def hand(self):
        try: return self.control('status').hand_state
        except Exception: return 'UNRECONCILED'
    def resume(self):
        if self.stop_future is not None: wait_future(self.stop_future,20)
        self.control('resume'); self.stopped.clear()
    def stop(self):
        self.stopped.set()
        if self.active: self.active.cancel_goal_async()
        self.stop_future=self.motion.call_async(MotionControl.Request(command='stop'))


class CoordinatorNode(Base):
    def __init__(self):
        super().__init__('lego_coordinator'); self.store=Store(self.c); self.transport=RosTransport(self); self.engine=Coordinator(self.c,self.store,self.transport)
        self.control=self.create_service(TaskControl,'task_control',self.execute_control)
        self.pub=self.create_publisher(String,'sorter_status',10); self.timer=self.create_timer(.2,self.publish)
        self.shutdown=threading.Event(); self.loop=threading.Thread(target=self.run,daemon=True); self.loop.start()
    def publish(self): self.pub.publish(String(data=json.dumps(self.engine.data)))
    def run(self):
        while not self.shutdown.wait(.05):
            if self.engine.data['status']=='RUNNING': self.engine.tick()
    async def execute_control(self,req,res):
        def run():
            words=req.command.split()
            if not words: raise ValueError('Missing command')
            if words[0]=='approve_inspection': self.transport.control('approve_inspection')
            elif words[0]=='reconcile_empty': self.transport.control('reconcile_empty',req.evidence)
            elif words[0]=='verify' and len(words)==3:
                if not req.evidence: raise ValueError('Physical placement evidence required')
                s=next(s for s in self.store.slots if s['id']==words[1] and s.get('transaction')==words[2] and s['state']=='RESERVED')
                path=inside(Path(self.c['runtime'])/'receipts'/f'{hashlib.sha256(words[2].encode()).hexdigest()}.json')
                atomic(path,dict(slot_id=s['id'],transaction_id=words[2],evidence=req.evidence,time_ns=time.time_ns()))
            elif words[0]=='resolve_slot' and len(words)==3:
                if self.engine.data['status']=='RUNNING' or self.transport.hand()!='EMPTY' or not req.evidence: raise RuntimeError('Stop and reconcile hand, then supply physical slot evidence')
                s=next(s for s in self.store.slots if s['id']==words[1])
                if s['state']!='UNCERTAIN' or words[2] not in ('EMPTY','PLACED'): raise ValueError('Only uncertain slots may be reconciled')
                with self.store.lock:
                    s.update(state=words[2],evidence=req.evidence); self.store.save()
                self.engine.log.event('slot_reconciled',slot=s['id'],state=words[2],evidence=req.evidence)
            else: self.engine.control(req.command)
            res.success=True; res.detail=self.engine.data['status']; return res
        try: return await self.worker(run)
        except Exception as e: res.success=False; res.detail=str(e); return res
    def destroy_node(self): self.shutdown.set(); self.transport.stop(); super().destroy_node()


def spin(cls):
    rclpy.init(); node=cls(); executor=MultiThreadedExecutor(num_threads=6); executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        if isinstance(node,ManipulationNode): node.machine.stop()
        node.destroy_node(); executor.shutdown(); rclpy.try_shutdown()


def perception_main(): spin(PerceptionNode)
def manipulation_main(): spin(ManipulationNode)
def coordinator_main(): spin(CoordinatorNode)
