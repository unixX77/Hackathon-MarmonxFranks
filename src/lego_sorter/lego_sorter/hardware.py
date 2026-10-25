"""Executor-safe adaptation of Ameed's Cartesian Mover; no nested ROS spinning."""
import collections
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .core import Cancelled, atomic, inside, layout, matrix
from .vendor import ameed_motion as baseline


def table_collision(c):
    """Build the same validated table box for diagnostics and real manipulation."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive
    centre=np.asarray(c['table_center'],dtype=float)
    size=np.asarray(c['table_size'],dtype=float)
    if centre.shape!=(3,) or size.shape!=(3,) or not np.isfinite([centre,size]).all() or np.any(size<=0):
        raise ValueError('Table requires finite XYZ centre and positive XYZ dimensions')
    if not np.isclose(centre[2]+size[2]/2,c['table_top_z'],rtol=0,atol=1e-6):
        raise ValueError('Table box top does not match table_top_z')
    obj=CollisionObject()
    obj.id='lego_table'; obj.header.frame_id=c['base_frame']
    shape=SolidPrimitive(); shape.type=SolidPrimitive.BOX; shape.dimensions=size.tolist()
    pose=Pose(); pose.position.x,pose.position.y,pose.position.z=centre.tolist(); pose.orientation.w=1.
    obj.primitives=[shape]; obj.primitive_poses=[pose]; obj.operation=CollisionObject.ADD
    return obj


def wait_future(future,timeout,cancel=None):
    done=threading.Event(); future.add_done_callback(lambda _:done.set())
    deadline=time.monotonic()+timeout
    while not done.wait(.02):
        if cancel is not None and cancel.is_set(): raise Cancelled('Operation cancelled')
        if time.monotonic()>deadline: raise TimeoutError('ROS response timeout; reconciliation required')
    return future.result()


class Feedback:
    def __init__(self,node,c):
        from sensor_msgs.msg import JointState
        self.node=node; self.c=c; self.samples=collections.deque(maxlen=500); self.lock=threading.Lock(); self.guard=None; self.guard_moved=False
        self.sub=node.create_subscription(JointState,'joint_states',self.receive,50)
    def receive(self,msg):
        names=[f'fr3_joint{i}' for i in range(1,8)]
        if not all(n in msg.name for n in names): return
        now=self.node.get_clock().now().nanoseconds
        stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        if abs(now-stamp)>self.c['feedback_max_age_s']*1e9: return
        q=np.array([msg.position[msg.name.index(n)] for n in names])
        if not np.isfinite(q).all(): return
        with self.lock:
            self.samples.append((time.monotonic(),q))
            if self.guard is not None and np.max(np.abs(q-self.guard))>self.c['stationary_joint_delta']: self.guard_moved=True
    def fresh(self):
        with self.lock:
            if not self.samples or time.monotonic()-self.samples[-1][0]>self.c['feedback_max_age_s']:
                raise RuntimeError('No fresh stamped FR3 joint feedback')
            return self.samples[-1][1].copy()
    def stationary(self):
        self.fresh(); now=time.monotonic(); window=self.c['stationary_window_s']
        with self.lock: samples=[s for s in self.samples if now-s[0]<=window+.15]
        if len(samples)<3 or now-samples[0][0]<window: raise RuntimeError('Insufficient stationary feedback history')
        if np.max(np.ptp([s[1] for s in samples],axis=0))>self.c['stationary_joint_delta']:
            raise RuntimeError('Robot is moving; snapshot/motion precondition refused')


class SafeMover(baseline.Mover):
    def __init__(self,node,c,tf,feedback):
        super().__init__(node,c,tf,[0,0,c['grasp_tcp_height']-c['block_size'][2]/2])
        self.feedback=feedback; self.cancel=threading.Event(); self.last_yaw=None
    def wait(self,future,timeout): return wait_future(future,timeout,self.cancel)
    def hand_pose(self):
        from rclpy.time import Time
        self.feedback.fresh()
        stamped=self.tf.lookup_transform(self.c['base_frame'],self.c['ee_frame'],Time())
        age=(self.node.get_clock().now().nanoseconds-stamped.header.stamp.sec*10**9-stamped.header.stamp.nanosec)/1e9
        if not 0<=age<=self.c['feedback_max_age_s']: raise RuntimeError('Stale TCP TF')
        t=stamped.transform; return matrix([t.translation.x,t.translation.y,t.translation.z],[t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w])
    def plan_line(self,label,xyz,target_yaw,wrist_limit):
        self.feedback.fresh()
        p=super().plan_line(label,xyz,target_yaw,min(wrist_limit,self.c['max_wrist_swing']))
        if p:
            trajectory,q,swing=p
            violation=baseline.swing_violation(swing,self.c['max_joint_swing'],self.c['max_wrist_swing'])
            if violation: raise RuntimeError(f'Configured joint swing limit: {violation}')
            if np.max(np.abs(q[0]-self.feedback.fresh()))>.03: raise RuntimeError('Trajectory start differs from fresh joints')
            if np.max(np.abs(np.diff(q,axis=0)))>.3: raise RuntimeError('Joint discontinuity')
        self.last_yaw=target_yaw
        return p
    def action(self,client,goal,timeout):
        from action_msgs.msg import GoalStatus
        # Goal acknowledgement must be consumed even when cancellation arrives meanwhile.
        if self.cancel.is_set(): raise Cancelled('Stopped before sending robot command')
        pending=client.send_goal_async(goal)
        try: handle=wait_future(pending,10)
        except (TimeoutError, KeyboardInterrupt):
            pending.add_done_callback(lambda f:f.result().cancel_goal_async() if f.result() and f.result().accepted else None)
            raise
        if not handle.accepted: raise RuntimeError('Robot action rejected')
        result_future=handle.get_result_async()
        try:
            deadline=time.monotonic()+timeout
            while not result_future.done():
                if self.cancel.wait(.02): raise Cancelled('Stopped during robot action')
                self.feedback.fresh()
                if time.monotonic()>deadline: raise TimeoutError('Robot action timeout')
            result=result_future.result()
        except BaseException:
            try:
                wait_future(handle.cancel_goal_async(),5)
                wait_future(result_future,10)
            finally: self.cancel.set()
            raise
        if result.status!=GoalStatus.STATUS_SUCCEEDED: raise RuntimeError(f'Robot action status {result.status}')
        return result.result
    def execute(self,label,trajectory):
        from moveit_msgs.action import ExecuteTrajectory
        if self.action(self.executor,ExecuteTrajectory.Goal(trajectory=trajectory),180).error_code.val!=1:
            raise RuntimeError(label+': MoveIt execution failed')
    def hand(self,close=False):
        from franka_msgs.action import Grasp,Move
        goal=Grasp.Goal() if close else Move.Goal()
        goal.width=float(self.c['grasp_width'] if close else self.c['open_width']); goal.speed=float(self.c['gripper_speed'])
        if close: goal.force=float(self.c['grasp_force']); goal.epsilon.inner=goal.epsilon.outer=.003
        result=self.action(self.grasp_client if close else self.move_client,goal,60)
        if not result.success: raise RuntimeError('Gripper success not confirmed')
    def move(self,label,xyz,yaw=None):
        actual=super().move(label,xyz,yaw)
        deadline=time.monotonic()+3
        while True:
            try: self.feedback.stationary(); break
            except RuntimeError:
                if time.monotonic()>deadline: raise
                if self.cancel.wait(.05): raise Cancelled('Stopped waiting for arrival')
        pose=self.hand_pose()
        if np.linalg.norm(pose[:3,3]-actual)>self.c['arrival_abort']: raise RuntimeError('TCP arrival position failed')
        if yaw is not None:
            # Equivalent pi wrist poses are valid; enforce downward tool and long-axis alignment.
            axis=np.array([math.cos(yaw),math.sin(yaw),0.])
            angle=math.degrees(math.acos(np.clip(abs(axis@pose[:3,0]),-1,1)))
            tilt=math.degrees(math.acos(np.clip(-pose[2,2],-1,1)))
            if max(angle,tilt)>self.c['arrival_angle_deg']: raise RuntimeError('TCP arrival orientation failed')
        return actual
    def apply_scene(self,scene):
        from moveit_msgs.srv import ApplyPlanningScene
        if not self.wait(self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene)),10).success:
            raise RuntimeError('Planning scene update rejected')


class Hardware:
    def __init__(self,node,c,tf):
        from moveit_msgs.srv import GetPlanningScene
        self.c=c; self.node=node; self.tf=tf; self.feedback=Feedback(node,c)
        self.mover=None; self.scene_client=node.create_client(GetPlanningScene,'get_planning_scene')
        self.path=inside(Path(c['runtime'])/'scene.json')
        self.scene=json.loads(self.path.read_text()) if self.path.exists() else dict(removed=[],placed={})
        self.target_source=None; self.target_id=None; self.saved_acm=None
        self.inspection_waiting=False; self.inspection_approval=threading.Event()
    def ready(self):
        if self.mover is None: self.mover=SafeMover(self.node,self.c,self.tf,self.feedback)
    def stationary(self): self.feedback.stationary()
    def begin_capture(self):
        q=self.feedback.fresh()
        with self.feedback.lock: self.feedback.guard=q; self.feedback.guard_moved=False
    def end_capture(self):
        with self.feedback.lock:
            moved=self.feedback.guard_moved; self.feedback.guard=None
        if moved: raise RuntimeError('Robot moved during capture/inference; observation invalid')
    def box(self,ident,centre,dims,yaw=0,frame=None):
        from moveit_msgs.msg import CollisionObject
        from geometry_msgs.msg import Pose
        from shape_msgs.msg import SolidPrimitive
        obj=CollisionObject(); obj.id=ident; obj.header.frame_id=frame or self.c['base_frame']
        shape=SolidPrimitive(); shape.type=SolidPrimitive.BOX; shape.dimensions=list(map(float,dims))
        pose=Pose(); pose.position.x,pose.position.y,pose.position.z=map(float,centre)
        q=Rotation.from_euler('z',yaw).as_quat(); pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,q)
        obj.primitives=[shape]; obj.primitive_poses=[pose]; obj.operation=CollisionObject.ADD; return obj
    def apply(self,objects=(),attached=(),acm=None):
        from moveit_msgs.msg import PlanningScene
        scene=PlanningScene(is_diff=True); scene.robot_state.is_diff=True
        scene.world.collision_objects=list(objects); scene.robot_state.attached_collision_objects=list(attached)
        if acm is not None: scene.allowed_collision_matrix=acm
        self.mover.apply_scene(scene)
    def prepare(self,target=None):
        from moveit_msgs.msg import PlanningSceneComponents,AllowedCollisionEntry
        from moveit_msgs.srv import GetPlanningScene
        source=[s for s in layout(self.c['source'],self.c['block_size'],'source') if s['id'] not in self.scene['removed']]
        objects=[table_collision(self.c)]
        objects += [self.box(s['id'],s['centre'],s['dimensions'],s['yaw']) for s in source]
        objects += [self.box(s['id'],s['centre'],s['dimensions'],s['yaw']) for s in self.scene['placed'].values()]
        objects += [self.box(f['id'],f['centre'],f['dimensions'],f.get('yaw',0)) for f in self.c.get('fixtures',[])]
        self.apply(objects)
        if target is not None:
            nearest=min(source,key=lambda s:np.linalg.norm(np.array(s['centre'])-target['centre']),default=None)
            if nearest is None or np.linalg.norm(np.array(nearest['centre'])-target['centre'])>.004:
                raise RuntimeError('Detection does not match configured structured source geometry')
            self.target_source=nearest; self.target_id=nearest['id']
            req=GetPlanningScene.Request(); req.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            acm=self.mover.wait(self.scene_client.call_async(req),10).scene.allowed_collision_matrix
            import copy
            self.saved_acm=copy.deepcopy(acm)
            names=[self.target_id,'fr3_leftfinger','fr3_rightfinger']
            for name in names:
                if name not in acm.entry_names:
                    for row in acm.entry_values: row.enabled.append(False)
                    acm.entry_names.append(name); acm.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(acm.entry_names)))
            i=acm.entry_names.index(self.target_id)
            for name in names[1:]:
                j=acm.entry_names.index(name); acm.entry_values[i].enabled[j]=acm.entry_values[j].enabled[i]=True
            self.apply(acm=acm)
    def attach(self,target):
        from moveit_msgs.msg import AttachedCollisionObject,CollisionObject
        t=np.linalg.inv(self.mover.hand_pose()) @ matrix(target['centre'],target['quaternion'])
        obj=self.box('lego_held',t[:3,3],target['dimensions'],frame=self.c['ee_frame'])
        q=Rotation.from_matrix(t[:3,:3]).as_quat(); pose=obj.primitive_poses[0]
        pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,q)
        attached=AttachedCollisionObject(link_name=self.c['ee_frame'],object=obj,touch_links=['fr3_leftfinger','fr3_rightfinger','fr3_hand'])
        remove=CollisionObject(id=self.target_id,operation=CollisionObject.REMOVE)
        self.apply([remove],[attached],self.saved_acm)
        self.scene['removed'].append(self.target_id); atomic(self.path,self.scene)
    def detach(self,target):
        from moveit_msgs.msg import AttachedCollisionObject,CollisionObject
        attached=AttachedCollisionObject(link_name=self.c['ee_frame'],object=CollisionObject(id='lego_held',operation=CollisionObject.REMOVE))
        # Keep released brick as an obstacle after retreat; remove only the detached transport ID.
        self.apply([CollisionObject(id='lego_held',operation=CollisionObject.REMOVE)],[attached])
        self.scene['placed'][target['id']]=target; atomic(self.path,self.scene)
    def approve_inspection(self):
        if not self.inspection_waiting: raise RuntimeError('Robot is not waiting at the inspection pose')
        self.inspection_approval.set()
    def step(self,stage,plan,target,cancel):
        if cancel.is_set(): raise Cancelled('Stopped before motion stage')
        self.ready(); m=self.mover; m.cancel=cancel
        if stage=='OPEN': self.prepare(target); m.hand(False)
        elif stage=='INSPECT':
            m.move(stage,plan['inspect'],plan['yaw'])
            self.inspection_approval.clear(); self.inspection_waiting=True
            self.node.get_logger().info('At inspection pose; task_control approve_inspection required before descent')
            deadline=time.monotonic()+self.c['inspection_timeout_s']
            try:
                while not self.inspection_approval.wait(.05):
                    if cancel.is_set(): raise Cancelled('Stopped during inspection')
                    self.feedback.stationary()
                    if time.monotonic()>deadline: raise TimeoutError('Inspection approval timed out')
            finally: self.inspection_waiting=False
        elif stage=='GRASP': m.hand(True); self.attach(target)
        elif stage=='RELEASE': m.hand(False); self.detach(target)
        elif stage=='TRANSFER_LIFT':
            now=m.hand_pose(); height=max(now[2,3],plan['above'][2],self.c['transfer_z'])
            if height>self.c['max_hover_z']: raise ValueError('Transfer too high')
            # Correction must not be added twice to an already measured TCP point.
            yaw=baseline.hand_yaw(now[:3,:3]); ox,oy=baseline.hand_frame_to_base(self.c['tcp_correction_hand'],yaw)
            m.move(stage,[now[0,3]-ox,now[1,3]-oy,height],yaw)
        else:
            key={'APPROACH':'above','LOWER':'grip','LIFT':'above','TRANSFER':'above','PLACE_LOWER':'release','RETREAT':'above','OBSERVE_MOVE':'above'}[stage]
            xyz=list(plan[key])
            if stage=='TRANSFER': xyz[2]=max(xyz[2],self.c['transfer_z'])
            if stage=='OBSERVE_MOVE': self.prepare()
            m.move(stage,xyz,plan['yaw'])
            if stage=='RETREAT': self.prepare()
