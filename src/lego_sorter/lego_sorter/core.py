"""ROS-independent geometry, transactions and manipulation state machine (metres/radians)."""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from .vendor.ameed_plans import plan_pick, plan_place

ROOT = Path('/home/tanay/Documents/Hackathon/project-planner')


def inside(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Output must be inside {ROOT}: {path}')
    return path


def load_config(path,supervised_trial=False):
    c = yaml.safe_load(Path(path).read_text())
    inside(c['runtime']).mkdir(parents=True, exist_ok=True)
    if c['mode'] not in ('mock', 'hardware', 'replay'):
        raise ValueError('mode must be mock, replay or hardware')
    if c.get('engage_studs'):
        raise ValueError('Stud engagement unsupported: no validated seating controller')
    if len(c['block_size']) != 3 or min(c['block_size']) <= 0:
        raise ValueError('Invalid dimensions')
    if not 0 < c['velocity_scale'] <= .1 or not 0 <= c['place_margin'] <= .005:
        raise ValueError('Invalid speed or release clearance')
    if c['open_width'] < c['block_size'][1] + c['min_open_clearance']:
        raise ValueError('Insufficient jaw opening')
    if c['policy']['mode'] not in ('single', 'separate', 'alternating'):
        raise ValueError('Unknown policy')
    if c['mode'] == 'hardware':
        if c['source'].get('stud_connected') or any(s.get('stud_connected') for s in c['destinations'].values()):
            raise ValueError('Stud-connected lattice dismantling/assembly requires validated extraction and seating controllers; currently unsupported')
        if any(s.get('kind')=='drop_box' for s in c['destinations'].values()):
            raise ValueError('Trash-box drops are currently mock-only; hardware needs measured box geometry and validated release/fill handling')
        if supervised_trial:
            from .physical_trial import validate_trial
            validate_trial(c)
        if not c.get('hardware_validated') and not supervised_trial:
            raise ValueError('Hardware configuration has not been commissioned; inspect README')
        calibration(c)
        if c['verification']!='operator': raise ValueError('Hardware requires explicit operator placement verification')
        if not c.get('geometry_confirmed') and not supervised_trial: raise ValueError('Confirm physical stack, tray, fingers and workspace geometry')
        if not supervised_trial and (not c.get('fixtures') or not any(f['id']=='discard_tray' for f in c['fixtures'])):
            raise ValueError('Hardware requires measured discard_tray collision geometry')
    for spec in [c['source'],*c['destinations'].values()]:
        if spec.get('kind')=='drop_box':
            if not isinstance(spec['capacity'],int) or isinstance(spec['capacity'],bool) or spec['capacity']<=0:
                raise ValueError('Drop-box capacity must be a positive integer')
            if not np.isfinite(spec['drop_centre_height']) or spec['drop_centre_height']<=0:
                raise ValueError('Drop-box release height must be positive')
        if any(not isinstance(spec[k],int) or isinstance(spec[k],bool) or spec[k]<=0 for k in ['rows','columns','layers']): raise ValueError('Layout counts must be positive integers')
        if min(spec['brick_gap'],spec['row_gap_studs'])<0 or spec['stud_pitch']<=0: raise ValueError('Invalid layout spacing')
        if not np.isfinite(spec['origin']).all(): raise ValueError('Invalid layout origin')
    if not 0<c['minimum_support_fraction']<=1: raise ValueError('Invalid support fraction')
    return c


def matrix(xyz, quat):
    if not np.isfinite([*xyz, *quat]).all() or abs(np.linalg.norm(quat)-1) > .01:
        raise ValueError('Nonfinite pose or nonunit quaternion')
    t = np.eye(4)
    t[:3,:3] = Rotation.from_quat(quat).as_matrix()
    t[:3,3] = xyz
    return t


def rigid(t):
    t = np.asarray(t, dtype=float)
    if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1]):
        raise ValueError('Invalid transform')
    if not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-3) or np.linalg.det(t[:3,:3]) < .999:
        raise ValueError('Transform must be a proper rotation')
    return t


def calibration(c):
    data = yaml.safe_load(Path(c['calibration_file']).read_text())
    if data['parent_frame'] != c['calibration_parent'] or data['child_frame'] != c['camera_frame']:
        raise ValueError('Calibration parent/child mismatch; flange and TCP are not interchangeable')
    if data.get('status') != 'passed_quality_gates' and not c.get('allow_rejected_calibration', False):
        raise ValueError('Calibration quality gates not passed')
    return rigid(data[c['calibration_matrix_key']])


def adapt_detection(metadata, observation_id, base_parent, parent_camera, c):
    """Accept ONLY top_layer_bricks schema. Transform once, then apply height correction once."""
    if 'brick_size_mm' not in metadata or 'bricks' not in metadata:
        raise ValueError('Not a top_layer_bricks detector record')
    if not np.allclose(np.array(metadata['brick_size_mm'])/1000, c['block_size'], atol=1e-6):
        raise ValueError('Detector/manipulation dimensions disagree')
    t = rigid(base_parent) @ rigid(parent_camera)
    out = []
    for b in metadata['bricks']:
        if not b.get('id') or b['status'] not in ('brick','split'):
            continue
        dims = np.asarray(b['box_size_m'])
        if not np.allclose(dims, c['block_size'], atol=1e-5):
            raise ValueError('Unexpected fitted brick dimensions')
        r = np.asarray(b['box_rotation_cam'], dtype=float)
        if r.shape != (3,3) or not np.allclose(r.T@r, np.eye(3), atol=.002) or np.linalg.det(r) < .99:
            raise ValueError('Invalid object rotation')
        top = (t @ np.r_[b['centre_cam_m'],1])[:3]
        centre = (t @ np.r_[b['box_centre_cam_m'],1])[:3]
        if c.get('source_mode')=='detected':
            region=c['source_boundary']
            if not (region['x'][0]<=centre[0]<=region['x'][1] and region['y'][0]<=centre[1]<=region['y'][1]):
                continue
        rb = t[:3,:3] @ r
        if not np.allclose(top-centre, rb[:,2]*dims[2]/2, atol=.0003):
            raise ValueError('Top-face and volume-centre conventions inconsistent')
        if rb[2,2] < math.cos(math.radians(c['max_object_tilt_deg'])):
            raise ValueError('Object is tilted or calibration orientation is wrong')
        top[2] -= c['camera_height_bias']
        if c['snap_heights']:
            n = max(1, round((top[2]-c['source']['origin'][2])/dims[2]))
            expected = c['source']['origin'][2]+n*dims[2]
            if abs(top[2]-expected) <= c['layer_snap_tolerance']:
                top[2] = expected
        if top[2]-c['source']['origin'][2] > c['max_brick_top_above_table'] or top[2]<c['source']['origin'][2]+dims[2]-.008:
            raise ValueError('Measured source height outside configured stack envelope')
        centre = top-rb[:,2]*dims[2]/2
        yaw = math.atan2(rb[1,0],rb[0,0])
        plan_pick(c,*centre,yaw=yaw)
        out.append(dict(id=str(b['id']), observation_id=observation_id, frame=c['base_frame'],
            centre=centre.tolist(), top=top.tolist(), quaternion=Rotation.from_matrix(rb).as_quat().tolist(),
            yaw=yaw, dimensions=dims.tolist(), colour=b.get('colour','unknown'), quality=b.get('quality','unknown'),
            confidence=b.get('colour_confidence','low'), colour_distance=b.get('colour_distance',0.),
            colour_margin=b.get('colour_margin',0.)))
    return out


def layout(spec, dims, name):
    """Origin is layer footprint CENTRE in XY and bottom in Z. Rotate entire row lattice."""
    if spec.get('kind')=='drop_box':
        # Simulation receipts share a release point, not physical packing positions.
        return [dict(id=f'{name}/drop/{i}',destination=name,layer=0,row=0,column=i,
            centre=[*spec['origin'][:2],spec['origin'][2]+spec['drop_centre_height']],
            yaw=math.radians(spec.get('yaw_deg',0)),dimensions=list(dims),
            kind='drop_box',state='EMPTY') for i in range(spec['capacity'])]
    L,W,H = dims
    slots=[]
    for layer in range(spec['layers']):
        override = spec.get('layer_overrides', {}).get(str(layer), {})
        origin = override.get('origin', spec['origin'])
        yaw = math.radians(override.get('yaw_deg',spec.get('yaw_deg',0)+90*(layer%2)))
        if abs(yaw/(math.pi/2)-round(yaw/(math.pi/2))) > 1e-6:
            raise ValueError('Layout currently requires orthogonal layer orientations')
        for row in range(spec['rows']):
            for col in range(spec['columns']):
                x=(col-(spec['columns']-1)/2)*(L+spec['brick_gap'])
                y=(row-(spec['rows']-1)/2)*(W+spec['row_gap_studs']*spec['stud_pitch'])
                centre=[origin[0]+x*math.cos(yaw)-y*math.sin(yaw),
                        origin[1]+x*math.sin(yaw)+y*math.cos(yaw),origin[2]+(layer+.5)*H]
                slots.append(dict(id=f'{name}/{layer}/{row}/{col}', destination=name, layer=layer,
                    row=row,column=col,centre=centre,yaw=yaw,dimensions=list(dims),state='EMPTY'))
    return slots


def bounds(item, inflate=0):
    l,w,_=item['dimensions']; a=item['yaw']; x,y,_=item['centre']
    dx=(abs(math.cos(a))*l+abs(math.sin(a))*w)/2+inflate
    dy=(abs(math.sin(a))*l+abs(math.cos(a))*w)/2+inflate
    return x-dx,y-dy,x+dx,y+dy


def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def supported(slot, slots, fraction):
    if slot['layer']==0: return True
    b=bounds(slot); area=slot['dimensions'][0]*slot['dimensions'][1]
    under=[s for s in slots if s['destination']==slot['destination'] and s['layer']==slot['layer']-1
           and s['state']=='PLACED' and abs((s['centre'][2]+s['dimensions'][2])-slot['centre'][2]) < .0001]
    # Same-layer cells do not overlap. Support area is therefore an exact sum.
    contacts=[bounds(s) for s in under if overlap(b,bounds(s))>0]
    if sum(overlap(b,a) for a in contacts)+1e-9 < fraction*area: return False
    x,y,_=slot['centre']
    # Exact boundary contacts must not fail because of floating-point roundoff.
    eps=1e-9
    return bool(contacts) and min(a[0] for a in contacts)-eps<=x<=max(a[2] for a in contacts)+eps and min(a[1] for a in contacts)-eps<=y<=max(a[3] for a in contacts)+eps


def clear(item, others, c):
    # Jaw fingers extend across the short axis; their longitudinal length is explicit.
    jaw=dict(item,dimensions=[c['finger_length'],c['open_width']+2*c['finger_thickness'],item['dimensions'][2]])
    for other in others:
        if other['id']==item['id']: continue
        if other['centre'][2]+other['dimensions'][2]/2 < item['centre'][2]-item['dimensions'][2]/2+.0001: continue
        if overlap(bounds(jaw),bounds(other))>1e-10: return False
    return True


def select_brick(bricks,c):
    if not bricks: return None
    top=max(b['top'][2] for b in bricks)
    accepted=[b for b in bricks if top-b['top'][2]<=c['top_layer_tolerance'] and clear(b,bricks,c)]
    # Unknown colours remain eligible, but go to controlled discard placement.
    return min(accepted,key=lambda b:(b['confidence']!='high',-b['top'][2],b['id']),default=None)


def atomic(path,data):
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


class Journal:
    def __init__(self,path): self.path=inside(path); self.lock=threading.RLock()
    def event(self,event,**kw):
        with self.lock, self.path.open('a') as f:
            f.write(json.dumps(dict(time_ns=time.time_ns(),event=event,**kw),allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())


class Store:
    def __init__(self,c):
        self.c=c; self.path=inside(Path(c['runtime'])/'slots.json'); self.lock=threading.RLock()
        self.fd=open(self.path.with_suffix('.lock'),'a')
        fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fingerprint=hashlib.sha256(json.dumps({k:c[k] for k in ['source','destinations','block_size','policy']},sort_keys=True).encode()).hexdigest()
        if self.path.exists():
            self.data=json.loads(self.path.read_text())
            if self.data['fingerprint']!=fingerprint: raise ValueError('Layout/policy changed: reconcile state before migration')
            for s in self.slots:
                if s['state']=='RESERVED': s['state']='UNCERTAIN'
        else:
            self.data=dict(fingerprint=fingerprint,slots=[s for name,spec in c['destinations'].items() for s in layout(spec,c['block_size'],name)])
        self.save()
    def close(self):
        if hasattr(self,'fd') and not self.fd.closed: self.fd.close()
    def __del__(self): self.close()
    @property
    def slots(self): return self.data['slots']
    def save(self): atomic(self.path,self.data)
    def reserve(self,brick):
        with self.lock:
            if any(s['state'] in ('RESERVED','UNCERTAIN') for s in self.slots):
                raise RuntimeError('Outstanding destination transaction needs reconciliation')
            p=self.c['policy']; colour=brick['colour'] if brick['confidence']=='high' else 'unknown'
            dest='discard'
            if p['mode']=='single' and colour==p['colour']: dest='selected'
            elif p['mode']=='separate' and colour in p['colours']: dest=colour
            elif p['mode']=='alternating':
                count=sum(s['state']=='PLACED' and s['destination']=='diverse' for s in self.slots)
                if colour==p['sequence'][count%len(p['sequence'])]: dest='diverse'
            check_clearance=self.c.get('destination_clearance_check',True)
            if not check_clearance and any(d['layers']!=1 for d in self.c['destinations'].values()):
                raise ValueError('Disabling destination clearance is restricted to single-layer table placement')
            obstacles=([s for s in self.slots if s['state']!='EMPTY' and s.get('kind')!='drop_box']
                       + ([] if self.c.get('source_mode')=='detected' else layout(self.c['source'],self.c['block_size'],'source'))) if check_clearance else []
            for s in self.slots:
                if s['destination']!=dest or s['state']!='EMPTY': continue
                b=bounds(s,self.c['finger_thickness'])
                if not (self.c['workspace_x'][0]<=b[0]<=b[2]<=self.c['workspace_x'][1] and self.c['workspace_y'][0]<=b[1]<=b[3]<=self.c['workspace_y'][1]): continue
                if check_clearance and s.get('kind')!='drop_box' and (not supported(s,self.slots,self.c['minimum_support_fraction']) or not clear(s,obstacles,self.c)): continue
                if any(abs(o['centre'][2]-s['centre'][2])<self.c['block_size'][2]-.0001 and overlap(bounds(s),bounds(o))>1e-10 for o in obstacles): continue
                plan_place(self.c,*s['centre'][:2],s['yaw'],z=s['centre'][2])
                s.update(state='RESERVED',transaction=uuid.uuid4().hex,brick=copy.deepcopy(brick)); self.save(); return copy.deepcopy(s)
            raise RuntimeError(f'No supported, clear destination capacity for {dest}')
    def transition(self,ident,state,evidence=''):
        with self.lock:
            s=next(s for s in self.slots if s['id']==ident)
            if state=='PLACED' and (s['state']!='RESERVED' or not evidence): raise ValueError('Commit requires reserved slot and verification evidence')
            if state not in ('EMPTY','RESERVED','PLACED','UNCERTAIN'): raise ValueError(state)
            s.update(state=state,evidence=evidence); self.save()


class Cancelled(RuntimeError): pass


class Manipulator:
    """One owner and durable hand context, shared by Pick, Place and observation motion."""
    def __init__(self,c,backend):
        self.c=c; self.backend=backend; self.lock=threading.Lock(); self.cancel=threading.Event()
        self.owner=open(inside(Path(c['runtime'])/'robot-owner.lock'),'a')
        fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.path=inside(Path(c['runtime'])/'hand.json'); self.log=Journal(Path(c['runtime'])/'manipulation.jsonl')
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(hand='UNRECONCILED',held=None,epoch=0,active=False)
        # Even a clean shutdown requires physical reconciliation on a hardware restart.
        if self.state.get('active') or c['mode']=='hardware': self.state['hand']='UNRECONCILED'
        self.state['active']=False; self.observation=None; self.lease=None; self.save()
    def close(self):
        if hasattr(self,'owner') and not self.owner.closed: self.owner.close()
    def __del__(self): self.close()
    def save(self): atomic(self.path,self.state)
    def invalidate(self):
        self.state['epoch']+=1; self.observation=None; self.save()
    def reconcile_empty(self,evidence):
        if not evidence: raise ValueError('Explicit physical reconciliation evidence required')
        if not self.lock.acquire(False): raise RuntimeError('Robot busy')
        try:
            self.backend.stationary()
            self.state.update(hand='EMPTY',held=None,active=False); self.cancel.clear(); self.invalidate()
            self.log.event('reconciled_empty',evidence=evidence)
        finally: self.lock.release()
    def resume(self):
        if not self.lock.acquire(False): raise RuntimeError('Robot busy')
        try:
            if self.state['hand']!='EMPTY': raise RuntimeError('Physical hand reconciliation required')
            self.backend.stationary(); self.cancel.clear(); self.invalidate()
        finally: self.lock.release()
    def stop(self): self.cancel.set()
    def check(self):
        if self.cancel.is_set(): raise Cancelled('Operator stop / action cancellation')
    def capture_begin(self):
        if not self.lock.acquire(False): raise RuntimeError('Robot busy')
        try:
            self.check()
            if self.state['hand']!='EMPTY': raise RuntimeError('Observation requires reconciled empty hand')
            self.backend.stationary()
            if hasattr(self.backend,'begin_capture'): self.backend.begin_capture()
            self.lease=uuid.uuid4().hex
            return self.lease,self.state['epoch']
        except Exception:
            self.lock.release(); raise
    def capture_end(self,token,observation_id,stamp_ns):
        if not self.lease or token!=self.lease: raise ValueError('Invalid capture lease')
        try:
            self.backend.stationary(); self.check()
            if hasattr(self.backend,'end_capture'): self.backend.end_capture()
            self.observation=dict(id=observation_id,stamp_ns=stamp_ns,epoch=self.state['epoch'])
        finally: self.lease=None; self.lock.release()
    def run(self,operation,target=None,epoch=None,feedback=lambda *_:None):
        if not self.lock.acquire(False): return self.result(False,'BUSY','Robot owned by another operation')
        stage='VALIDATE'
        try:
            self.check()
            if operation=='pick':
                if self.state['hand']!='EMPTY': raise ValueError('Pick requires empty hand')
                o=self.observation
                if not o or target['observation_id']!=o['id'] or epoch!=o['epoch'] or time.time_ns()-o['stamp_ns']>self.c['observation_max_age_s']*1e9:
                    raise ValueError('Stale observation')
                if target['frame']!=self.c['base_frame']: raise ValueError('Pick must be in base frame')
                if not np.allclose(target['dimensions'],self.c['block_size'],atol=1e-5): raise ValueError('Pick dimensions mismatch')
                r=matrix(target['centre'],target.get('quaternion',Rotation.from_euler('z',target['yaw']).as_quat()))[:3,:3]
                if r[2,2]<math.cos(math.radians(self.c['max_object_tilt_deg'])): raise ValueError('Tilted pick unsupported')
                plan=plan_pick(self.c,*target['centre'],yaw=target['yaw'])
                steps=['OPEN','APPROACH']+(['INSPECT'] if self.c.get('inspection_mode') else [])+['LOWER','GRASP','LIFT']
            elif operation=='place':
                if self.state['hand']!='HOLDING' or target['held_id']!=self.state['held']['held_id']: raise ValueError('No matching held object')
                if target.get('engage_studs'): raise ValueError('Stud engagement unsupported')
                if not np.allclose(target['dimensions'],self.state['held']['dimensions']): raise ValueError('Held dimensions mismatch')
                plan=plan_place(self.c,*target['centre'][:2],target['yaw'],z=target['centre'][2])
                steps=['TRANSFER_LIFT','TRANSFER','PLACE_LOWER','RELEASE','RETREAT']
            elif operation=='observe_pose':
                if self.state['hand']!='EMPTY': raise ValueError('Observation motion requires empty hand')
                plan={'above':self.c['observation_pose'][:3],'yaw':self.c['observation_pose'][3]}; steps=['OBSERVE_MOVE']
            else: raise ValueError(operation)
            self.backend.stationary()
            self.state['active']=True; self.invalidate()
            for stage in steps:
                self.check(); feedback(stage,self.state['hand']); self.check(); self.log.event('stage',operation=operation,stage=stage)
                if stage=='GRASP': self.state['hand']='GRASP_UNCERTAIN'; self.save()
                if stage=='RELEASE': self.state['hand']='RELEASE_UNCERTAIN'; self.save()
                self.backend.step(stage,plan,target,self.cancel)
                if stage=='GRASP':
                    self.state.update(hand='HOLDING',held=dict(target,held_id=uuid.uuid4().hex)); self.save()
                if stage=='RELEASE': self.state.update(hand='EMPTY',held=None); self.save()
            return self.result(True,'SUCCESS','Completed')
        except Exception as e:
            outcome='OPERATOR_STOP' if isinstance(e,Cancelled) else self.state['hand'] if self.state['hand'] in ('GRASP_UNCERTAIN','RELEASE_UNCERTAIN','HOLDING') else 'FAILURE_BEFORE_GRASP' if operation=='pick' else 'FAILED'
            # Failed/cancelled motion is never auto-restarted, even if the hand is known empty.
            if stage!='VALIDATE': self.cancel.set()
            self.log.event('failure',stage=stage,outcome=outcome,detail=str(e))
            return self.result(False,outcome,str(e))
        finally:
            self.state['active']=False; self.save(); self.lock.release()
    def result(self,success,outcome,detail):
        return dict(success=success,outcome=outcome,detail=detail,hand_state=self.state['hand'],held_id=(self.state.get('held') or {}).get('held_id',''))
