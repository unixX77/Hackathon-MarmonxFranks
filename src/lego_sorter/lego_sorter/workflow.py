"""Complete mission logic and deterministic, explicitly simulated world."""
import copy
import json
from pathlib import Path
import threading
import time
import uuid
from .core import atomic, inside, Journal, layout, select_brick, Cancelled


class MockWorld:
    def __init__(self,c):
        self.c=c; self.path=inside(Path(c['runtime'])/'mock_world.json')
        if not self.path.exists():
            bricks=layout(c['source'],c['block_size'],'source')
            count=c.get('mock_brick_count',len(bricks))
            if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=len(bricks):
                raise ValueError(f'mock_brick_count must be between 1 and {len(bricks)}')
            bricks=bricks[:count]
            colours=c['mock_colours']
            for i,b in enumerate(bricks):
                b.update(colour=colours[i%len(colours)],confidence='high',quality='mock',frame=c['base_frame'])
                b['top']=[*b['centre'][:2],b['centre'][2]+c['block_size'][2]/2]
            atomic(self.path,dict(bricks=bricks,held=None,placed={},steps=0))
    def read(self): return json.loads(self.path.read_text())
    def stationary(self): pass
    def step(self,stage,plan,target,cancel):
        if cancel.is_set(): raise Cancelled('Mock cancelled')
        if cancel.wait(self.c.get('mock_step_delay_s',0)): raise Cancelled('Mock cancelled')
        w=self.read(); w['steps']+=1
        if self.c.get('mock_fail_stage')==stage:
            raise RuntimeError('Injected mock failure: '+stage)
        if stage=='GRASP':
            b=next((b for b in w['bricks'] if b['centre']==target['centre']),None)
            if b is None: raise ValueError('Mock target no longer present')
            w['bricks'].remove(b); w['held']=b
        if stage=='RELEASE':
            if not w['held']: raise ValueError('Mock world hand empty')
            w['placed'][target['id']]=dict(w['held'],centre=target['centre'],yaw=target['yaw']); w['held']=None
        atomic(self.path,w)
    def observe(self,epoch=0):
        w=self.read(); oid=uuid.uuid4().hex
        top=max((b['top'][2] for b in w['bricks']),default=0)
        bricks=[dict(b,id=str(i),observation_id=oid) for i,b in enumerate(w['bricks']) if top-b['top'][2]<self.c['top_layer_tolerance']]
        return dict(id=oid,stamp_ns=time.time_ns(),epoch=epoch,bricks=bricks,empty_verified=not w['bricks'],
                    evidence='SIMULATED complete-world occupancy',diagnostics='',base_parent=None,parent_camera=None)
    def verify(self,slot):
        actual=self.read()['placed'].get(slot['id'])
        return (bool(actual and actual['centre']==slot['centre']), 'SIMULATED world placement at reserved slot')


class ReplayWorld(MockWorld):
    """Replay observations are evidence, never evidence of a real grasp or placement."""
    def observe(self,epoch=0):
        paths=self.c['replay_observations']; index=len(self.read()['placed'])
        if index>=len(paths): raise RuntimeError('Replay exhausted; no implicit source-empty success')
        o=json.loads(Path(paths[index]).read_text())
        if o.get('mode')!='replay': raise ValueError('Replay records must be explicitly tagged mode: replay')
        oid=uuid.uuid4().hex
        return dict(o,id=oid,stamp_ns=time.time_ns(),epoch=epoch,
                    bricks=[dict(b,observation_id=oid) for b in o['bricks']])


class LocalTransport:
    def __init__(self,m,world,stage_feedback=None):
        self.m=m; self.world=world
        self.stage_feedback=stage_feedback or (lambda *_:None)
    def observation_pose(self): return self.m.run('observe_pose',feedback=self.stage_feedback)
    def observe(self):
        self.stage_feedback('CAPTURE',self.hand())
        token,epoch=self.m.capture_begin()
        try: o=self.world.observe(epoch)
        except Exception:
            self.m.capture_end(token,'',0); raise
        self.m.capture_end(token,o['id'],o['stamp_ns']); return o
    def pick(self,b,o): return self.m.run('pick',b,o['epoch'],feedback=self.stage_feedback)
    def place(self,s,held): return self.m.run('place',dict(s,held_id=held,engage_studs=False),feedback=self.stage_feedback)
    def verify(self,s):
        self.stage_feedback('VERIFY_PLACEMENT',self.hand())
        return self.world.verify(s)
    def hand(self): return self.m.state['hand']
    def stop(self): self.m.stop()
    def resume(self):
        self.m.resume()


class Coordinator:
    def __init__(self,c,store,transport):
        self.c=c; self.store=store; self.t=transport
        self.path=inside(Path(c['runtime'])/'mission.json')
        previous=json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data=dict(status='RECONCILE' if previous else 'IDLE',completed=previous.get('completed',0),detail='',observation=None)
        self.log=Journal(Path(c['runtime'])/'events.jsonl'); self.lock=threading.RLock(); self.pause=False; self.stopped=False
        self.save()
    def save(self): atomic(self.path,self.data)
    def status(self,state,detail=''):
        self.data.update(status=state,detail=detail); self.save(); self.log.event('status',state=state,detail=detail)
    def control(self,command):
        with self.lock:
            if command=='pause': self.pause=True; return
            if command=='stop':
                self.stopped=True; self.t.stop()
                if self.data['status']!='RUNNING': self.status('STOPPED')
                return
            if command not in ('start','resume'): raise ValueError('Use start, pause, resume, stop')
            if self.data['status'] not in ('IDLE','PAUSED','STOPPED','RECONCILE','FAILED'):
                raise RuntimeError('Mission already running or complete')
            if any(s['state'] in ('RESERVED','UNCERTAIN') for s in self.store.slots):
                raise RuntimeError('Destination reconciliation required')
            self.t.resume()  # requires reconciled EMPTY; no blind replay of held/uncertain operations
            self.pause=False; self.stopped=False; self.data['observation']=None; self.status('RUNNING')
    def tick(self):
        if self.data['status']!='RUNNING': return False
        if self.stopped: self.status('STOPPED'); return False
        if self.pause: self.status('PAUSED'); return False
        slot=None; picked=False
        try:
            result=self.t.observation_pose()
            if not result['success']: raise RuntimeError(result['detail'])
            observation=None
            for attempt in range(self.c['observe_attempts']):
                if self.stopped: raise Cancelled('Stopped before observation')
                try:
                    observation=self.t.observe()
                    if observation['bricks'] or observation['empty_verified']: break
                except Exception:
                    if attempt+1==self.c['observe_attempts']: raise
            if not observation: raise RuntimeError('Observation failed')
            self.data['observation']=observation['id']; self.save()
            if not observation['bricks']:
                if observation['empty_verified']:
                    self.status('COMPLETE',observation['evidence']); return False
                raise RuntimeError('No detections is inconclusive; source-empty verification failed')
            brick=select_brick(observation['bricks'],self.c)
            if brick is None: raise RuntimeError('No accessible top-layer brick; finger clearance or geometry must be resolved')
            slot=self.store.reserve(brick); self.log.event('reserved',slot=slot['id'],transaction=slot['transaction'],observation=observation['id'],local_id=brick['id'])
            if self.stopped: raise Cancelled('Stopped before pick')
            r=self.t.pick(brick,observation); self.data['observation']=None; self.save()
            if not r['success']: raise RuntimeError(r['outcome']+': '+r['detail'])
            picked=True
            if self.stopped: raise Cancelled('Stopped holding brick')
            r=self.t.place(slot,r['held_id'])
            if not r['success']: raise RuntimeError(r['outcome']+': '+r['detail'])
            verified,evidence=self.t.verify(slot)
            if not verified: raise RuntimeError('Placement unverified: '+evidence)
            self.store.transition(slot['id'],'PLACED',evidence)
            self.data['completed']+=1; self.save(); self.log.event('committed',slot=slot['id'],evidence=evidence)
            return True
        except Exception as e:
            if slot:
                # Free only when the hand is conclusively empty AND no grasp completed.
                uncertain=picked or self.t.hand()!='EMPTY'
                self.store.transition(slot['id'],'UNCERTAIN' if uncertain else 'EMPTY',str(e))
            self.data['observation']=None
            self.status('STOPPED' if self.stopped else 'FAILED',str(e)); return False
