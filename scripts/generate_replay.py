#!/usr/bin/python3 -s
"""Generate a complete synthetic replay fixture without ROS, camera or robot."""
import json
from pathlib import Path
import time
import yaml
from lego_sorter.core import ROOT,load_config,Store,Manipulator,select_brick
from lego_sorter.workflow import MockWorld,LocalTransport
runtime=ROOT/'runtime'/f'replay-fixture-{time.time_ns()}'; runtime.mkdir()
c=load_config(ROOT/'src/lego_sorter/config/sorter.yaml'); c['runtime']=str(runtime/'generation'); Path(c['runtime']).mkdir()
w=MockWorld(c); m=Manipulator(c,w); m.reconcile_empty('SIMULATED fixture'); t=LocalTransport(m,w); s=Store(c); paths=[]
while True:
    assert t.observation_pose()['success']
    o=t.observe(); path=runtime/f'{len(paths):04d}.json'; path.write_text(json.dumps(dict(o,mode='replay'),indent=2)); paths.append(str(path))
    if o['empty_verified']: break
    b=select_brick(o['bricks'],c); slot=s.reserve(b); pick=t.pick(b,o); assert pick['success']
    assert t.place(slot,pick['held_id'])['success']; s.transition(slot['id'],'PLACED','SIMULATED fixture')
c.update(mode='replay',runtime=str(runtime/'execution'),replay_observations=paths)
(runtime/'config.yaml').write_text(yaml.safe_dump(c)); print(runtime/'config.yaml')
