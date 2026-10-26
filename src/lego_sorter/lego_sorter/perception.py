"""Subprocess isolation, capture-time transforms and independent source-empty depth test."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import uuid
import numpy as np
from .core import adapt_detection, calibration, inside, layout, bounds


def empty_depth(depth,info,base_camera,c):
    """Require entire model footprint in view and nearly all valid pixels at bare support plane."""
    K=np.array(info['aligned_depth_intrinsics']['K']); inv=np.linalg.inv(K)
    if c.get('source_boundary'):
        x0,x1=c['source_boundary']['x']; y0,y1=c['source_boundary']['y']
    else:
        cells=layout(c['source'],c['block_size'],'source'); boxes=[bounds(s) for s in cells]
        x0=min(b[0] for b in boxes); x1=max(b[2] for b in boxes)
        y0=min(b[1] for b in boxes); y1=max(b[3] for b in boxes)
    z=c['table_top_z']
    corners=np.array([[x0,y0,z,1],[x1,y0,z,1],[x1,y1,z,1],[x0,y1,z,1]])
    camera=(np.linalg.inv(base_camera)@corners.T)[:3]; uv=K@camera
    if np.any(camera[2]<=0): return False,'Source footprint behind camera'
    uv=uv[:2]/uv[2]
    h,w=depth.shape
    if np.any(uv[0]<2) or np.any(uv[0]>w-3) or np.any(uv[1]<2) or np.any(uv[1]>h-3): return False,'Source footprint not fully visible'
    import cv2
    roi=np.zeros(depth.shape,np.uint8); cv2.fillConvexPoly(roi,np.round(uv.T).astype(np.int32),1)
    valid=(depth>0)&(roi>0)
    if roi.sum()<500 or valid.sum()/roi.sum()<.98: return False,'Insufficient valid source depth coverage'
    v,u=np.nonzero(valid); pc=(inv@np.stack([u,v,np.ones_like(u)]))*depth[v,u]*info['depth_scale_m']
    base=base_camera[:3,:3]@pc+base_camera[:3,3,None]
    error=base[2]-z
    # Conservative: holes, tall occluders and low surfaces are inconclusive.
    verified=bool(np.quantile(np.abs(error),.999)<c['empty_plane_tolerance'])
    return verified,f'Full-footprint depth: {valid.sum()} pixels; q99.9 plane error {np.quantile(np.abs(error),.999):.6f}m'


class SamProcess:
    def __init__(self,c,config_path):
        self.c=c; self.process=None; self.config_path=config_path
    def start(self):
        if self.process is not None: return
        runtime=inside(self.c['runtime']); self.stderr=open(runtime/'sam_worker.log','a')
        env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'
        env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
        worker=Path(__file__).with_name('sam_worker.py')
        self.process=subprocess.Popen([self.c['sam_python'],'-u',str(worker),self.config_path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1,cwd=runtime,env=env)
        self.selector=selectors.DefaultSelector(); self.selector.register(self.process.stdout,selectors.EVENT_READ)
        if self.read(180).get('event')!='ready': raise RuntimeError('SAM worker did not initialize')
    def read(self,timeout):
        if not self.selector.select(timeout):
            self.close(); raise TimeoutError('SAM worker timed out')
        line=self.process.stdout.readline()
        if not line: raise RuntimeError('SAM worker terminated; inspect sam_worker.log')
        data=json.loads(line)
        if data.get('event')=='error': raise RuntimeError(data['detail'])
        return data
    def send(self,data): self.process.stdin.write(json.dumps(data)+'\n'); self.process.stdin.flush()
    def observe(self,epoch,lookup):
        self.start(); oid=uuid.uuid4().hex
        self.send({'id':oid}); capture=self.read(15)
        if capture.get('event')!='captured' or capture['id']!=oid: raise RuntimeError('Invalid worker capture response')
        try:
            bp=lookup(capture['stamp_ns']); pc=calibration(self.c)
            self.send(dict(event='transform',id=oid,base_parent=bp.tolist(),parent_camera=pc.tolist()))
            result=self.read(self.c['snapshot_timeout_s'])
        except Exception: self.close(); raise
        if result.get('event')!='result' or result['id']!=oid: raise RuntimeError('Invalid detector response identity')
        bricks=adapt_detection(result['metadata'],oid,bp,pc,self.c)
        import cv2
        path=inside(result['path']); depth=cv2.imread(str(path/'depth.png'),cv2.IMREAD_UNCHANGED)
        info=json.loads((path/'camera_info.json').read_text())
        empty,evidence=empty_depth(depth,info,bp@pc,self.c)
        observation=dict(id=oid,stamp_ns=capture['stamp_ns'],epoch=epoch,bricks=bricks,empty_verified=empty and not bricks,
                    evidence=evidence,diagnostics=str(path),base_parent=bp,parent_camera=pc)
        (path/'normalized_observation.json').write_text(json.dumps(dict(observation,mode='replay',base_parent=bp.tolist(),parent_camera=pc.tolist()),indent=2))
        return observation
    def close(self):
        if self.process:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
            self.process=None; self.stderr.close()
