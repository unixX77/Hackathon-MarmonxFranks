"""Persistent SAM3/D435 owner. JSON-lines IPC; runs in the SAM3 Python environment."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    import yaml
    c=yaml.safe_load(Path(sys.argv[1]).read_text())
    root=Path('/home/tanay/Documents/Hackathon/project-planner').resolve()
    runtime=Path(c['runtime']).resolve()
    if not runtime.is_relative_to(root): raise ValueError('Runtime outside integration directory')
    runtime.mkdir(parents=True,exist_ok=True)
    lock=open(runtime/'camera.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for key in ['HF_HOME','TORCH_HOME','XDG_CACHE_HOME','TMPDIR']:
        path=runtime/'cache'/key; path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
    sys.path.insert(0,str(Path(__file__).parent/'vendor'))
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from transformers import Sam3Model,Sam3Processor
        from top_layer_bricks import build_parser,process_frame,save_result
        if not c['model_path'] or not Path(c['model_path']).exists(): raise ValueError('Explicit local SAM3 model path required')
        device='cuda' if torch.cuda.is_available() else 'cpu'
        dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
        processor=Sam3Processor.from_pretrained(c['model_path'],local_files_only=True)
        model=Sam3Model.from_pretrained(c['model_path'],torch_dtype=dtype,local_files_only=True).to(device).eval()
    pipe=rs.pipeline(); config=rs.config()
    if c['camera_serial']: config.enable_device(c['camera_serial'])
    config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
    config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    profile=pipe.start(config); align=rs.align(rs.stream.color)
    for sensor in profile.get_device().query_sensors():
        if sensor.supports(rs.option.global_time_enabled): sensor.set_option(rs.option.global_time_enabled,1)
    if c.get('camera_settings'):
        settings=json.loads(Path(c['camera_settings']).read_text())
        # Explicit rs400 advanced settings JSON, not a capture application's unrelated schema.
        rs.rs400_advanced_mode(profile.get_device()).load_json(json.dumps(settings))
    scale=profile.get_device().first_depth_sensor().get_depth_scale()
    # Match the standalone capture test: discard startup frames while camera
    # exposure/depth settle. The requested observation is still captured later.
    warmup_end=time.monotonic()+3.0
    while True:
        pipe.wait_for_frames(3000)
        if time.monotonic()>=warmup_end: break
    def send(value): print(json.dumps(value,allow_nan=False),flush=True)
    send({'event':'ready'})
    try:
        for line in sys.stdin:
            req=json.loads(line); oid=req['id']; output=(runtime/'observations'/oid).resolve()
            if not output.is_relative_to(runtime/'observations'): raise ValueError('Invalid observation ID')
            try:
                requested=time.time_ns(); deadline=time.monotonic()+8
                while True:
                    frames=align.process(pipe.wait_for_frames(3000)); rgb=frames.get_color_frame(); depth=frames.get_depth_frame()
                    if not rgb or not depth: continue
                    stamp_ns=int(rgb.get_timestamp()*1e6)
                    if rgb.get_frame_timestamp_domain()!=rs.timestamp_domain.global_time:
                        if time.monotonic()>deadline: raise RuntimeError('D435 global timestamp domain unavailable')
                        continue
                    if requested<=stamp_ns<=time.time_ns()+10_000_000: break
                    if time.monotonic()>deadline: raise RuntimeError('No fresh globally timestamped frame')
                bgr=np.asanyarray(rgb.get_data()).copy(); depth_image=np.asanyarray(depth.get_data()).copy()
                intr=depth.profile.as_video_stream_profile().intrinsics
                camera_info={'aligned_depth_intrinsics':{
                    'width':intr.width,'height':intr.height,'fx':intr.fx,'fy':intr.fy,
                    'ppx':intr.ppx,'ppy':intr.ppy,'distortion_coeffs':list(intr.coeffs),
                    'K':[[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]]},'depth_scale_m':scale}
                output.mkdir(parents=True)
                cv2.imwrite(str(output/'rgb.png'),bgr); cv2.imwrite(str(output/'depth.png'),depth_image)
                (output/'camera_info.json').write_text(json.dumps(camera_info))
                quality=dict(usable_depth_fraction=float(((depth_image>0)&(depth_image<65535)).mean()),
                             zero_depth_fraction=float((depth_image==0).mean()),
                             max_raw_depth_fraction=float((depth_image==65535).mean()))
                (output/'capture_quality.json').write_text(json.dumps(quality,indent=2))
                send({'event':'captured','id':oid,'stamp_ns':stamp_ns,'path':str(output),'capture_quality':quality})
                # Capture-time TF is fetched by ROS before allowing inference to proceed.
                ack=json.loads(sys.stdin.readline())
                if ack.get('event')!='transform' or ack.get('id')!=oid: raise RuntimeError('Missing capture transform acknowledgement')
                (output/'capture.json').write_text(json.dumps(dict(ack,stamp_ns=stamp_ns)))
                args=build_parser().parse_args([str(output),'--no-show','--brick-size-mm',*[str(x*1000) for x in c['block_size']],'--top-min-support','1'])
                args.block_height_mm=args.brick_size_mm[2]
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        images,metadata,final=process_frame(bgr,depth_image,camera_info,model,processor,args,device,dtype)
                        save_result(output,images,metadata,final)
                except ValueError as e:
                    # No foreground is inconclusive; depth verification runs independently in ROS.
                    if 'No foreground instances' not in str(e): raise
                    metadata={'brick_size_mm':args.brick_size_mm,'bricks':[],'detector_warning':str(e)}
                    (output/'bricks.json').write_text(json.dumps(metadata))
                send({'event':'result','id':oid,'metadata':metadata,'path':str(output),'stamp_ns':stamp_ns})
            except Exception as e:
                traceback.print_exc(file=sys.stderr); send({'event':'error','id':oid,'detail':str(e)})
    finally: pipe.stop()


if __name__=='__main__': main()
