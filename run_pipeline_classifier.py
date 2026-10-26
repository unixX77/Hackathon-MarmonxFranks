"""Normal SAM3 pipeline geometry followed by the standalone modular classifier."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def run(script,*args):
    command=[sys.executable,str(ROOT/script),*map(str,args)]
    print('\nRunning: '+' '.join(command),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,help='Existing recording; omit for new D435 capture')
    p.add_argument('--frame',type=int,help='Default: latest saved frame')
    p.add_argument('--settings',type=Path,default=ROOT/'d435_settings.json')
    p.add_argument('--out',type=Path,default=ROOT/'data/pipeline_classifier_results')
    p.add_argument('--brick-size-mm',type=float,nargs=3,default=[30,15,10])
    p.add_argument('--colour-model',default=str(ROOT/'classifier/colour_model.json'),
                   help="Learned colour model JSON, or 'none' for the fixed Lab prototypes")
    p.add_argument('--detector-args',nargs=argparse.REMAINDER,default=[])
    a=p.parse_args()
    recording=a.recording.resolve() if a.recording else None
    settings=a.settings.resolve()
    if recording and not (recording/'camera_info.json').is_file():p.error('Recording not found')
    if a.frame is not None and a.frame<0:p.error('Frame must be nonnegative')
    colour_model=None if a.colour_model.lower()=='none' else Path(a.colour_model).resolve()
    if colour_model and not colour_model.is_file():
        p.error(f'Colour model not found: {colour_model}. Train it with classifier/colour_model.py or pass --colour-model none')
    output=a.out.resolve()/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:
        if recording is None:
            run('data_collection/capture_d435.py','--settings',settings,'--output',output/'00_capture')
            sessions=sorted(x for x in (output/'00_capture').iterdir() if x.is_dir())
            if len(sessions)!=1:raise RuntimeError('Expected one new capture session')
            recording=sessions[0]
        frames=[int(x.stem) for x in (recording/'rgb').glob('*.png') if x.stem.isdigit()]
        if not frames:raise RuntimeError('No saved frame. Press Enter before closing capture.')
        frame=f'{a.frame if a.frame is not None else max(frames):06d}'
        run('perception/sam3_detection/top_layer_bricks.py',recording,'--frame',frame,
            '--out',output/'00_normal_geometry','--brick-size-mm',*a.brick_size_mm,'--no-show',*a.detector_args)
        geometry=output/'00_normal_geometry'/recording.name/frame
        run('pipeline_classifier/reclassify.py',geometry,'--out',output,
            *(['--colour-model',colour_model] if colour_model else []))
        destination=output/'03_final'/'capture'/recording.name/frame
        (output/'pipeline.json').write_text(json.dumps({'recording':str(recording),'frame':frame,
            'original_result':str(geometry),'final_result':str(destination),
            'change':'standalone classification only; original detection and 3D geometry preserved'},indent=2))
        print('\nFinal results:',destination)
    except Exception as error:
        (output/'INCOMPLETE.json').write_text(json.dumps({'error':str(error)},indent=2))
        raise


if __name__=='__main__':main()
