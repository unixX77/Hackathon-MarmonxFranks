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
    p.add_argument('--auto-capture',action='store_true',
                   help='New capture without the window: one frame is saved after about a second, no Enter / q')
    p.add_argument('--settings',type=Path,default=ROOT/'d435_settings.json')
    p.add_argument('--out',type=Path,default=ROOT/'data/pipeline_classifier_results')
    p.add_argument('--brick-size-mm',type=float,nargs=3,default=[32,16,9.6])
    p.add_argument('--colour-model',default=str(ROOT/'classifier/colour_model.json'),
                   help="Learned colour model JSON, or 'none' for the fixed Lab prototypes")
    p.add_argument('--second-opinion',nargs='?',const='claude',choices=['claude','codex'],
                   help='Ask a vision model to check the final colours: claude (Anthropic API key, '
                        'the default when no value is given) or codex (Codex CLI login)')
    p.add_argument('--second-opinion-effort',default='medium',choices=['low','medium','high','xhigh','max'])
    p.add_argument('--review',nargs='?',const='claude',choices=['claude','codex'],
                   help='A vision model checks the result, fixes wrong colours, drops wrong masks and adds '
                        'its result to preview.png: claude (claude -p, the default when no value is given) '
                        'or codex (codex exec)')
    p.add_argument('--review-model',help='default: opus for claude, your Codex config for codex')
    p.add_argument('--review-effort',default='medium',choices=['low','medium','high','xhigh','max'])
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
            run('data_collection/capture_d435.py','--settings',settings,'--output',output/'00_capture',
                *(['--auto'] if a.auto_capture else []))
            sessions=sorted(x for x in (output/'00_capture').iterdir() if x.is_dir())
            if len(sessions)!=1:raise RuntimeError('Expected one new capture session')
            recording=sessions[0]
        frames=[int(x.stem) for x in (recording/'rgb').glob('*.png') if x.stem.isdigit()]
        if not frames:raise RuntimeError('No saved frame. Press Enter before closing capture.')
        frame=f'{a.frame if a.frame is not None else max(frames):06d}'
        run('perception/sam3_detection/top_layer_bricks.py',recording,'--frame',frame,
            '--out',output/'00_normal_geometry','--brick-size-mm',*a.brick_size_mm,'--no-show','--no-print-poses',*a.detector_args)
        geometry=output/'00_normal_geometry'/recording.name/frame
        run('pipeline_classifier/reclassify.py',geometry,'--out',output,
            *(['--colour-model',colour_model] if colour_model else []))
        destination=output/'03_final'/'capture'/recording.name/frame
        (output/'pipeline.json').write_text(json.dumps({'recording':str(recording),'frame':frame,
            'original_result':str(geometry),'final_result':str(destination),
            'change':'standalone classification only; original detection and 3D geometry preserved'},indent=2))
        sys.path.insert(0,str(ROOT/'perception/sam3_detection'))
        from top_layer_bricks import pose_line
        bricks_file=destination/'bricks.json'
        if a.second_opinion:
            sys.path.insert(0,str(ROOT/'classifier'))
            from claude_second_opinion import review
            try:
                if review(destination,a.second_opinion_effort,a.second_opinion):
                    bricks_file=destination/'bricks_second_opinion.json'
            except Exception as error:  # the classifier result stands on its own
                print(f'Second opinion skipped: {error}')
        if a.review:
            sys.path.insert(0,str(ROOT/'classifier'))
            from claude_review import review
            try:
                bricks_file=review(destination,a.review_model,a.review_effort,backend=a.review) or bricks_file
            except Exception as error:  # the classifier result stands on its own
                print(f'{a.review.capitalize()} review skipped: {error}')
        final=[b for b in json.loads(bricks_file.read_text())['bricks'] if b.get('id')]
        print(f'\n{len(final)} bricks. x y z = top-face centre in camera_color_optical_frame (m); '
              'angle = long side in the image from +x (right) toward +y (down), deg in (-90, 90]')
        for brick in final:
            opinion=brick.get('second_opinion',{})
            fixed=brick.get('claude_review',{})
            print(pose_line(brick)+(f"  CHECK: second opinion says {opinion['colour']}" if opinion.get('status')=='disagree' else '')
                  +(f"  ({(a.review or 'claude').capitalize()} corrected from {fixed['classifier_colour']})"
                    if fixed.get('action')=='recoloured' else ''))
        print('\nFinal results:',destination)
    except Exception as error:
        (output/'INCOMPLETE.json').write_text(json.dumps({'error':str(error)},indent=2))
        raise


if __name__=='__main__':main()
