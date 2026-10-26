"""Run the pipelines on every saved D435 capture in the repository and compare them.

For each capture this runs run_pipeline_classifier.py (whose unchanged geometry
stage is exactly run_pipeline.sh's result) and run_modular_pipeline.py, each
into its normal per-run result folder.  The batch folder adds:

  comparisons/<capture>.png  RGB | run_pipeline.sh / run_modular_pipeline.sh | run_pipeline_classifier.sh
  index.html                 table of counts per pipeline plus every comparison
  summary.csv, summary.json  one row per capture
  logs/                      full output of every run
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from compare_pipeline_results import PIPELINES, collect, counts, make_comparison, write_index

ROOT=Path(__file__).resolve().parent
DEFAULT_ROOTS=['data/captures','data/manual_pipeline/captures',
               'data/pipeline_classifier_results/*/00_capture','data/modular_pipeline_results/*/inputs/capture',
               'data/temporal_pipeline_results/*/00_capture']


def frames(recording):
    return sorted(int(p.stem) for p in (recording/'rgb').glob('*.png')
                  if p.stem.isdigit() and (recording/'depth'/p.name).is_file())


def discover(patterns):
    """Recording folders (camera_info.json + numbered rgb/depth), deduplicated through symlinks."""
    found={}
    for pattern in patterns:
        for parent in sorted(ROOT.glob(pattern)):
            for recording in sorted(p for p in parent.iterdir() if p.is_dir()):
                if (recording/'camera_info.json').is_file() and frames(recording):
                    found.setdefault(recording.resolve(),recording)
    return list(found.values())


def run(command,log_path):
    with log_path.open('w') as log:
        return subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roots',nargs='+',default=DEFAULT_ROOTS,
                   help='Glob patterns (relative to the repo) of folders that contain recordings')
    p.add_argument('--out',type=Path,default=ROOT/'data/all_captures_results')
    p.add_argument('--all-frames',action='store_true',help='Process every saved frame, not just the latest')
    p.add_argument('--colour-model',default=str(ROOT/'classifier/colour_model.json'))
    p.add_argument('--no-modular',action='store_true',help='Skip run_modular_pipeline.py')
    p.add_argument('--limit',type=int,help='Process only the first N jobs (for a quick test)')
    p.add_argument('--dry-run',action='store_true',help='List the jobs without running them')
    p.add_argument('--detector-args',nargs=argparse.REMAINDER,default=[])
    a=p.parse_args()
    jobs=[]
    for recording in discover(a.roots):
        saved=frames(recording)
        for frame in (saved if a.all_frames else saved[-1:]):
            jobs.append((recording,frame))
    jobs=jobs[:a.limit] if a.limit else jobs
    for recording,frame in jobs:print(f'{recording.relative_to(ROOT)}  frame {frame}')
    print(f'{len(jobs)} jobs')
    if a.dry_run or not jobs:return
    batch=a.out.resolve()/datetime.now().strftime('batch_%Y%m%d_%H%M%S')
    (batch/'logs').mkdir(parents=True)
    rows=[]
    for index,(recording,frame) in enumerate(jobs,1):
        name=f'{recording.name}_{frame:06d}'
        while (batch/name).exists():name+='_dup'
        print(f'\n[{index}/{len(jobs)}] {recording.relative_to(ROOT)} frame {frame}',flush=True)
        start=time.time()
        command=['run_pipeline_classifier.py','--recording',recording,'--frame',frame,
                 '--out',batch/name,'--colour-model',a.colour_model]
        if a.detector_args:command+=['--detector-args',*a.detector_args]
        classifier_code=run(command,batch/'logs'/f'{name}_classifier.log')
        modular_code=None
        if not a.no_modular:
            modular_code=run(['run_modular_pipeline.py','--recording',recording,'--frame',frame,
                              '--out',batch/name/'modular'],batch/'logs'/f'{name}_modular.log')
        rgb,results=collect(batch/name)
        row={'name':name,'recording':str(recording),'frame':frame,'seconds':round(time.time()-start,1),
             'classifier_exit':classifier_code,'modular_exit':modular_code}
        for key,_ in PIPELINES:
            boxes,error=results.get(key,(None,'not run'))
            row[key]=counts(boxes)
            row[f'{key}_error']=error
        row['comparison']=(str(make_comparison(rgb,results,batch/'comparisons'/f'{name}.png',name))
                           if rgb and rgb.is_file() else None)
        rows.append(row)
        print('  '+' | '.join(f"{label}: "+(f"{row[key]['good']} good / {row[key]['bad']} bad" if row[key] else 'failed')
                              for key,label in PIPELINES)+f" ({row['seconds']}s)",flush=True)
        # Rewrite outputs after every job so an interrupted batch keeps its results.
        (batch/'summary.json').write_text(json.dumps({'colour_model':a.colour_model,'results':rows},indent=2))
        flat=[{**{k:v for k,v in r.items() if k not in dict(PIPELINES)},
               **{f'{key}_{field}':(r[key] or {}).get(field,'') for key,_ in PIPELINES
                  for field in ('bricks','good','bad','low_confidence')}} for r in rows]
        with (batch/'summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        write_index(batch,rows)
    print(f'\nDone. Open {batch/"index.html"}')


if __name__=='__main__':main()
