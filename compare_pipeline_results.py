"""Side-by-side comparison of run_pipeline.sh, run_modular_pipeline.sh and run_pipeline_classifier.sh.

All three results are drawn the same way on the raw RGB frame so colours can be
compared directly: green outline = good, red = bad, label = colour (R red,
DR dark red, O dark orange, N medium nougat, B reddish brown), '?' = low
confidence.  Layout per capture: RGB | run_pipeline.sh / run_modular | classifier.

Rebuild the comparisons and index.html of a run_all_captures.py batch:
  python compare_pipeline_results.py data/all_captures_results/batch_<timestamp>
"""
import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

SHORT={'red':'R','dark_red':'DR','dark_orange':'O','medium_nougat':'N','reddish_brown':'B'}
PIPELINES=[('normal','run_pipeline.sh'),('modular','run_modular_pipeline.sh'),('final','run_pipeline_classifier.sh')]


def load_bricks(path):
    """Normal or final bricks.json -> uniform box list."""
    result=json.loads(Path(path).read_text())
    return [{'corners':b['footprint_px'],'colour':b.get('colour'),'quality':b.get('quality'),
             'low':b.get('colour_confidence')=='low'} for b in result['bricks'] if b.get('id')]


def load_quality_boxes(path):
    """Modular quality_boxes.json -> uniform box list."""
    result=json.loads(Path(path).read_text())
    return [{'corners':b['corners_xy'],'colour':b['colour'],'quality':b['quality'],
             'low':b.get('confidence')=='low'} for b in result['boxes']]


def counts(boxes):
    if boxes is None:return None
    bad=sum(b['quality']=='bad' for b in boxes)
    return {'bricks':len(boxes),'good':len(boxes)-bad,'bad':bad,'low_confidence':sum(b['low'] for b in boxes)}


def title(image,text,scale):
    bar=int(34*scale)
    cv2.rectangle(image,(0,0),(image.shape[1],bar),(0,0,0),-1)
    cv2.putText(image,text,(int(10*scale),int(24*scale)),cv2.FONT_HERSHEY_SIMPLEX,.65*scale,(255,255,255),
                max(1,int(round(1.5*scale))),cv2.LINE_AA)
    return image


def panel(bgr,boxes,error,name,scale):
    out=bgr.copy()
    if boxes is None:
        out=(out*.35).astype(np.uint8)
        text=f'FAILED: {error or "no result"}'[:90]
        cv2.putText(out,text,(int(20*scale),out.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,.6*scale,(80,80,255),
                    max(1,int(round(1.5*scale))),cv2.LINE_AA)
        return title(out,f'{name}: failed',scale)
    for b in boxes:
        colour=(60,210,60) if b['quality']=='good' else (40,40,240)
        corners=np.rint(b['corners']).astype(np.int32)
        cv2.polylines(out,[corners],True,colour,max(2,int(round(2*scale))),cv2.LINE_AA)
        label=SHORT.get(b['colour'],str(b['colour']))+('?' if b['low'] else '')
        x,y=corners.mean(axis=0).astype(int)
        size,_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.6*scale,1)
        org=(x-size[0]//2,y+size[1]//2)
        cv2.putText(out,label,org,cv2.FONT_HERSHEY_SIMPLEX,.6*scale,(0,0,0),max(3,int(4*scale)),cv2.LINE_AA)
        cv2.putText(out,label,org,cv2.FONT_HERSHEY_SIMPLEX,.6*scale,(255,255,255),max(1,int(round(1.5*scale))),cv2.LINE_AA)
    c=counts(boxes)
    return title(out,f"{name}: {c['bricks']} bricks, {c['good']} good / {c['bad']} bad"
                 +(f", {c['low_confidence']} unsure" if c['low_confidence'] else ''),scale)


def make_comparison(rgb_path,results,out_path,heading=''):
    """results: {key: (boxes or None, error)} for the PIPELINES keys. Writes a 2x2 grid."""
    bgr=cv2.imread(str(rgb_path))
    if bgr is None:raise FileNotFoundError(rgb_path)
    corners=[np.asarray(b['corners']) for boxes,_ in results.values() if boxes for b in boxes]
    if corners:
        # Crop to the union of every pipeline's boxes so the stack fills each panel.
        points=np.concatenate(corners)
        margin=60
        x0,y0=np.maximum(points.min(axis=0).astype(int)-margin,0)
        x1,y1=np.minimum(points.max(axis=0).astype(int)+margin,[bgr.shape[1],bgr.shape[0]])
        offset=np.array([x0,y0])
        bgr=bgr[y0:y1,x0:x1]
        results={k:([{**b,'corners':np.asarray(b['corners'])-offset} for b in boxes] if boxes is not None else None,e)
                 for k,(boxes,e) in results.items()}
    scale=max(.6,min(1.6,bgr.shape[1]/900))
    panels=[title(bgr.copy(),f'RGB {heading}'.strip(),scale)]
    panels+=[panel(bgr,*results.get(key,(None,'not run')),name,scale) for key,name in PIPELINES]
    grid=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out_path),grid)
    return out_path


def log_error(log_path):
    """First 'SomethingError: message' line of a failed run's log, if any."""
    if not log_path.is_file():return None
    for line in log_path.read_text(errors='replace').splitlines():
        head=line.split(':',1)[0].strip()
        if head.endswith(('Error','Exception')) and ' ' not in head and ':' in line:
            return line.strip()
    return None


def collect(job_dir):
    """Boxes and errors of the three pipelines inside one run_all_captures.py job folder."""
    job_dir=Path(job_dir)
    results=_collect(job_dir)
    rgb=results.pop('rgb')
    logs={'normal':'classifier','final':'classifier','modular':'modular'}
    for key,(boxes,error) in list(results.items()):
        if boxes is None and error and error!='not run':
            # Runner errors only name the failed subprocess; the log holds the real reason.
            results[key]=(None,log_error(job_dir.parent/'logs'/f'{job_dir.name}_{logs[key]}.log') or error)
    return rgb,results


def _collect(job_dir):
    results,rgb={},None
    runs=sorted(job_dir.glob('run_*'))
    if runs and (runs[-1]/'pipeline.json').is_file():
        manifest=json.loads((runs[-1]/'pipeline.json').read_text())
        rgb=Path(manifest['recording'])/'rgb'/f"{int(manifest['frame']):06d}.png"
        results['normal']=(load_bricks(Path(manifest['original_result'])/'bricks.json'),None)
        results['final']=(load_bricks(Path(manifest['final_result'])/'bricks.json'),None)
    else:
        error=json.loads((runs[-1]/'INCOMPLETE.json').read_text())['error'] if runs and (runs[-1]/'INCOMPLETE.json').is_file() else None
        geometry=sorted(runs[-1].glob('00_normal_geometry/*/*/bricks.json')) if runs else []
        results['normal']=(load_bricks(geometry[0]),None) if geometry else (None,error)
        results['final']=(None,error)
    modular=sorted((job_dir/'modular').glob('run_*'))
    if modular and (modular[-1]/'pipeline.json').is_file():
        manifest=json.loads((modular[-1]/'pipeline.json').read_text())
        rgb=rgb or Path(manifest['recording'])/'rgb'/f"{int(manifest['frame']):06d}.png"
        results['modular']=(load_quality_boxes(manifest['classification']),None)
    elif modular and (modular[-1]/'INCOMPLETE.json').is_file():
        results['modular']=(None,json.loads((modular[-1]/'INCOMPLETE.json').read_text())['error'])
    else:
        results['modular']=(None,'not run')
    results['rgb']=rgb
    return results


def write_index(batch,rows):
    """Self-contained local HTML page listing every comparison with its counts."""
    def cell(row,key):
        c=row.get(key)
        return '<td class="na">failed</td>' if not c else f"<td>{c['bricks']}</td><td>{c['good']}</td><td class='bad'>{c['bad']}</td>"
    body=[]
    for row in rows:
        image=Path(row['comparison']).relative_to(batch) if row.get('comparison') else None
        body.append(f"<tr><td><a href='#{html.escape(row['name'])}'>{html.escape(row['name'])}</a></td>"
                    +''.join(cell(row,key) for key,_ in PIPELINES)+'</tr>')
    cards=''.join(f"<section id='{html.escape(r['name'])}'><h2>{html.escape(r['name'])}</h2>"
                  f"<p>{html.escape(r['recording'])}, frame {r['frame']}</p>"
                  +(f"<a href='{Path(r['comparison']).relative_to(batch)}'><img loading='lazy' src='{Path(r['comparison']).relative_to(batch)}'></a>"
                    if r.get('comparison') else '<p>No comparison image.</p>')+'</section>' for r in rows)
    heads=''.join(f"<th colspan=3>{name}</th>" for _,name in PIPELINES)
    page=f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pipeline Comparison</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#ddd;--bad:#c62828}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--muted:#aaa;--line:#333;--bad:#ff6b6b}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}}
table{{border-collapse:collapse;margin-bottom:24px}}th,td{{border:1px solid var(--line);padding:4px 8px;text-align:right}}
td:first-child{{text-align:left}}.bad{{color:var(--bad)}}.na{{color:var(--muted)}}a{{color:inherit}}
section{{margin:28px 0;border-top:1px solid var(--line)}}p{{color:var(--muted)}}img{{max-width:100%;height:auto}}
</style></head><body><h1>Pipeline comparison ({len(rows)} captures)</h1>
<p>Green outline = good, red = bad. R red, DR dark red, O dark orange, N medium nougat, B reddish brown; ? = low confidence.</p>
<table><tr><th rowspan=2>capture</th>{heads}</tr><tr>{'<th>bricks</th><th>good</th><th>bad</th>'*len(PIPELINES)}</tr>
{''.join(body)}</table>{cards}</body></html>"""
    (Path(batch)/'index.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('batch',type=Path,help='run_all_captures.py batch folder')
    a=p.parse_args()
    batch=a.batch.resolve()
    summary=json.loads((batch/'summary.json').read_text())['results']
    rows=[]
    for row in summary:
        rgb,results=collect(batch/row['name'])
        row={**row,**{key:counts(results.get(key,(None,))[0]) for key,_ in PIPELINES}}
        if rgb and rgb.is_file():
            row['comparison']=str(make_comparison(rgb,results,batch/'comparisons'/f"{row['name']}.png",row['name']))
        rows.append(row)
        print(row['name'],{key:row[key] for key,_ in PIPELINES})
    write_index(batch,rows)
    print('Wrote',batch/'index.html')


if __name__=='__main__':main()
