"""Show brick crops in a window; choose lighting categories in the terminal."""
import argparse
import csv
from pathlib import Path
import json
import select
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OPTIONS={
 'A':('normal_light','normal','none'),
 'B':('low_light','low','none'),
 'C':('low_light_slight_glare','low','slight'),
 'D':('normal_light_slight_glare','normal','slight'),
 'E':('strong_glare','unknown','strong'),
 'F':('bright_light','bright','none'),
 'G':('bright_light_glare','bright','present'),
 'U':('unsure','unknown','unknown'),
}
FIELDS=['sample_id','color','quality','option','condition','lighting','glare','notes','crop','masked','source']


def write_csv(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.csv')
    with temp.open('w',newline='') as f:
        extras=sorted(set().union(*(row.keys() for row in records.values()))-set(FIELDS)) if records else []
        writer=csv.DictWriter(f,fieldnames=FIELDS+extras);writer.writeheader()
        writer.writerows(records[k] for k in sorted(records))
    temp.replace(path)


def panel(img,w=400,h=400):
    canvas=np.full((h,w,3),230,np.uint8)
    s=min(w/img.shape[1],h/img.shape[0])
    resized=cv2.resize(img,(max(1,round(img.shape[1]*s)),max(1,round(img.shape[0]*s))))
    y=(h-resized.shape[0])//2;x=(w-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/brick_dataset_extracted')
    p.add_argument('--output',type=Path)
    p.add_argument('--review',action='store_true',help='Review existing suggested answers; Enter accepts current option')
    a=p.parse_args();root=a.dataset.resolve();output=a.output.resolve() if a.output else root/'lighting_annotations.csv'
    if not sys.stdin.isatty():p.error('Run this script in an interactive terminal')
    samples=[r for r in json.loads((root/'manifest.json').read_text()) if r.get('crop') and r.get('masked') and r['status']!='no_mask']
    samples.sort(key=lambda r:(r['quality'],r['color'],Path(r['source']).name))
    records={}
    if output.exists():
        with output.open(newline='') as f:records={r['sample_id']:r for r in csv.DictReader(f)}
    pending=[i for i,s in enumerate(samples) if s['id'] not in records]
    if not pending and not a.review:print('All crops annotated. CSV:',output);return
    index=0 if a.review else pending[0];last=None
    title='Lighting review | choose A-G or U in the TERMINAL'
    cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
    print('This ignores capture_groups.csv. Appearance descriptions do not establish capture sessions or data splits.')
    print('Type the option in this terminal and press Enter. Optional notes: C some reflection on studs')
    print('P previous | N skip for now | Q quit. Every answer autosaves; restart resumes.')
    try:
        while True:
            row=samples[index]
            if index!=last:
                crop=cv2.imread(str(root/row['crop']));masked=cv2.imread(str(root/row['masked']))
                if crop is None or masked is None:raise ValueError('Missing crop')
                canvas=np.full((490,800,3),245,np.uint8)
                canvas[60:460]=np.hstack([panel(crop),panel(masked)])
                cv2.putText(canvas,f"{index+1}/{len(samples)} | {row['quality']}/{row['color']}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
                cv2.putText(canvas,Path(row['source']).name,(10,45),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
                cv2.putText(canvas,'Original-color crop                     Background removed',(10,480),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
                cv2.imshow(title,canvas)
                print(f"\n[{index+1}/{len(samples)}] {row['quality']}/{row['color']} | {Path(row['source']).name}")
                print('A normal light        B low light\nC low + slight glare  D normal + slight glare\nE strong glare        F bright light (no glare)\nG bright + glare      U unsure')
                if row['id'] in records:print('Current answer:',records[row['id']]['condition'])
                print('Choice (Enter accepts existing answer; optional notes after letter): ',end='',flush=True);last=index
            key=cv2.waitKey(30)&255
            if key in [27,ord('q')] or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            readable,_,_=select.select([sys.stdin],[],[],0)
            if not readable:continue
            answer=sys.stdin.readline()
            if not answer:break
            parts=answer.strip().split(maxsplit=1)
            choice=parts[0].upper() if parts else ''
            previous=records.get(row['id'],{})
            if not parts and previous:
                choice=previous['option']
                parts=[choice,previous.get('notes','')]
            if choice=='Q':break
            if choice=='P':index=max(0,index-1);last=None;continue
            if choice=='N':
                if index+1<len(samples):index+=1;last=None
                else:print('Last crop; Q quits or P goes back.',flush=True)
                continue
            if choice not in OPTIONS:
                print('Choose A-G, U, P, N or Q: ',end='',flush=True);continue
            condition,lighting,glare=OPTIONS[choice]
            records[row['id']]={**previous,'sample_id':row['id'],'color':row['color'],'quality':row['quality'],
                'option':choice,'condition':condition,'lighting':lighting,'glare':glare,
                'notes':parts[1] if len(parts)>1 else '', 'crop':str((root/row['crop']).resolve()),
                'masked':str((root/row['masked']).resolve()),'source':row['source'],
                'annotation_source':'user_validated','review_status':'validated'}
            write_csv(output,records)
            print('Saved:',condition,flush=True)
            if index+1<len(samples):index+=1;last=None
            else:
                print('Reached last crop. Q quits or P goes back. Skipped crops appear on restart.',flush=True)
    finally:
        cv2.destroyAllWindows()
        print('\nCSV:',output,'| Saved:',len(records),flush=True)


if __name__=='__main__':main()
