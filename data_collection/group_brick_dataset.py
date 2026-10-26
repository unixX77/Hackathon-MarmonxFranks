"""Review crops and assign capture groups/splits without moving dataset images."""
import argparse
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPLITS=('unassigned','train','val','test')
FIELDS=('sample_id','color','quality','group','split','crop','masked','source','extraction_status')


def save_assignment(assignments,sample,group,split,root,output):
    group=group.strip()
    if not group or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in group):
        raise ValueError('Use letters, numbers, underscores or hyphens for group names')
    if split not in SPLITS:raise ValueError('Unknown split')
    # Splits belong to groups, never to individual neighboring frames.
    updated={k:dict(v) for k,v in assignments.items()}
    for row in updated.values():
        if row['group']==group:row['split']=split
    updated[sample['id']]={'sample_id':sample['id'],'color':sample['color'],'quality':sample['quality'],
        'group':group,'split':split,'crop':str((root/sample['crop']).resolve()),
        'masked':str((root/sample['masked']).resolve()),'source':sample['source'],
        'extraction_status':sample['status']}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp.csv')
    with temporary.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader()
        writer.writerows(updated[k] for k in sorted(updated))
    temporary.replace(output)
    return updated


def letterbox(image,w,h):
    result=np.full((h,w,3),220,np.uint8)
    scale=min(w/image.shape[1],h/image.shape[0])
    resized=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
    y=(h-resized.shape[0])//2;x=(w-resized.shape[1])//2
    result[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/brick_dataset_extracted')
    p.add_argument('--output',type=Path,help='Default: DATASET/capture_groups.csv')
    p.add_argument('--color',choices=['red','dark_red','dark_orange','medium_nougat','reddish_brown'])
    p.add_argument('--start',type=int,help='1-based index, including already assigned images')
    a=p.parse_args();root=a.dataset.resolve();output=a.output.resolve() if a.output else root/'capture_groups.csv'
    all_samples=json.loads((root/'manifest.json').read_text())
    samples=[r for r in all_samples if r.get('crop') and r.get('masked') and r['status']!='no_mask'
             and (not a.color or r['color']==a.color)]
    samples.sort(key=lambda r:(r['quality'],r['color'],Path(r['source']).name))
    if not samples:p.error('No extracted crops found')
    assignments={}
    if output.exists():
        with output.open(newline='') as handle:
            assignments={row['sample_id']:row for row in csv.DictReader(handle)}
    if a.start is not None and not 1<=a.start<=len(samples):p.error('--start is outside this image list')
    pending=[i for i,r in enumerate(samples) if r['id'] not in assignments]
    if not pending and a.start is None:
        print('All crops have group assignments. Use --start 1 to review them. CSV:',output);return
    index=a.start-1 if a.start is not None else pending[0]
    state={'group':'','split':'unassigned','editing':True,'buffer':'','message':''}
    window='Capture groups | g: type group | Tab: split | Enter: save/next | p/n: navigate | q: quit'

    def load():
        sample=samples[index]
        existing=assignments.get(sample['id'])
        if existing:state.update(group=existing['group'],split=existing['split'])
        state.update(editing=not bool(state['group']),buffer='',message='')
        imgs=[cv2.imread(str(root/sample[key])) for key in ['crop','masked']]
        source=cv2.imread(sample['source'])
        if source is None or any(img is None for img in imgs):raise ValueError('Missing crop or source image')
        state['panels']=np.hstack([letterbox(source,560,400),letterbox(imgs[0],270,400),letterbox(imgs[1],270,400)])

    def finish_edit():
        name=state['buffer'].strip()
        if not name:state['message']='Group cannot be empty';return
        state.update(group=name,editing=False)
        existing=next((row for row in assignments.values() if row['group']==name),None)
        if existing:state['split']=existing['split']

    load();cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
    print('Type a group (e.g. red_low_light_session1), Enter to finish, Enter again to save/next.')
    print('g: new group; Tab: unassigned/train/val/test; p/n: navigate; q: quit.')
    print('Group persists across images. Split changes apply to EVERY image with that group.')
    print('You can leave split unassigned and decide group-level splits later. CSV:',output,flush=True)
    try:
        while True:
            sample=samples[index]
            canvas=np.full((640,1100,3),245,np.uint8);canvas[120:520]=state['panels']
            text=[f"{index+1}/{len(samples)} | {sample['quality']}/{sample['color']} | {sample['status']}",
                  Path(sample['source']).name,'Source context                                      Original crop          Masked crop']
            for j,line in enumerate(text):cv2.putText(canvas,line,(12,25+j*28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
            group=state['buffer']+'_' if state['editing'] else state['group']
            cv2.putText(canvas,'Group: '+group,(12,550),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,100,0),2)
            cv2.putText(canvas,'Split: '+state['split']+' (Tab cycles; applies to whole group)',(12,579),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
            hint='TYPE GROUP NAME; Enter finishes typing' if state['editing'] else 'Enter save+next | g new group | p/n prev/next | q quit'
            cv2.putText(canvas,state['message'] or hint,(12,615),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,180),1)
            cv2.imshow(window,canvas);key=cv2.waitKey(30)&255
            if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if state['editing']:
                if key in (10,13):finish_edit()
                elif key in (8,127):state['buffer']=state['buffer'][:-1]
                elif key==27:state['editing']=False
                elif key!=255 and chr(key) in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-':state['buffer']+=chr(key)
                continue
            if key in (ord('q'),27):break
            if key==ord('g'):state.update(editing=True,buffer='',message='');continue
            if key==9:state['split']=SPLITS[(SPLITS.index(state['split'])+1)%len(SPLITS)];continue
            move=0
            if key in (10,13):
                try:assignments=save_assignment(assignments,sample,state['group'],state['split'],root,output)
                except ValueError as error:state['message']=str(error);continue
                print(f"Saved {sample['id']} -> {state['group']} / {state['split']}",flush=True);move=1
            elif key==ord('p'):move=-1
            elif key==ord('n'):move=1
            if move:
                if index+move>=len(samples):state['message']='Reached final crop. q quits; p goes back.'
                else:index=max(0,index+move);load()
    finally:
        cv2.destroyAllWindows()
        print('Saved assignments:',len(assignments),'CSV:',output,flush=True)


if __name__=='__main__':main()
