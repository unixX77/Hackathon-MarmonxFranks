"""Correct ambiguous automatic brick extractions with four-point polygons."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from review_brick_boxes import polygon_crop
from extract_brick_dataset import reviews

ROOT=Path(__file__).resolve().parents[1]
AMBIGUOUS={'multiple_candidates','low_score','no_mask','touches_border'}
PRIORITY=['rgb_20261025_182017_273489.png','20261025_170633_000000.png','20261025_174137_000014.png']


def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def load_records(root):
    return [json.loads(p.read_text()) for p in sorted((root/'metadata').glob('*/*/*.json'))]


def refresh(root,changed_colors=(),verify=False):
    rows=load_records(root)
    atomic_json(root/'manifest.json',rows)
    atomic_json(root/'summary.json',{'total':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
        'classes':dict(Counter(r['quality']+'/'+r['color'] for r in rows)),
        'note':'Manual corrections preserve original class labels. Other automatic candidates still require visual review.'})
    if changed_colors:
        reviews(root,[r for r in rows if (r['quality'],r['color']) in changed_colors])
    if verify:
        errors=[];count=0
        for r in rows:
            if r['status']=='no_mask':continue
            images={key:cv2.imread(str(root/r[key]),cv2.IMREAD_UNCHANGED) for key in ['crop','masked','mask','crop_mask']}
            if any(v is None for v in images.values()):errors.append(r['source']);continue
            crop,masked,mask=images['crop'],images['masked'],images['crop_mask']
            valid=crop.shape==masked.shape and crop.shape[:2]==mask.shape
            if valid:valid=np.array_equal(crop[mask>0],masked[mask>0]) and np.all(masked[mask==0]==127)
            if valid:count+=1
            else:errors.append(r['source'])
        atomic_json(root/'verification.json',{'sources':len(rows),'verified_crops':count,
            'no_mask':sum(r['status']=='no_mask' for r in rows),'errors':errors,
            'checks':'Artifact existence, crop dimensions, preserved RGB foreground and neutral background',
            'note':'This checks file consistency; manual approval is recorded per sample.'})
    return rows


def save_correction(root,row,points,image,backup_root):
    crop,cmask,masked,bounds=polygon_crop(image,points)
    relative=Path(row['quality'])/row['color']/(Path(row['source']).stem+'.png')
    metadata_path=root/'metadata'/relative.with_suffix('.json')
    backup=backup_root/relative.with_suffix('')
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(metadata_path,backup/'metadata.json')
    for key in ['crop','masked','mask','crop_mask']:
        if row.get(key) and (root/row[key]).exists():shutil.copy2(root/row[key],backup/(key+'.png'))
    fullmask=np.zeros(image.shape[:2],np.uint8)
    cv2.fillConvexPoly(fullmask,np.asarray(points,np.int32),255)
    updated=dict(row,status='manual_corrected',review_required=False,
        original_status=row.get('original_status',row['status']),polygon_xy=[list(p) for p in points],
        bbox_xyxy=bounds,annotation='manual ordered quadrilateral; labels unchanged',
        corrected_at=datetime.now().isoformat(),backup=str(backup.relative_to(root)))
    for key,folder,img in [('crop','crops',crop),('masked','masked',masked),('mask','masks',fullmask),('crop_mask','crop_masks',cmask)]:
        relative_path=Path(folder)/relative;target=root/relative_path
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.stem+'.pending.png')
        if not cv2.imwrite(str(temp),img):raise RuntimeError(f'Cannot save {target}')
        temp.replace(target);updated[key]=str(relative_path)
    atomic_json(metadata_path,updated)
    return updated


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/brick_dataset_extracted')
    p.add_argument('--list',action='store_true',help='List queue without opening a window')
    a=p.parse_args();root=a.dataset.resolve();rows=load_records(root)
    queue=[r for r in rows if r['status'] in AMBIGUOUS or
           Path(r['source']).name in PRIORITY and r['status']!='manual_corrected']
    order={name:i for i,name in enumerate(PRIORITY)}
    queue.sort(key=lambda r:(order.get(Path(r['source']).name,len(order)),r['quality'],r['color'],r['source']))
    atomic_json(root/'correction_queue.json',[{'color':r['color'],'source':r['source'],'status':r['status']} for r in queue])
    print(f'{len(queue)} ambiguous extractions queued; folder labels stay unchanged.',flush=True)
    for i,r in enumerate(queue,1):print(f"{i:02d}: {r['quality']}/{r['color']} | {r['status']} | {Path(r['source']).name}",flush=True)
    if a.list or not queue:return
    session=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup_root=root/'manual_backups'/session
    window='Correct extraction | click 4 corners | Enter save | u/right undo | c clear | n/p next/prev | q quit'
    state={'points':[],'image':None,'scale':1.};index=0;changed=set();saved=0

    def load():
        image=cv2.imread(queue[index]['source'])
        if image is None:raise ValueError('Source image missing')
        state.update(image=image,scale=min(1.,1280/image.shape[1],720/image.shape[0]),points=[])

    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(state['points'])<4:
            image=state['image'];s=state['scale']
            state['points'].append((int(np.clip(round(x/s),0,image.shape[1]-1)),int(np.clip(round(y/s),0,image.shape[0]-1))))
        elif event==cv2.EVENT_RBUTTONDOWN and state['points']:state['points'].pop()

    load();cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(window,mouse)
    print('Click top-face corners in perimeter order. Enter replaces all extraction artifacts. q exits safely.',flush=True)
    try:
        while True:
            row=queue[index];image=state['image'];s=state['scale']
            view=cv2.resize(image,(round(image.shape[1]*s),round(image.shape[0]*s)))
            polygon=state['points']
            if polygon:
                pts=np.rint(np.array(polygon)*s).astype('int32')
                if len(pts)>1:cv2.polylines(view,[pts],len(pts)==4,(0,255,255),2)
                for i,(x,y) in enumerate(pts,1):
                    cv2.circle(view,(int(x),int(y)),4,(0,255,255),-1)
                    cv2.putText(view,str(i),(int(x)+5,int(y)-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.rectangle(view,(0,0),(view.shape[1],48),(0,0,0),-1)
            cv2.putText(view,f"{index+1}/{len(queue)} | {row['quality']}/{row['color']} | {row['status']}",(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(view,Path(row['source']).name,(8,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            cv2.imshow(window,view);key=cv2.waitKey(20)&255
            if key in [ord('q'),27] or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key==ord('c'):state['points']=[]
            if key==ord('u') and state['points']:state['points'].pop()
            move=0
            if key in [10,13]:
                try:
                    # Unique backup per approval, including repeated corrections.
                    updated=save_correction(root,row,polygon,image,backup_root/datetime.now().strftime('%H%M%S_%f'))
                except ValueError as error:print(error,flush=True);continue
                queue[index]=updated;changed.add((row['quality'],row['color']));saved+=1
                refresh(root,[(row['quality'],row['color'])])
                atomic_json(root/'verification.json',{'status':'pending_final_recheck','note':'Final artifact verification runs when reviewer closes normally.'})
                print('Corrected:',Path(row['source']).name,flush=True);move=1
            elif key in [ord('n'),ord('s')]:move=1
            elif key==ord('p'):move=-1
            if move:
                if index+move>=len(queue):
                    print('End of queue. Press q to finish, or p to go back.',flush=True)
                else:index=max(0,index+move);load()
    finally:
        cv2.destroyAllWindows()
        if changed:refresh(root,changed,verify=True)
        print(f'Saved {saved} corrections. Restart to continue remaining ambiguous cases.',flush=True)


if __name__=='__main__':main()
