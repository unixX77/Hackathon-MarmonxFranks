"""Interactively draw brick boxes, approve original-color crops, or skip images."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def polygon_crop(image,points):
    polygon=np.asarray(points,dtype=np.int32)
    if polygon.shape!=(4,2) or len(np.unique(polygon,axis=0))!=4:
        raise ValueError('Select four different corners')
    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon))<25:
        raise ValueError('Click corners in clockwise/counterclockwise order, without crossing edges')
    if (polygon<0).any() or (polygon[:,0]>=image.shape[1]).any() or (polygon[:,1]>=image.shape[0]).any():
        raise ValueError('Corners must lie inside the image')
    x,y,w,h=cv2.boundingRect(polygon)
    crop=image[y:y+h,x:x+w].copy()
    mask=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(mask,polygon-[x,y],255)
    masked=np.full_like(crop,127);masked[mask>0]=crop[mask>0]
    return crop,mask,masked,[x,y,x+w,y+h]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/brick_dataset')
    p.add_argument('--out',type=Path,default=ROOT/'data/brick_dataset_manual')
    p.add_argument('--color',choices=['red','dark_red','dark_orange','medium_nougat','reddish_brown'])
    p.add_argument('--max-width',type=int,default=1280)
    p.add_argument('--mode',choices=['box','polygon'],default='box',help='Polygon: click four ordered corners')
    p.add_argument('--start',type=int,help='Start at this image number (1-based), including reviewed images')
    a=p.parse_args()
    if a.max_width<320:p.error('--max-width must be >=320')
    labels={'good':['red','dark_red','dark_orange'],'bad':['medium_nougat','reddish_brown']}
    items=[]
    for quality,colors in labels.items():
        for color in colors:
            if a.color and color!=a.color:continue
            for path in sorted((a.input/quality/color).glob('*')):
                if path.suffix.lower() in ('.png','.jpg','.jpeg'):items.append((quality,color,path))
    if not items:p.error('No source images found')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    annotations=out/'annotations.json'
    records=json.loads(annotations.read_text()) if annotations.exists() else {}
    index=next((i for i,(q,c,path) in enumerate(items) if f'{q}/{c}/{path.name}' not in records),0)
    if a.start is not None:
        if not 1<=a.start<=len(items):p.error('--start is outside the image list')
        index=a.start-1
    window='Brick annotation | Enter: save | u/right click: undo corner | c: clear | s: skip | n/p: navigate | q: quit'
    state={'start':None,'box':None,'drag':None,'scale':1.,'image':None,'points':[]}

    def point(x,y):
        image=state['image']
        return max(0,min(image.shape[1]-1,round(x/state['scale']))), max(0,min(image.shape[0]-1,round(y/state['scale'])))

    def mouse(event,x,y,flags,param):
        if state['image'] is None:return
        if a.mode=='polygon':
            if event==cv2.EVENT_LBUTTONDOWN and len(state['points'])<4:
                state['points'].append(point(x,y))
            elif event==cv2.EVENT_RBUTTONDOWN and state['points']:
                state['points'].pop()
            return
        if event==cv2.EVENT_LBUTTONDOWN:
            state['start']=point(x,y);state['drag']=state['start'];state['box']=None
        elif event==cv2.EVENT_MOUSEMOVE and state['start'] is not None:
            state['drag']=point(x,y)
        elif event==cv2.EVENT_LBUTTONUP and state['start'] is not None:
            x0,y0=state['start'];x1,y1=point(x,y)
            left,right=sorted([x0,x1]);top,bottom=sorted([y0,y1])
            state['box']=[left,top,right+1,bottom+1]
            state['start']=None;state['drag']=None

    def persist():
        temporary=out/'annotations.tmp.json'
        temporary.write_text(json.dumps(records,indent=2));temporary.replace(annotations)

    def load():
        quality,color,path=items[index]
        image=cv2.imread(str(path))
        if image is None:raise ValueError(f'Cannot read {path}')
        state.update(image=image,scale=min(1.,a.max_width/image.shape[1],760/image.shape[0]),
                     box=None,start=None,drag=None,points=[])
        record=records.get(f'{quality}/{color}/{path.name}',{})
        if record.get('status')=='approved':
            if a.mode=='polygon' and record.get('polygon_xy'):state['points']=list(map(tuple,record['polygon_xy']))
            elif a.mode=='box':state['box']=record['bbox_xyxy']

    cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(window,mouse)
    load()
    print('Click four corners around the brick in order.' if a.mode=='polygon' else 'Drag tightly around the entire brick.')
    print('Enter saves the original-resolution crop, polygon mask and neutral-background crop.')
    print('s skips, n/p navigates, c clears, q quits. Labels are inherited from source folders.')
    print('Annotations autosave. Relaunch resumes at the first unreviewed image.')
    try:
        while True:
            quality,color,path=items[index]
            image=state['image'];scale=state['scale']
            display=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
            box=state['box']
            if state['start'] is not None:
                x0,y0=state['start'];x1,y1=state['drag'];box=[min(x0,x1),min(y0,y1),max(x0,x1)+1,max(y0,y1)+1]
            if box:
                x0,y0,x1,y1=box
                cv2.rectangle(display,(round(x0*scale),round(y0*scale)),(round((x1-1)*scale),round((y1-1)*scale)),(0,255,255),2)
            if a.mode=='polygon' and state['points']:
                pts=np.rint(np.array(state['points'])*scale).astype('int32')
                if len(pts)>1:cv2.polylines(display,[pts],len(pts)==4,(0,255,255),2)
                for i,(x,y) in enumerate(pts,1):
                    cv2.circle(display,(int(x),int(y)),4,(0,255,255),-1)
                    cv2.putText(display,str(i),(int(x)+5,int(y)-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            key_name=f'{quality}/{color}/{path.name}'
            status=records.get(key_name,{}).get('status','unreviewed')
            cv2.rectangle(display,(0,0),(display.shape[1],48),(0,0,0),-1)
            cv2.putText(display,f'{index+1}/{len(items)} | {quality}/{color} | {status}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(display,path.name,(8,39),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            cv2.imshow(window,display)
            key=cv2.waitKey(20)&255
            if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key==ord('c'):state.update(box=None,points=[],start=None,drag=None)
            if key==ord('u') and state['points']:state['points'].pop()
            move=0
            if key in (10,13):
                if a.mode=='polygon':
                    points=state['points']
                else:
                    if not state['box']:
                        print('Draw a bounding box first.',flush=True);continue
                    x0,y0,x1,y1=state['box']
                    points=[(x0,y0),(x1-1,y0),(x1-1,y1-1),(x0,y1-1)]
                try:crop,mask,masked,bounds=polygon_crop(image,points)
                except ValueError as error:
                    print(error,flush=True);continue
                target=out/'crops'/quality/color/f'{path.stem}.png'
                target.parent.mkdir(parents=True,exist_ok=True)
                paths={'crop':target,'mask':out/'masks'/quality/color/target.name,
                       'masked':out/'masked'/quality/color/target.name}
                for name,data in [('crop',crop),('mask',mask),('masked',masked)]:
                    paths[name].parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(paths[name]),data):raise RuntimeError(f'Could not save {name}')
                records[key_name]={'status':'approved','quality':quality,'color':color,
                    'source':str(path.resolve()),'bbox_xyxy':bounds,'polygon_xy':[list(point) for point in points],
                    **{name:str(path.relative_to(out)) for name,path in paths.items()},
                    'annotation':'manual four-corner polygon' if a.mode=='polygon' else 'manual axis-aligned box',
                    'mask_coordinates':'crop-local; add bbox origin for source image coordinates'}
                persist();print('Approved:',target,flush=True);move=1
            elif key==ord('s'):
                previous=records.get(key_name,{})
                for artifact in ('crop','mask','masked'):
                    if previous.get(artifact) and (out/previous[artifact]).exists():
                        old=out/previous[artifact]
                        archived=out/'skipped_crops'/artifact/quality/color/f'{old.stem}_{time.time_ns()}.png'
                        archived.parent.mkdir(parents=True,exist_ok=True)
                        old.rename(archived)
                records[key_name]={'status':'skipped','quality':quality,'color':color,'source':str(path.resolve())}
                persist();move=1
            elif key==ord('n'):move=1
            elif key==ord('p'):move=-1
            if move:
                if index+move>=len(items):
                    print('Reached last image. Press p to go back or q to quit.',flush=True)
                else:
                    index=max(0,index+move);load()
    finally:cv2.destroyAllWindows()


if __name__=='__main__':main()
