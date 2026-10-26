"""Extract labeled single-brick crops with SAM3, without using the color classifier."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def write_image(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):raise RuntimeError(f'Cannot write {path}')


def crop_outputs(bgr,mask,padding=.12):
    ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError('Empty mask')
    pad=max(4,int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*padding))
    x0=max(0,int(xs.min())-pad);x1=min(bgr.shape[1],int(xs.max())+1+pad)
    y0=max(0,int(ys.min())-pad);y1=min(bgr.shape[0],int(ys.max())+1+pad)
    crop=bgr[y0:y1,x0:x1].copy();crop_mask=mask[y0:y1,x0:x1]
    masked=np.full_like(crop,127);masked[crop_mask]=crop[crop_mask]
    return crop,masked,crop_mask.astype('uint8')*255,[x0,y0,x1,y1]


def select_mask(masks,scores,shape):
    candidates=[]
    for i,mask in enumerate(masks):
        area=int(mask.sum())
        if area<100 or area>shape[0]*shape[1]*.15:continue
        contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        (_, _),(w,h),_=cv2.minAreaRect(contour)
        aspect=max(w,h)/max(min(w,h),1)
        rectangularity=min(1.,area/max(w*h,1))
        if not 1.2<=aspect<=4 or rectangularity<.55:continue
        candidates.append({'index':i,'score':float(scores[i]),'area_px':area,
            'aspect_ratio':float(aspect),'rectangularity':float(rectangularity),
            'rank':float(scores[i])*rectangularity})
    candidates.sort(key=lambda x:x['rank'],reverse=True)
    return candidates


def letterbox(img,width,height):
    canvas=np.full((height,width,3),230,'uint8')
    scale=min(width/img.shape[1],height/img.shape[0])
    resized=cv2.resize(img,(max(1,round(img.shape[1]*scale)),max(1,round(img.shape[0]*scale))))
    y=(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return canvas


def reviews(output,records):
    groups={}
    for row in records:groups.setdefault((row['quality'],row['color']),[]).append(row)
    for (quality,color),rows in groups.items():
        for page_start in range(0,len(rows),24):
            tiles=[]
            for index,row in enumerate(rows[page_start:page_start+24],page_start+1):
                bgr=cv2.imread(row['source']);thumb=letterbox(bgr,180,160)
                if row['status']!='no_mask':
                    mask=cv2.imread(str(output/row['mask']),-1)
                    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    overlay=bgr.copy();cv2.drawContours(overlay,contours,-1,(0,255,255),3)
                    thumb=letterbox(overlay,180,160)
                    crop=cv2.imread(str(output/row['masked']))
                    crop=letterbox(crop,120,160)
                else:crop=np.full((160,120,3),127,'uint8')
                tile=np.vstack([np.full((35,300,3),245,'uint8'),np.hstack([thumb,crop])])
                cv2.putText(tile,f"#{index:03d} {row['status']}",(5,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
                cv2.putText(tile,Path(row['source']).stem[-32:],(5,30),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
                tiles.append(tile)
            while len(tiles)%4:tiles.append(np.full((195,300,3),245,'uint8'))
            sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
            write_image(output/'review'/quality/color/f'page_{page_start//24+1:02d}.jpg',sheet)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/brick_dataset')
    p.add_argument('--output',type=Path,default=ROOT/'data/brick_dataset_extracted')
    p.add_argument('--threshold',type=float,default=.3)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--limit',type=int)
    a=p.parse_args();output=a.output.resolve();root=a.input.resolve()
    if output.exists() and not a.resume:p.error('Output exists; use --resume or a new --output')
    labels={'good':['red','dark_red','dark_orange'],'bad':['medium_nougat','reddish_brown']}
    files=[]
    for quality,colors in labels.items():
        for color in colors:
            files.extend((quality,color,path) for path in sorted((root/quality/color).glob('*'))
                         if path.suffix.lower() in ['.png','.jpg','.jpeg'])
    if a.limit:files=files[:a.limit]
    if not files:p.error('No labeled images found')
    output.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import Sam3Model,Sam3Processor
    sys.path.insert(0,str(ROOT/'perception/sam3_detection'))
    from segment_lego_sam3_single_prompt import run_sam3
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use sam3 environment')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model=Sam3Model.from_pretrained('facebook/sam3',torch_dtype=dtype).to('cuda').eval()
    processor=Sam3Processor.from_pretrained('facebook/sam3')
    for n,(quality,color,path) in enumerate(files,1):
        relative=Path(quality)/color/f'{path.stem}.png'
        meta_path=output/'metadata'/quality/color/f'{path.stem}.json'
        if a.resume and meta_path.is_file():continue
        bgr=cv2.imread(str(path))
        if bgr is None:raise ValueError(f'Cannot decode {path}')
        masks,scores=run_sam3(model,processor,Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),
                             'lego brick',a.threshold,.5,'cuda',dtype)
        candidates=select_mask(masks,scores,bgr.shape)
        row={'source':str(path.resolve()),'quality':quality,'color':color,'label_source':'parent folders',
             'id':str(relative.with_suffix('')),'status':'no_mask','candidates':candidates,
             'review_required':True}
        if candidates:
            mask=masks[candidates[0]['index']]
            crop,masked,crop_mask,bounds=crop_outputs(bgr,mask)
            status='auto_candidate'
            if len(candidates)>1:status='multiple_candidates'
            if candidates[0]['score']<.5:status='low_score'
            if mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any():status='touches_border'
            row.update(status=status,bbox_xyxy=bounds,crop=str(Path('crops')/relative),
                       masked=str(Path('masked')/relative),mask=str(Path('masks')/relative),
                       crop_mask=str(Path('crop_masks')/relative))
            for key,img in [('crop',crop),('masked',masked),('mask',mask.astype('uint8')*255),('crop_mask',crop_mask)]:
                write_image(output/row[key],img)
        meta_path.parent.mkdir(parents=True,exist_ok=True)
        meta_path.write_text(json.dumps(row,indent=2))
        print(f'[{n}/{len(files)}] {quality}/{color}/{path.name}: {row["status"]}',flush=True)
    records=[json.loads(path.read_text()) for path in sorted((output/'metadata').glob('*/*/*.json'))]
    (output/'manifest.json').write_text(json.dumps(records,indent=2))
    summary={'total':len(records),'statuses':dict(Counter(r['status'] for r in records)),
             'classes':dict(Counter(r['quality']+'/'+r['color'] for r in records)),
             'note':'Automatic masks/crops require human review before training. Originals unchanged.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    reviews(output,records)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
