"""Draft lighting appearance annotations for review; does not infer capture sessions."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from annotate_brick_lighting import ROOT,write_csv


def metrics(root,row):
    source=cv2.imread(row['source']);crop=cv2.imread(str(root/row['crop']))
    mask=cv2.imread(str(root/row['crop_mask']),0)>0
    # The neutral table is a better brightness reference than brick color.
    small=cv2.resize(source,(320,180));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    neutral=(hsv[...,1]<55)&(hsv[...,2]>15)&(hsv[...,2]<250)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    background=float(np.median(gray[neutral])) if neutral.sum()>500 else float(np.median(gray))
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(float)
    px=lab[mask];L=px[:,0];chroma=np.linalg.norm(px[:,1:]-128,axis=1)
    # Bright, desaturated regions relative to the brick interior suggest highlights.
    highlight=(L>max(150,float(np.median(L))+35))&(chroma<max(12,float(np.median(chroma))*.8))
    return {'background_brightness':round(background,2),'brick_median_luminance':round(float(np.median(L)),2),
            'highlight_fraction':round(float(highlight.mean()),4)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/brick_dataset_extracted')
    a=p.parse_args();root=a.dataset.resolve()
    originals={}
    original_path=root/'lighting_annotations.csv'
    if original_path.exists():
        with original_path.open(newline='') as f:originals={r['sample_id']:r for r in csv.DictReader(f)}
    samples=[r for r in json.loads((root/'manifest.json').read_text()) if r.get('crop')]
    measurements={r['id']:metrics(root,r) for r in samples}
    brightness=np.array([v['background_brightness'] for v in measurements.values()])
    low,high=np.percentile(brightness,[30,80])
    records={}
    for r in samples:
        m=measurements[r['id']];b=m['background_brightness'];g=m['highlight_fraction']
        lighting='low' if b<low else 'bright' if b>high else 'normal'
        glare='strong' if g>=.12 else 'slight' if g>=.015 else 'none'
        if glare=='strong':option='E';condition='strong_glare'
        elif lighting=='low':option='C' if glare!='none' else 'B';condition='low_light_slight_glare' if glare!='none' else 'low_light'
        elif lighting=='bright':option='G' if glare!='none' else 'F';condition='bright_light_glare' if glare!='none' else 'bright_light'
        else:option='D' if glare!='none' else 'A';condition='normal_light_slight_glare' if glare!='none' else 'normal_light'
        row={'sample_id':r['id'],'color':r['color'],'quality':r['quality'],
             'option':option,'condition':condition,'lighting':lighting,'glare':glare,
             'notes':'Automatic appearance estimate; validate visually. Not a capture-session label.',
             'crop':str((root/r['crop']).resolve()),'masked':str((root/r['masked']).resolve()),'source':r['source'],
             'annotation_source':'automatic_heuristic','review_status':'unverified',**m}
        if r['id'] in originals:
            row.update(originals[r['id']],annotation_source='existing_user_annotation',review_status='user_provided')
        records[r['id']]=row
    output=root/'lighting_suggestions.csv'
    if output.exists():raise RuntimeError(f'{output} already exists; preserve your review edits')
    write_csv(output,records)
    print(f'Created {output}: {len(records)} rows; preserved {len(originals)} existing user annotations')
    print('Automatic brightness labels are relative to this dataset; glare is a heuristic estimate.')


if __name__=='__main__':main()
