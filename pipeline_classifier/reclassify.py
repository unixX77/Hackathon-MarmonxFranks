"""Apply the standalone classifier to normal-pipeline boxes without changing geometry."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'classifier'))
sys.path.insert(0,str(ROOT/'perception/sam3_detection'))
from classify_top_brick_quality import classify_result
from top_layer_bricks import final_view, save_result, titled


def reclassify(geometry_folder, output_root, erosion=.12, min_pixels=80, max_distance=35., min_margin=4.,
               colour_model=None):
    """Return final folder. Instance IDs, corners and 3D poses remain unchanged."""
    original=json.loads((geometry_folder/'bricks.json').read_text())
    recording=Path(original['recording']).resolve()
    frame=f"{int(original['frame']):06d}"
    bgr=cv2.imread(str(recording/'rgb'/f'{frame}.png'))
    instance=cv2.imread(str(geometry_folder/'final_instances.png'),-1)
    preview=cv2.imread(str(geometry_folder/'preview.png'))
    if bgr is None or instance is None or preview is None:
        raise ValueError('Missing normal-pipeline RGB, instance mask or preview')
    boxes=[]
    for b in original['bricks']:
        if not b.get('id'):continue
        corners=np.asarray(b['footprint_px'],float)
        edge=corners[0]-corners[1]
        boxes.append({'id':b['id'],'center_xy':corners.mean(axis=0).tolist(),
                      'corners_xy':corners.tolist(),
                      'size_wh':[float(np.linalg.norm(edge)),float(np.linalg.norm(corners[1]-corners[2]))],
                      'angle_deg':float(np.degrees(np.arctan2(edge[1],edge[0])))})
    adapter=output_root/'01_box_adapter'/'capture'/recording.name/frame
    adapter.mkdir(parents=True,exist_ok=True)
    top_boxes=adapter/'top_boxes.json'
    top_boxes.write_text(json.dumps({'strategy':'normal-pipeline-geometry','dataset':'capture',
        'recording':recording.name,'frame':int(frame),'rgb':str(recording/'rgb'/f'{frame}.png'),
        'geometry_source':str(geometry_folder.resolve()/'bricks.json'),'boxes':boxes},indent=2))
    args=SimpleNamespace(erosion_fraction=erosion,min_pixels=min_pixels,max_distance=max_distance,min_margin=min_margin,
                         colour_model=colour_model)
    quality,classified_folder=classify_result(top_boxes,output_root/'02_classification',args)
    predictions={b['id']:b for b in quality['boxes']}
    from segment_lego_sam3_single_prompt import NAMES
    updated=copy.deepcopy(original)
    updated['classification_method']='standalone '+quality['strategy']
    updated['geometry_source']=str(geometry_folder.resolve()/'bricks.json')
    updated['classification_source']=str((classified_folder/'quality_boxes.json').resolve())
    for brick in updated['bricks']:
        if not brick.get('id'):continue
        prediction=predictions[brick['id']]
        brick.update(colour=prediction['colour'],quality=prediction['quality'],
                     class_id=NAMES.index(prediction['colour'])+1,
                     colour_confidence=prediction['confidence'],
                     colour_distance=prediction['colour_distance'],
                     colour_margin=prediction['distance_margin'],
                     colour_runner_up=prediction['runner_up'],lab=prediction['lab_raw'])
        brick['classification']=prediction
    updated['colour']={'white_balance':quality['white_balance'],'scene_lab_offset':quality['scene_lab_offset'],
                       'scene_feature_transform':quality['scene_feature_transform']}
    final=[b for b in updated['bricks'] if b.get('id')]
    updated['good']=sum(b['quality']=='good' for b in final)
    updated['bad']=len(final)-updated['good']
    updated['low_confidence_colour']=sum(b['colour_confidence']=='low' for b in final)
    semantic=np.zeros(instance.shape,'uint8')
    for b in final:semantic[instance==b['id']]=b['class_id']
    overlay=final_view(bgr,instance,updated['bricks'])
    h,w=bgr.shape[:2]
    if preview.shape[:2]!=(h*2,w*3):raise ValueError('Expected normal six-panel preview')
    preview[h:2*h,2*w:3*w]=titled(overlay,f"6 standalone classifier: {updated['good']} good, {updated['bad']} bad")
    destination=output_root/'03_final'/'capture'/recording.name/frame
    save_result(destination,{'final_instances.png':instance,'final_semantic.png':semantic,
                            'overlay.png':overlay,'preview.png':preview},updated,final)
    # Geometry and instance masks must remain byte-for-byte/numerically identical.
    geometry_keys=['id','source_instance','status','height_m','centre_cam_m','box_centre_cam_m',
                   'box_size_m','box_rotation_cam','box_corners_cam_m','yaw_deg_table','footprint_px']
    originals={b['id']:b for b in original['bricks'] if b.get('id')}
    for b in final:
        assert all(b.get(k)==originals[b['id']].get(k) for k in geometry_keys), 'Geometry changed'
    with (geometry_folder/'poses.csv').open() as handle:
        old_rows=list(csv.DictReader(handle))
    with (destination/'poses.csv').open() as handle:
        new_rows=list(csv.DictReader(handle))
    assert [{k:v for k,v in row.items() if k not in ('colour','quality')} for row in old_rows] == [
        {k:v for k,v in row.items() if k not in ('colour','quality')} for row in new_rows], 'Pose CSV geometry changed'
    print(f"Standalone classification: {len(final)} boxes, {updated['good']} good / {updated['bad']} bad")
    print('Geometry preserved. Results:',destination)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry_folder',type=Path,help='Normal result folder containing bricks.json and final_instances.png')
    p.add_argument('--out',type=Path,default=ROOT/'data/pipeline_classifier_results')
    p.add_argument('--erosion-fraction',type=float,default=.12)
    p.add_argument('--min-pixels',type=int,default=80)
    p.add_argument('--max-distance',type=float,default=35.)
    p.add_argument('--min-margin',type=float,default=4.)
    p.add_argument('--colour-model',type=Path,help='Learned colour model JSON; omit for fixed Lab prototypes')
    a=p.parse_args()
    reclassify(a.geometry_folder.resolve(),a.out.resolve(),a.erosion_fraction,a.min_pixels,a.max_distance,a.min_margin,
               a.colour_model.resolve() if a.colour_model else None)


if __name__=='__main__':main()
