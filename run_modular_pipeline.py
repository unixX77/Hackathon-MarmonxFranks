"""Run separate capture, top-plane segmentation, quality and camera-pose stages."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print('\nRunning:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, help='Existing recording; omit to capture from D435')
    parser.add_argument('--frame', type=int, help='Frame number; default: last saved frame')
    parser.add_argument('--settings', type=Path, default=ROOT / 'd435_settings.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/modular_pipeline_results')
    parser.add_argument('--camera-pose', type=Path, help='Optional calibrated T_base_camera JSON')
    parser.add_argument('--temporal-frames', type=int, default=1,
                        help='Use >1 only for consecutive frames of an unchanged scene')
    parser.add_argument('--segmentation-args', nargs=argparse.REMAINDER, default=[],
                        help='Additional segmentation options; must come last')
    args = parser.parse_args()
    if args.temporal_frames < 1 or (args.frame is not None and args.frame < 0):
        parser.error('Temporal frames must be positive and frame nonnegative')
    # Check paths before switching working directory or starting capture.
    recording = args.recording.resolve() if args.recording else None
    settings = args.settings.resolve()
    camera_pose = args.camera_pose.resolve() if args.camera_pose else None
    output = args.out.resolve() / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    if recording and not (recording / 'camera_info.json').is_file():
        parser.error('Recording must contain camera_info.json, rgb/ and depth/')
    if not recording and not settings.is_file():
        parser.error(f'Settings file not found: {settings}')
    if camera_pose and not camera_pose.is_file():
        parser.error(f'Camera-pose file not found: {camera_pose}')
    import os
    os.chdir(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    try:
        inputs = output / 'inputs' / 'capture'
        inputs.mkdir(parents=True)
        if recording is None:
            run('data_collection/capture_d435.py', '--settings', settings, '--output', inputs)
            sessions = sorted(p for p in inputs.iterdir() if p.is_dir())
            if len(sessions) != 1:
                raise RuntimeError('Expected one new capture session; no previous captures are reused')
            recording = sessions[0]
        else:
            (inputs / recording.name).symlink_to(recording, target_is_directory=True)
        frames = sorted(p for p in (recording / 'rgb').glob('*.png') if p.stem.isdigit())
        if not frames:
            raise RuntimeError('No frames saved. Press Enter in capture before quitting.')
        frame = args.frame if args.frame is not None else max(int(p.stem) for p in frames)
        for path in [recording / 'rgb' / f'{frame:06d}.png', recording / 'depth' / f'{frame:06d}.png']:
            if not path.is_file():
                raise FileNotFoundError(path)
        info = json.loads((recording / 'camera_info.json').read_text())
        if not info.get('depth_aligned_to_color') or 'aligned_depth_intrinsics' not in info:
            raise ValueError('Aligned depth and aligned_depth_intrinsics are required')
        segmentation = output / '01_segmentation'
        classification = output / '02_classification'
        run('segmentation_sam3_bounding_box/segment_lego_rgbd_topplane.py', output / 'inputs',
            '--datasets', 'capture', '--recordings', recording.name, '--frame', frame,
            '--temporal-frames', args.temporal_frames, '--out', segmentation, *args.segmentation_args)
        top_boxes = segmentation / 'capture' / recording.name / f'{frame:06d}' / 'top_boxes.json'
        if not top_boxes.is_file():
            raise RuntimeError('Segmentation produced no frame output')
        run('classifier/classify_top_brick_quality.py', segmentation, '--out', classification)
        pose_args = ['--camera-pose', camera_pose] if camera_pose else []
        run('pose_info/export_camera_boxes.py', classification, *pose_args)
        result = classification / 'capture' / recording.name / f'{frame:06d}'
        boxes = json.loads((result / 'camera_boxes.json').read_text())['bricks']
        with (result / 'centers_camera_m.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'colour', 'quality', 'x_m', 'y_m', 'z_m'])
            for box in boxes:
                center = box['box_center_camera_m']
                writer.writerow([box['id'], box['colour'], box['quality'], *center])
                print(f"#{box['id']} {box['colour']} {box['quality']} box center (m): {center}")
        manifest = {'recording': str(recording), 'frame': frame, 'segmentation': str(top_boxes),
                    'classification': str(result / 'quality_boxes.json'),
                    'camera_boxes': str(result / 'camera_boxes.json'),
                    'poses': str(result / 'brick_poses.json'), 'box_count': len(boxes),
                    'temporal_frames': args.temporal_frames, 'camera_pose': str(camera_pose) if camera_pose else None}
        (output / 'pipeline.json').write_text(json.dumps(manifest, indent=2))
        print('\nResults:', output)
        print('Camera optical frame: x right, y down, z forward; metres.')
    except Exception as error:
        (output / 'INCOMPLETE.json').write_text(json.dumps({'error': str(error)}, indent=2))
        raise


if __name__ == '__main__':
    main()
