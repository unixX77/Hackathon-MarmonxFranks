#!/usr/bin/env python3
"""Capture one D435 frame and run the real detector, without ROS or robot motion."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'src/lego_sorter/config/sorter.yaml')
    opts = parser.parse_args()
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    import torch
    import yaml
    from transformers import Sam3Model, Sam3Processor

    config = yaml.safe_load(opts.config.read_text())
    model_path = Path(config['model_path']).expanduser()
    if not config['model_path'] or not model_path.is_dir():
        raise ValueError('Configure model_path with the downloaded local model directory')
    output = root / 'runtime' / ('perception-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)
    pipeline = rs.pipeline()
    streams = rs.config()
    if config.get('camera_serial'):
        streams.enable_device(config['camera_serial'])
    streams.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    streams.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(streams)
    try:
        if config.get('camera_settings'):
            settings = json.loads(Path(config['camera_settings']).read_text())
            rs.rs400_advanced_mode(profile.get_device()).load_json(json.dumps(settings))
        align = rs.align(rs.stream.color)
        # Allow exposure and depth acquisition to settle after each stream restart.
        warmup_end = time.monotonic() + 3.0
        while True:
            frames = pipeline.wait_for_frames(5000)
            if time.monotonic() >= warmup_end:
                break
        frames = align.process(frames)
        rgb, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if not rgb or not depth_frame:
            raise RuntimeError('Missing colour or aligned depth frame')
        bgr = np.asanyarray(rgb.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        info = dict(aligned_depth_intrinsics=dict(
            width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
            ppx=intr.ppx, ppy=intr.ppy, distortion_coeffs=list(intr.coeffs),
            K=[[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]),
            depth_scale_m=profile.get_device().first_depth_sensor().get_depth_scale())
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
    finally:
        pipeline.stop()
    print(f'Camera serial: {serial}', flush=True)
    print(f'Capture saved to: {output}', flush=True)
    for name, array in [('rgb.png', bgr), ('depth.png', depth)]:
        if not cv2.imwrite(str(output / name), array):
            raise RuntimeError(f'Could not save {name}')
    (output / 'camera_info.json').write_text(json.dumps(info, indent=2))
    usable = (depth > 0) & (depth < np.iinfo(np.uint16).max)
    quality = dict(usable_depth_fraction=float(usable.mean()),
                   zero_depth_fraction=float((depth == 0).mean()),
                   max_raw_depth_fraction=float((depth == np.iinfo(np.uint16).max).mean()))
    (output / 'capture_quality.json').write_text(json.dumps(quality, indent=2))
    print(f'Usable depth across image: {100 * quality["usable_depth_fraction"]:.1f}%; '
          f'zero: {100 * quality["zero_depth_fraction"]:.1f}%; '
          f'max raw value: {100 * quality["max_raw_depth_fraction"]:.1f}%', flush=True)
    # This standalone test expects visible table surrounding the bricks. This is
    # an early capture sanity check, not the detector's per-mask acceptance rule.
    if quality['usable_depth_fraction'] < .10 or quality['max_raw_depth_fraction'] > .50:
        raise RuntimeError('Depth capture is mostly missing or saturated. Inspect the camera/depth stream '
                           'before changing SAM3 thresholds. Captured inputs remain in the printed directory.')

    sys.path.insert(0, str(root / 'src/lego_sorter/lego_sorter/vendor'))
    from top_layer_bricks import build_parser, process_frame, save_result
    args = build_parser().parse_args([str(output), '--no-show', '--brick-size-mm',
        *[str(x * 1000) for x in config['block_size']], '--top-min-support', '1'])
    args.block_height_mm = args.brick_size_mm[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    print(f'Loading SAM3 on {device}; running detector...', flush=True)
    processor = Sam3Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Sam3Model.from_pretrained(str(model_path), dtype=dtype, local_files_only=True).to(device).eval()
    images, metadata, final = process_frame(bgr, depth, info, model, processor, args, device, dtype)
    metadata['coordinate_frame'] = 'camera_color_optical_frame'
    metadata['purpose'] = 'Perception inspection only; no robot-base transform or motion'
    save_result(output, images, metadata, final)
    print(f'Detector returned {len(final)} fitted candidates. Inspect statuses in bricks.json.')
    print(f'Preview: {output / "preview.png"}')
    print('Coordinates are camera-relative. This does not validate calibration or robot motion.')


if __name__ == '__main__':
    main()
