"""Extract RealSense SDK recordings into calibrated RGB-D datasets."""
import argparse
import csv
import json
import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from capture_rgbd import camera_info, frame_metadata, intrinsics_to_dict


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def recording_strings(source):
    """Preserve SDK string topics (settings, metadata, device information)."""
    result = {}
    with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as db:
        for name, timestamp, data in db.execute(
            "SELECT t.name,m.timestamp,m.data FROM messages m JOIN topics t "
            "ON t.id=m.topic_id WHERE t.type='std_msgs/msg/String' ORDER BY m.timestamp,m.id"
        ):
            # ROS2 CDR encapsulation, then a length-prefixed UTF-8 string.
            endian = '<' if data[1] & 1 else '>'
            size = struct.unpack_from(endian + 'I', data, 4)[0]
            value = data[8:8 + size].rstrip(b'\0').decode('utf-8')
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            result.setdefault(name, []).append({'record_timestamp_ns': timestamp, 'value': value})
    return result


def extract(source, destination, every, max_delta):
    if destination.exists():
        raise RuntimeError(f'Refusing to overwrite {destination}; choose a new --output directory.')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(source.resolve()), repeat_playback=False)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        info = camera_info(profile, scale, None)
        info.update({
            'source': str(source.resolve()),
            'depth_invalid_value': 0,
            'depth_encoding': 'uint16 PNG; depth_m = pixel * depth_scale_m',
            'depth_native_frame': 'depth optical frame',
            'depth_frame': 'aligned to color optical grid',
            'coordinate_convention': 'x right, y down, z forward; metres',
            'camera_pose': None,
            'post_processing': 'none; SDK alignment only',
        })
        for folder in ('rgb', 'depth', 'depth_native', 'meta'):
            (destination / folder).mkdir(parents=True, exist_ok=True)
        write_json(destination / 'recording_settings.json', recording_strings(source))
        align = rs.align(rs.stream.color)
        seen_color, seen_depth = set(), set()
        count = candidates = rejected = 0
        with (destination / 'frames.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'id', 'rgb', 'depth', 'depth_native', 'metadata',
                'color_timestamp_ms', 'depth_timestamp_ms', 'delta_ms',
                'depth_valid_fraction',
            ])
            writer.writeheader()
            while True:
                ok, frames = pipeline.try_wait_for_frames(2000)
                if not ok:
                    if playback.current_status() == rs.playback_status.stopped:
                        break
                    raise RuntimeError('Playback timed out before end of recording')
                color = frames.get_color_frame()
                native = frames.get_depth_frame()
                if not color or not native:
                    continue
                ck = (color.get_frame_number(), color.get_timestamp())
                dk = (native.get_frame_number(), native.get_timestamp())
                if ck in seen_color or dk in seen_depth:
                    continue
                if color.get_frame_timestamp_domain() != native.get_frame_timestamp_domain():
                    raise RuntimeError('RGB/depth timestamp domains differ; cannot safely pair frames')
                delta = color.get_timestamp() - native.get_timestamp()
                if abs(delta) > max_delta:
                    rejected += 1
                    continue
                seen_color.add(ck)
                seen_depth.add(dk)
                candidates += 1
                if (candidates - 1) % every:
                    continue
                aligned = align.process(frames).get_depth_frame()
                if not aligned:
                    raise RuntimeError('Depth alignment failed')
                if 'aligned_depth_intrinsics' not in info:
                    info['aligned_depth_intrinsics'] = intrinsics_to_dict(
                        aligned.profile.as_video_stream_profile().get_intrinsics())
                    write_json(destination / 'camera_info.json', info)
                pixels = np.asanyarray(color.get_data())
                fmt = color.profile.format()
                if fmt == rs.format.rgb8:
                    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                elif fmt != rs.format.bgr8:
                    raise RuntimeError(f'Unsupported color encoding: {fmt}')
                depth = np.asanyarray(aligned.get_data())
                native_pixels = np.asanyarray(native.get_data())
                if depth.dtype != np.uint16 or native_pixels.dtype != np.uint16:
                    raise RuntimeError('Expected Z16 depth frames')
                name = f'{count:06d}'
                paths = {folder: f'{folder}/{name}.png' for folder in ('rgb', 'depth', 'depth_native')}
                for folder, image in [('rgb', pixels), ('depth', depth), ('depth_native', native_pixels)]:
                    if not cv2.imwrite(str(destination / paths[folder]), image):
                        raise RuntimeError(f'Could not save {paths[folder]}')
                metadata = f'meta/{name}.json'
                valid = float(np.mean(depth > 0))
                write_json(destination / metadata, {
                    'color': frame_metadata(color), 'depth': frame_metadata(native),
                    'delta_ms': delta, 'depth_valid_fraction': valid,
                })
                writer.writerow(dict(id=name, **paths, metadata=metadata,
                                     color_timestamp_ms=color.get_timestamp(),
                                     depth_timestamp_ms=native.get_timestamp(), delta_ms=delta,
                                     depth_valid_fraction=valid))
                count += 1
        if not count:
            raise RuntimeError('No matching RGB-D pairs found')
        summary = {'source': str(source.resolve()), 'saved_pairs': count,
                   'unique_pairs_before_sampling': candidates, 'rejected_timestamp_pairs': rejected,
                   'every': every, 'max_delta_ms': max_delta}
        write_json(destination / 'summary.json', summary)
        print(f'{source}: {count} pairs -> {destination}', flush=True)
        return summary
    except Exception as error:
        if destination.exists():
            write_json(destination / 'INCOMPLETE.json', {'error': str(error)})
        raise
    finally:
        if started:
            pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='Recording files or folders of .db3 files')
    parser.add_argument('--output', type=Path, default=Path('data_collection/data/extracted'))
    parser.add_argument('--every', type=int, default=1, help='Save every Nth unique pair')
    parser.add_argument('--max-delta-ms', type=float, default=20, help='Maximum RGB/depth timestamp difference')
    args = parser.parse_args()
    if args.every < 1 or args.max_delta_ms < 0:
        parser.error('--every must be positive and --max-delta-ms nonnegative')
    sources = []
    for path in args.inputs:
        found = sorted(path.glob('*.db3')) if path.is_dir() else [path]
        if not found or any(not p.is_file() for p in found):
            parser.error(f'No recordings found at {path}')
        sources.extend(found)
    destinations = [args.output / p.parent.name / p.stem for p in sources]
    if len(set(destinations)) != len(destinations) or any(p.exists() for p in destinations):
        parser.error('Duplicate or existing output folders; choose a fresh --output')
    for source, destination in zip(sources, destinations):
        extract(source, destination, args.every, args.max_delta_ms)


if __name__ == '__main__':
    main()
