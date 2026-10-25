"""Live D435: Enter detects, s saves, q quits. Reuse a Viewer recording's settings."""
import copy
import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

from top_layer_bricks import build_parser, process_frame, save_result

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'data_collection'))
from capture_rgbd import camera_info, frame_metadata, intrinsics_to_dict
import pyrealsense2 as rs


def normalize(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


# Imaging settings only: do not restore read-only telemetry, sync outputs, or HDR
# sequence registers. Those are reported so differences are visible.
RESTORE = {
    'visualpreset', 'enableautoexposure', 'enableautowhitebalance', 'exposure',
    'gain', 'whitebalance', 'laserpower', 'emitterenabled', 'emitteronoff',
    'emitteralwayson', 'depthunits', 'globaltimeenabled', 'backlightcompensation',
    'brightness', 'contrast', 'gamma', 'hue', 'saturation', 'sharpness',
    'powerlinefrequency', 'autoexposurepriority', 'autoexposurelimit',
    'autogainlimit', 'autoexposurelimittoggle', 'autogainlimittoggle',
}


def reference_settings(reference):
    info = json.loads((reference / 'camera_info.json').read_text())
    topics = json.loads((reference / 'recording_settings.json').read_text())
    modes, sensors = {}, {}
    for topic, events in topics.items():
        if not events:
            continue
        value = events[0]['value']  # match the beginning of the recording
        match = re.search(r'/sensor_(\d+)/(Depth|Color)_0/info$', topic)
        if match:
            fields = dict(x.split('=', 1) for x in value.split(';') if '=' in x)
            modes[match[2].lower()] = fields
        match = re.search(r'/sensor_(\d+)/option/([^/]+)/value$', topic)
        if match:
            sensors.setdefault(match[1], {})[normalize(match[2])] = float(value)
    if set(modes) != {'depth', 'color'}:
        raise ValueError('Reference must include both depth and color stream information')
    return info, topics, modes, sensors


def apply_settings(device, topics, settings, advanced_json=None):
    option_lookup = {normalize(name): getattr(rs.option, name) for name in dir(rs.option)
                     if not name.startswith('_')}
    by_name = {s.get_info(rs.camera_info.name): s for s in device.query_sensors()}
    log = []
    for ident, values in settings.items():
        events = topics.get(f'/device_0/sensor_{ident}/info', [])
        name = None
        if events:
            fields = dict(x.split('=', 1) for x in events[0]['value'].split(';') if '=' in x)
            name = fields.get('Name')
        sensor = by_name.get(name)
        if sensor is None:
            raise RuntimeError(f'Recorded sensor {ident} ({name}) is unavailable')
        # Preset first, then controls. Manual values are applied only when the
        # corresponding automatic control was disabled in the recording.
        order = sorted(values, key=lambda k: (0 if k == 'visualpreset' else
                                              1 if k.startswith('enableauto') else 2, k))
        for key in order:
            row = {'sensor': name, 'option': key, 'requested': values[key]}
            option = option_lookup.get(key)
            reason = None
            if key not in RESTORE:
                reason = 'not an imaging control (not restored)'
            elif key == 'visualpreset' and advanced_json:
                reason = 'using supplied advanced JSON instead'
            elif key in ('exposure', 'gain') and values.get('enableautoexposure', 0):
                reason = 'auto exposure enabled; recorded value was dynamic'
            elif key == 'whitebalance' and values.get('enableautowhitebalance', 0):
                reason = 'auto white balance enabled; recorded value was dynamic'
            elif option is None or not sensor.supports(option):
                reason = 'unsupported'
            elif sensor.is_option_read_only(option):
                reason = 'read only'
            if reason:
                row['skipped'] = reason
            else:
                try:
                    value = values[key]
                    limits = sensor.get_option_range(option)
                    if not limits.min <= value <= limits.max:
                        raise ValueError(f'outside [{limits.min}, {limits.max}]')
                    sensor.set_option(option, value)
                    row['actual'] = sensor.get_option(option)
                    if not np.isclose(row['actual'], value, rtol=1e-4, atol=1e-5):
                        row['error'] = 'read-back differs from requested value'
                except Exception as error:
                    row['error'] = str(error)
            log.append(row)
    errors = [r for r in log if 'error' in r or r.get('skipped') == 'unsupported']
    for row in errors:
        print('SETTING DIFFERENCE:', row, flush=True)
    return log


def sensor_snapshot(device):
    snapshot = {}
    for sensor in device.query_sensors():
        values = {}
        for option in sensor.get_supported_options():
            try:
                values[str(option)] = sensor.get_option(option)
            except RuntimeError as error:
                values[str(option)] = {'unavailable': str(error)}
        snapshot[sensor.get_info(rs.camera_info.name)] = values
    return snapshot


def capture_packet(frames, align, profile, log):
    color = frames.get_color_frame()
    native = frames.get_depth_frame()
    if not color or not native:
        return None
    aligned = align.process(frames).get_depth_frame()
    if not aligned:
        return None
    rgb = np.asanyarray(color.get_data()).copy()
    fmt = color.profile.format()
    if fmt == rs.format.rgb8:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif fmt != rs.format.bgr8:
        raise ValueError(f'Unsupported live color format {fmt}')
    if color.get_frame_timestamp_domain() != native.get_frame_timestamp_domain():
        return None
    delta = color.get_timestamp() - native.get_timestamp()
    if abs(delta) > 20:
        return None
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    calibration = camera_info(profile, scale, None)
    calibration['aligned_depth_intrinsics'] = intrinsics_to_dict(
        aligned.profile.as_video_stream_profile().get_intrinsics())
    calibration['depth_invalid_value'] = 0
    calibration['applied_settings'] = log
    calibration['coordinate_convention'] = 'x right, y down, z forward; metres'
    return {'rgb': rgb, 'depth': np.asanyarray(aligned.get_data()).copy(),
            'depth_native': np.asanyarray(native.get_data()).copy(), 'calibration': calibration,
            'metadata': {'color': frame_metadata(color), 'depth': frame_metadata(native),
                         'delta_ms': delta, 'capture_time': datetime.now().isoformat()},
            'captured_monotonic': time.monotonic()}


def combine_burst(packets):
    result = dict(packets[len(packets)//2])
    for key in ('depth', 'depth_native'):
        stack = np.stack([p[key] for p in packets]).astype(float)
        stack[stack == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result[key] = np.rint(np.nan_to_num(np.nanmedian(stack, axis=0))).astype(np.uint16)
    result['metadata'] = dict(result['metadata'], burst_frames=[p['metadata'] for p in packets],
                              depth_processing='median of valid depths; keep camera and scene stationary')
    return result


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--serial', help='Override the recorded camera serial')
    parser.add_argument('--advanced-json', type=Path, help='Optional advanced settings JSON exported by Viewer')
    parser.add_argument('--burst', type=int, default=1, help='Stationary burst median; 1 matches extracted raw depth')
    parser.add_argument('--allow-setting-differences', action='store_true',
                        help='Continue if a recorded imaging control cannot be restored; see settings_report.json')
    args = parser.parse_args()
    if args.no_show:
        parser.error('This interactive live tool requires a window; omit --no-show')
    if args.burst < 1:
        parser.error('--burst must be positive')
    args.block_height_mm = args.brick_size_mm[2]
    args.out = args.out if args.out != Path('data/top_layer_bricks') else Path('data/live_top_layer_bricks')
    info, topics, modes, settings = reference_settings(args.recording)
    print('Reference:', args.recording)
    print('Stream modes:', modes, flush=True)
    serial = args.serial or info['device']['serial_number']
    context = rs.context()
    connected = [d.get_info(rs.camera_info.serial_number) for d in context.query_devices()]
    if serial not in connected:
        raise RuntimeError(f'RealSense {serial} not found. Connect it, close Viewer, or use --serial. Connected: {connected}')
    import torch
    from transformers import Sam3Model, Sam3Processor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate the sam3 environment and check nvidia-smi.')
    device = 'cuda'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    for name, stream, intr in [('depth', rs.stream.depth, info['depth_intrinsics']),
                               ('color', rs.stream.color, info['color_intrinsics'])]:
        mode = modes[name]
        encoding = {'Z16': rs.format.z16, 'RGB8': rs.format.rgb8, 'BGR8': rs.format.bgr8}.get(mode['encoding'])
        if encoding is None:
            raise ValueError(f'Unsupported recorded encoding {mode}')
        config.enable_stream(stream, intr['width'], intr['height'], encoding, int(mode['fps']))
    started = False
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    result = None
    pending_packet = None
    burst = None
    live_window, result_window = 'Live D435 | Enter detect | s save | q quit', 'Detection result (captured snapshot)'
    session = args.out / datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
    try:
        # Resolve first to configure imaging controls before streaming.
        resolved = config.resolve(rs.pipeline_wrapper(pipeline))
        camera = resolved.get_device()
        if args.advanced_json:
            rs.rs400_advanced_mode(camera).load_json(args.advanced_json.read_text())
        log = apply_settings(camera, topics, settings, args.advanced_json)
        session.mkdir(parents=True, exist_ok=False)
        (session/'settings_report.json').write_text(json.dumps({
            'reference': str(args.recording), 'serial': serial, 'streams': modes, 'options': log,
            'advanced_json': str(args.advanced_json) if args.advanced_json else None,
            'note': 'Recorded filter names lack enable flags/parameters. No Viewer post-processing applied; matches offline extraction. Advanced custom preset values require exported JSON.'}, indent=2))
        if not args.allow_setting_differences and any('error' in row or row.get('skipped') == 'unsupported' for row in log):
            raise RuntimeError(f'Cannot match some recorded imaging settings. See {session / "settings_report.json"}. Use --allow-setting-differences only after reviewing them.')
        profile = pipeline.start(config)
        started = True
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipeline.wait_for_frames()
        cv2.namedWindow(live_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(live_window, 1000, 650)
        print('Enter: detect snapshot | s: save completed result | q: quit', flush=True)
        while True:
            frames = pipeline.wait_for_frames(5000)
            packet = capture_packet(frames, align, profile, log)
            if packet is None:
                continue
            if future is not None and future.done():
                try:
                    images, metadata, final = future.result()
                    result = (pending_packet, images, metadata, final)
                    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(result_window, 1440, 800)
                    cv2.imshow(result_window, images['preview.png'])
                    print(f'Detected {len(final)} bricks. Snapshot age: {time.monotonic()-pending_packet["captured_monotonic"]:.1f}s. Press s to save.', flush=True)
                except Exception as error:
                    print('Detection failed:', error, flush=True)
                future = None
            if burst is not None:
                burst.append(packet)
                if len(burst) >= args.burst:
                    pending_packet = combine_burst(burst) if args.burst > 1 else burst[0]
                    result = None
                    work_args = copy.copy(args)
                    work_args.frame = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    pending_packet['id'] = work_args.frame
                    pending_packet['settings_snapshot'] = sensor_snapshot(profile.get_device())
                    future = executor.submit(process_frame, pending_packet['rgb'], pending_packet['depth'],
                                             pending_packet['calibration'], model, processor, work_args, device, dtype)
                    burst = None
            view = packet['rgb'].copy()
            status = 'DETECTING - result is a snapshot' if future else f'Collecting burst {len(burst)}/{args.burst}' if burst is not None else 'Enter: detect | s: save result | q: quit'
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
            cv2.imshow(live_window, view)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty(live_window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13) and future is None and burst is None:
                burst = []
            if key == ord('s'):
                if result is None:
                    print('No completed result available to save.', flush=True)
                    continue
                saved, images, metadata, final = result
                output = session / saved['id']
                if output.exists():
                    print('This result was already saved:', output, flush=True)
                    continue
                output.mkdir()
                for folder, data in [('rgb', saved['rgb']), ('depth', saved['depth']), ('depth_native', saved['depth_native'])]:
                    (output/folder).mkdir()
                    if not cv2.imwrite(str(output/folder/'000000.png'), data):
                        raise RuntimeError(f'Cannot save {folder}')
                (output/'camera_info.json').write_text(json.dumps(saved['calibration'], indent=2))
                (output/'frame_metadata.json').write_text(json.dumps(saved['metadata'], indent=2))
                (output/'sensor_settings.json').write_text(json.dumps(saved['settings_snapshot'], indent=2))
                metadata.update(live_capture=saved['metadata'], reference_recording=str(args.recording))
                save_result(output, images, metadata, final)
                print('Saved:', output, flush=True)
    finally:
        if started:
            pipeline.stop()
        executor.shutdown(wait=True)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
