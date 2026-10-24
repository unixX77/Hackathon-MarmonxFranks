"""Capture aligned RGB-D frames from a D435 with settings exported from RealSense Viewer.

Loads the Viewer JSON (depth preset, laser, exposure and colour controls) into the camera,
then Enter saves a frame and q quits. Each run creates <output>/<timestamp>/ with
  rgb/000000.png     colour image (BGR)
  depth/000000.png   uint16 depth aligned to the colour image, multiply by depth_scale_m for metres
  meta/000000.json   frame numbers, timestamps, exposure
  camera_info.json   intrinsics, extrinsics, depth scale and the camera options actually applied
which is the layout perception/sam3_detection/top_layer_bricks.py reads directly:
  python perception/sam3_detection/top_layer_bricks.py <session folder> --frame 000000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from capture_rgbd import camera_info, frame_metadata, intrinsics_to_dict

ROOT = Path(__file__).resolve().parents[1]


def find_device(serial, timeout=15):
    """The RealSense with this serial (any when None), waiting while it reconnects."""
    end = time.monotonic() + timeout
    while True:
        for device in rs.context().query_devices():
            if serial in (None, device.get_info(rs.camera_info.serial_number)):
                return device
        if time.monotonic() > end:
            raise RuntimeError(f"No RealSense found{f' with serial {serial}' if serial else ''}")
        time.sleep(0.5)


def load_settings(serial, settings_text):
    """Load Viewer JSON into the camera (enabling advanced mode once if needed); return the serial."""
    device = find_device(serial)
    serial = device.get_info(rs.camera_info.serial_number)
    advanced = rs.rs400_advanced_mode(device)
    if not advanced.is_enabled():
        print("Enabling advanced mode, the camera restarts...")
        advanced.toggle_advanced_mode(True)
        time.sleep(3)
        advanced = rs.rs400_advanced_mode(find_device(serial))
    advanced.load_json(settings_text)
    return serial


def options_snapshot(device):
    """Every readable option per sensor, to record what the camera actually used."""
    snapshot = {}
    for sensor in device.query_sensors():
        values = {}
        for option in sensor.get_supported_options():
            try:
                values[str(option).split(".")[-1]] = sensor.get_option(option)
            except RuntimeError:
                pass
        snapshot[sensor.get_info(rs.camera_info.name)] = values
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", type=Path, default=ROOT / "d435_settings.json",
                        help="JSON exported from RealSense Viewer (default: d435_settings.json in the repo root)")
    parser.add_argument("--rgb-width", type=int, default=1920)
    parser.add_argument("--rgb-height", type=int, default=1080)
    parser.add_argument("--depth-width", type=int, help="Default: stream-width in the settings JSON")
    parser.add_argument("--depth-height", type=int, help="Default: stream-height in the settings JSON")
    parser.add_argument("--fps", type=int, help="Default: stream-fps in the settings JSON")
    parser.add_argument("--serial", help="Serial number of the camera to use (default: first found)")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "captures")
    args = parser.parse_args()

    settings_text = args.settings.read_text()
    viewer = json.loads(settings_text).get("viewer", {})
    depth_w = args.depth_width or int(viewer.get("stream-width", 848))
    depth_h = args.depth_height or int(viewer.get("stream-height", 480))
    fps = args.fps or int(viewer.get("stream-fps", 30))

    serial = load_settings(args.serial, settings_text)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.rgb_width, args.rgb_height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as error:
        raise RuntimeError(f"Cannot start {args.rgb_width}x{args.rgb_height} colour + {depth_w}x{depth_h} "
                           f"depth at {fps} fps ({error}). Check the USB 3 connection or the modes.") from error

    session = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    window = "D435 capture - Enter: save | q: quit"
    count = 0
    try:
        device = profile.get_device()
        depth_scale = device.first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        # Let auto exposure settle before the first saved frame.
        for _ in range(30):
            pipeline.wait_for_frames()

        info = camera_info(profile, depth_scale, None)
        info.update(settings_file=str(args.settings.resolve()), applied_options=options_snapshot(device),
                    fps=fps, post_processing="none; SDK alignment to colour only", depth_invalid_value=0,
                    camera_coordinate_convention="x right, y down, z forward; metres")
        session.mkdir(parents=True)
        (session / "d435_settings.json").write_text(settings_text)
        print(f"Depth scale {depth_scale} m, saving to {session.resolve()}")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 360)
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            if "aligned_depth_intrinsics" not in info:
                # Saved depth is on the colour grid, so these (not depth_intrinsics) describe it.
                info["aligned_depth_intrinsics"] = intrinsics_to_dict(
                    depth_frame.profile.as_video_stream_profile().get_intrinsics())
                (session / "camera_info.json").write_text(json.dumps(info, indent=2))

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=255.0 / (1.0 / depth_scale)), cv2.COLORMAP_JET)
            depth_vis[depth == 0] = 0
            preview = np.hstack([color, depth_vis])
            cv2.putText(preview, f"Enter: save | q: quit | saved: {count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13):
                name = f"{count:06d}"
                for folder in ("rgb", "depth", "meta"):
                    (session / folder).mkdir(exist_ok=True)
                if not (cv2.imwrite(str(session / "rgb" / f"{name}.png"), color)
                        and cv2.imwrite(str(session / "depth" / f"{name}.png"), depth)):
                    raise RuntimeError(f"Could not save frame {name}")
                (session / "meta" / f"{name}.json").write_text(json.dumps({
                    "color": frame_metadata(color_frame), "depth": frame_metadata(depth_frame),
                    "capture_time": datetime.now().isoformat()}, indent=2))
                print(f"Saved frame {name}")
                count += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
