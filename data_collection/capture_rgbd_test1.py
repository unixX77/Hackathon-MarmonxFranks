"""Interactively save RGB-D frames and camera calibration from an Intel RealSense (tuned for D435 depth quality)."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def intrinsics_to_dict(intr):
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "K": [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        "distortion_model": str(intr.model).split(".")[-1],
        "distortion_coeffs": list(intr.coeffs),
    }


def extrinsics_to_dict(extr):
    # librealsense stores the rotation column-major.
    rotation = np.asarray(extr.rotation).reshape(3, 3).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = extr.translation
    return {
        "rotation": rotation.tolist(),
        "translation_m": list(extr.translation),
        "T": transform.tolist(),
    }


def device_info(device):
    info = {}
    for key, field in [("name", rs.camera_info.name),
                       ("serial_number", rs.camera_info.serial_number),
                       ("firmware_version", rs.camera_info.firmware_version),
                       ("product_line", rs.camera_info.product_line),
                       ("usb_type", rs.camera_info.usb_type_descriptor)]:
        if device.supports(field):
            info[key] = device.get_info(field)
    return info


def camera_info(profile, depth_scale, camera_pose):
    color = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    return {
        "device": device_info(profile.get_device()),
        "depth_scale_m": depth_scale,
        "depth_aligned_to_color": True,
        "color_intrinsics": intrinsics_to_dict(color.get_intrinsics()),
        "depth_intrinsics": intrinsics_to_dict(depth.get_intrinsics()),
        "depth_to_color_extrinsics": extrinsics_to_dict(depth.get_extrinsics_to(color)),
        "color_to_depth_extrinsics": extrinsics_to_dict(color.get_extrinsics_to(depth)),
        "camera_pose": camera_pose,
    }


def frame_metadata(frame):
    meta = {
        "frame_number": frame.get_frame_number(),
        "timestamp_ms": frame.get_timestamp(),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()).split(".")[-1],
    }
    for name, field in [("exposure", rs.frame_metadata_value.actual_exposure),
                        ("gain", rs.frame_metadata_value.gain_level),
                        ("white_balance", rs.frame_metadata_value.white_balance)]:
        if frame.supports_frame_metadata(field):
            meta[name] = frame.get_frame_metadata(field)
    return meta


def make_filters():
    """Intel's recommended chain: filter in the disparity domain, edge-preserving, no hole filling."""
    to_disparity = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 0)  # never invent depth
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    to_depth = rs.disparity_transform(False)
    return [to_disparity, spatial, temporal, to_depth]


def capture(pipeline, align, use_filters, n_frames):
    """Grab n_frames, run them through a fresh filter chain, and return the last result.

    A fresh chain per capture means the temporal filter only averages frames of the
    current (static) view, never stale frames from while the camera was moving.
    """
    filters = make_filters() if use_filters else []
    frames = raw_depth = None
    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        raw_depth = frames.get_depth_frame()
        for f in filters:
            frames = f.process(frames).as_frameset()
    native_depth = frames.get_depth_frame()
    aligned = align.process(frames)
    return aligned.get_color_frame(), aligned.get_depth_frame(), native_depth, raw_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=None, help="Override both stream widths")
    parser.add_argument("--height", type=int, default=None, help="Override both stream heights")
    parser.add_argument("--rgb-width", type=int, default=1920)
    parser.add_argument("--rgb-height", type=int, default=1080)
    parser.add_argument("--depth-width", type=int, default=848,
                        help="848x480 is Intel's recommended depth mode for the D435")
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--laser-power", type=float,
                        help="Projector power in device units (default: camera maximum)")
    parser.add_argument("--depth-exposure", type=float, help="Manual depth exposure in microseconds")
    parser.add_argument("--depth-gain", type=float, help="Manual depth gain in device units")
    parser.add_argument("--depth-units", type=float, default=0.0001,
                        help="Metres per depth unit (default 0.0001 = 0.1 mm, max range ~6.5 m)")
    parser.add_argument("--capture-frames", type=int, default=15,
                        help="Frames fed through the temporal filter per saved capture")
    parser.add_argument("--no-filter", action="store_true",
                        help="Save a single unfiltered frame instead")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--serial", help="Serial number of the camera to use (default: first found)")
    parser.add_argument(
        "--camera-pose", type=Path,
        help="JSON file with the camera pose in the robot/world frame, copied into camera_info.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory (default: data beside this script)",
    )
    args = parser.parse_args()
    if args.capture_frames < 1:
        raise RuntimeError("--capture-frames must be at least 1")
    camera_pose = json.loads(args.camera_pose.read_text()) if args.camera_pose else None
    use_filters = not args.no_filter
    n_frames = args.capture_frames if use_filters else 1

    session = args.output / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width or args.rgb_width,
                         args.height or args.rgb_height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width or args.depth_width,
                         args.height or args.depth_height, rs.format.z16, args.fps)
    window = "RGB-D capture - Enter: save | q: quit"
    count = 0
    started = False
    try:
        try:
            profile = pipeline.start(config)
        except RuntimeError as error:
            raise RuntimeError(
                f"Cannot start the RealSense ({error}). Check the USB 3 connection, "
                "or try a supported mode such as --width 640 --height 480."
            ) from error
        started = True
        sensor = profile.get_device().first_depth_sensor()
        settings = {}

        def set_option(option, value, required=False):
            name = str(option).split(".")[-1]
            if not sensor.supports(option):
                if required:
                    raise RuntimeError(f"Camera does not support {name}")
                print(f"Skipping unsupported setting: {name}")
                return
            limits = sensor.get_option_range(option)
            if not limits.min <= value <= limits.max:
                raise RuntimeError(f"{name} must be between {limits.min} and {limits.max}")
            sensor.set_option(option, value)
            settings[name] = sensor.get_option(option)

        # Preset first: it resets laser power, exposure etc., so overrides come after it.
        # Preset IDs vary by device, so find High Accuracy by its description.
        if sensor.supports(rs.option.visual_preset):
            limits = sensor.get_option_range(rs.option.visual_preset)
            for preset in range(int(limits.min), int(limits.max) + 1):
                description = sensor.get_option_value_description(rs.option.visual_preset, preset)
                if "high accuracy" in description.lower():
                    set_option(rs.option.visual_preset, preset)
                    settings["preset_name"] = description
                    break
            else:
                print("High Accuracy preset unavailable on this camera.")

        manual_exposure = args.depth_exposure is not None or args.depth_gain is not None
        set_option(rs.option.enable_auto_exposure, 0 if manual_exposure else 1)
        set_option(rs.option.emitter_enabled, 1)

        power = args.laser_power
        if power is None and sensor.supports(rs.option.laser_power):
            power = sensor.get_option_range(rs.option.laser_power).max
        if power is not None:
            set_option(rs.option.laser_power, power, required=args.laser_power is not None)

        if args.depth_exposure is not None:
            set_option(rs.option.exposure, args.depth_exposure, required=True)
        if args.depth_gain is not None:
            set_option(rs.option.gain, args.depth_gain, required=True)
        set_option(rs.option.depth_units, args.depth_units)

        depth_scale = sensor.get_depth_scale()
        print("Depth settings:", settings)
        align = rs.align(rs.stream.color)
        # Let auto-exposure settle before the first saved frame.
        for _ in range(60):
            pipeline.wait_for_frames()

        info = camera_info(profile, depth_scale, camera_pose)
        info["depth_settings"] = settings
        info["post_processing"] = (
            f"disparity -> spatial(mag 2, alpha 0.5, delta 20, no hole fill) -> "
            f"temporal(alpha 0.4, delta 20) over {n_frames} frames -> depth; filtered before alignment"
            if use_filters else "none"
        )
        info["fps"] = args.fps
        session.mkdir(parents=True, exist_ok=False)

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 480)
        print(f"Saving to {session.resolve()}")
        print("Focus the preview window. Hold the camera still and press Enter to save; q to quit.")
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            if "aligned_depth_intrinsics" not in info:
                # depth_intrinsics describes the native sensor grid (depth_native/ images);
                # aligned depth (depth/ images) uses the color grid.
                info["aligned_depth_intrinsics"] = intrinsics_to_dict(
                    depth_frame.profile.as_video_stream_profile().get_intrinsics())
                info["depth_invalid_value"] = 0
                info["camera_coordinate_convention"] = "x right, y down, z forward; metres"
                (session / "camera_info.json").write_text(json.dumps(info, indent=2))

            # Preview shows raw (unfiltered) depth, 0-2 m.
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=255.0 / (2.0 / depth_scale)), cv2.COLORMAP_JET)
            depth_vis[depth == 0] = 0
            preview = np.hstack([color, depth_vis])
            cv2.putText(preview, f"Enter: save | q: quit | Saved: {count}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13):
                color_frame, depth_frame, native_frame, raw_depth_frame = capture(
                    pipeline, align, use_filters, n_frames)
                if not color_frame or not depth_frame:
                    print("Capture failed, try again.")
                    continue
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                paths = {
                    "rgb": session / "rgb" / f"rgb_{stamp}.png",
                    "depth": session / "depth" / f"depth_{stamp}.png",
                    "depth_native": session / "depth_native" / f"depth_native_{stamp}.png",
                    "meta": session / "meta" / f"meta_{stamp}.json",
                }
                for path in paths.values():
                    path.parent.mkdir(exist_ok=True)
                # Depth is saved as 16-bit PNG in raw units; multiply by depth_scale_m for metres.
                images = {
                    "rgb": np.asanyarray(color_frame.get_data()),
                    "depth": np.asanyarray(depth_frame.get_data()),
                    "depth_native": np.asanyarray(native_frame.get_data()),
                }
                for name, image in images.items():
                    if not cv2.imwrite(str(paths[name]), image):
                        raise RuntimeError(f"Could not save {name} for frame {stamp}")
                paths["meta"].write_text(json.dumps({
                    "rgb": paths["rgb"].name,
                    "depth": paths["depth"].name,
                    "depth_native": paths["depth_native"].name,
                    "depth_scale_m": depth_scale,
                    "filtered": use_filters,
                    "frames_filtered": n_frames,
                    "color": frame_metadata(color_frame),
                    "depth_frame": frame_metadata(raw_depth_frame),
                }, indent=2))
                count += 1
                print(f"Saved {stamp}")
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, cv2.error) as error:
        raise SystemExit(str(error)) from error
    except KeyboardInterrupt:
        pass