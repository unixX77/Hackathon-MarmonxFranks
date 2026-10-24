# Webcam RGB capture

For RGB-D capture with an Intel RealSense depth camera, see [realsense_rgbd](realsense_rgbd/README.md).

Install and run from a terminal on the computer with the webcam:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python capture_rgb.py
```

For the Logitech C920 connected to this system, run
`python capture_rgb.py --camera 2`.

Click the preview window to focus it. Press **Enter** to save the displayed
frame, and **q** (or Escape) to quit. Closing the window also quits.

Images are saved as timestamped, full-resolution color PNGs in `data/rgb/`
beside the script. Preview text is not included in saved images.

For a different camera or destination:

```bash
python capture_rgb.py --camera 1 --output data/rgb
```

Requires a graphical desktop and the GUI-enabled `opencv-python` package
(not `opencv-python-headless`).

## Intel RealSense RGB-D capture

Requires an Intel RealSense camera with color and depth streams; the Logitech
C920 only provides color. From this directory:

```bash
pip install -r requirements-rgbd.txt
python capture_rgbd.py
```

Press **Enter** in the preview to save a paired color and aligned depth image;
press **q** to quit. Each run creates `data/session_<timestamp>/` containing
`rgb/`, `depth/`, `meta/`, and `camera_info.json`. Preview overlays are not saved.
Depth PNGs contain 16-bit raw values: multiply by `depth_scale_m` to get metres;
zero means missing depth. Use `aligned_depth_intrinsics` for the saved depth,
not the native `depth_intrinsics`. Frame metadata records camera timestamps
and frame numbers for both streams; streams may have different timestamps.

Calibration includes sensor intrinsics and depth/color extrinsics. A pose in
the robot/world frame is not estimated: optionally supply your own JSON via
`--camera-pose path/to/pose.json`.

If the default 1280x720 mode is unavailable, try:

```bash
python capture_rgbd.py --width 640 --height 480
```

Use `--serial SERIAL` to select a RealSense, or `--output PATH` to change the
session parent directory. Installing `pyrealsense2` requires a supported Python
version/platform.

RGB-D defaults now request 1920x1080 color and 1280x720 native depth at
30 FPS, enable the emitter and depth auto-exposure, and select High Accuracy
when supported. Saved aligned depth has the color image dimensions; alignment
does not increase the native depth detail. Supported modes depend on the camera.
Override separately with `--rgb-width`, `--rgb-height`, `--depth-width`, and
`--depth-height`, or override both with the existing `--width`/`--height` flags.

High Accuracy rejects uncertain measurements and can leave more missing pixels.
Optional `--laser-power VALUE`, `--depth-exposure MICROSECONDS`, and
`--depth-gain VALUE` are checked against device limits. Manual exposure disables
depth auto-exposure. Projector power is left at the preset/device default unless
specified. Applied settings are recorded in `camera_info.json`. Depth is saved
without smoothing or hole filling to preserve measurements.

## Extract Viewer SDK recordings for depth integration

From the project root, with `.venv` activated:

```bash
python data_collection/extract_rgbd.py data3 data4
```

Output is grouped by source folder and recording:

```text
data_collection/data/extracted/data3/<recording>/
  rgb/000000.png          # color image
  depth/000000.png        # uint16 depth aligned to RGB
  depth_native/000000.png # original uint16 sensor depth
  meta/000000.json        # frame numbers, timestamps, exposure when available
  camera_info.json       # intrinsics, depth scale, sensor extrinsics
  recording_settings.json # original SDK string topics, settings and metadata
  frames.csv             # relative image paths and timestamp associations
  summary.json           # extraction counts and pairing limits
```

`data4/` has the same structure. Multiply depth pixels by `depth_scale_m` from
`camera_info.json` for metres. Zero means missing depth. For aligned depth use
`aligned_depth_intrinsics`; for native depth use `depth_intrinsics`. Color and
native depth use different optical coordinate frames. The saved depth-to-color
extrinsics describe that sensor relationship, not motion between captures.
Depth coordinates are x right, y down, z forward.

Pairs come from SDK framesets with matching timestamp domains, a maximum 20 ms
timestamp difference, and no reused color/depth frame. This does not guarantee
simultaneous hardware exposure. `--max-delta-ms` changes the pairing limit;
`--every 10` saves every tenth pair. Outputs are never overwritten; use a fresh
`--output PATH` for another run. `INCOMPLETE.json` marks failed extractions.
The extractor applies alignment only, without smoothing or hole filling.

**Multi-view depth/TSDF integration also requires a camera pose for each view**
(from registration, odometry, or external tracking). Recordings do not provide
these poses. Neighboring frames within each short recording may be nearly
identical; sampling can avoid overweighting repeated views. SDK playback may
require running from a regular desktop terminal rather than a restricted
sandbox. This tool needs the installed SDK version with `.db3` playback support.
