# RealSense RGB-D capture

Captures colour and depth frames from an Intel RealSense depth camera (e.g. D435 / D455), together with the camera intrinsics and extrinsics.

Install and run from a terminal on the computer with the camera, connected over USB 3:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python capture_rgbd.py
```

The preview shows colour on the left and depth (0–2 m) on the right. Click the preview window to focus it. Press **Enter** to save the displayed frame, and **q** (or Escape) to quit. Closing the window also quits.

## Output

Each run creates a session folder in `data/` beside the script:

```
data/session_<date>_<time>/
  camera_info.json          intrinsics, extrinsics, depth scale, device info
  rgb/rgb_<stamp>.png       colour image (8-bit, 3 channels)
  depth/depth_<stamp>.png   depth aligned to the colour image (16-bit, raw units)
  meta/meta_<stamp>.json    per-frame number, timestamps, exposure, gain
```

Depth is aligned to the colour camera, so each depth pixel matches the colour pixel at the same position and the **colour intrinsics** apply to both images. To get metres, multiply the depth values by `depth_scale_m` (usually 0.001). A value of 0 means no depth.

`camera_info.json` contains:

| Field | Content |
|---|---|
| `color_intrinsics` | `fx`, `fy`, `ppx`, `ppy`, 3x3 `K`, distortion model and coefficients |
| `depth_intrinsics` | the same for the raw (unaligned) depth stream |
| `depth_to_color_extrinsics`, `color_to_depth_extrinsics` | rotation, translation (m) and 4x4 `T` between the two sensors |
| `depth_scale_m` | metres per depth unit |
| `device` | name, serial number, firmware, USB type |
| `camera_pose` | the camera pose in the robot/world frame, if given with `--camera-pose` |

The camera only knows its internal extrinsics (between its own sensors). Its pose relative to the robot or table comes from a hand-eye calibration. Save that pose as JSON (for example `{"frame": "robot_base", "T_base_camera": [[...], [...], [...], [...]]}`) and pass it with `--camera-pose pose.json` to store it with the session.

Back-projecting a pixel `(u, v)` with depth `d` to a 3D point in the colour camera frame:

```python
z = d * depth_scale_m
x = (u - ppx) * z / fx
y = (v - ppy) * z / fy
```

## Options

```bash
python capture_rgbd.py --width 640 --height 480 --fps 30 --serial <serial> --output data
```

The default stream is 1280x720 at 30 fps. If the camera does not support the requested mode (or is on USB 2), the script exits with an error; try 640x480.

Requires a graphical desktop and the GUI-enabled `opencv-python` package (not `opencv-python-headless`). On Linux, the [librealsense udev rules](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md) may be needed for camera access without root.
