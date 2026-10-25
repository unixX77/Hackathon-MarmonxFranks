# Camera-frame brick bounding boxes and poses

This folder contains the standalone geometry exporter used after top-layer
segmentation and good/bad classification. It converts every detected oriented
brick box from image coordinates into a fitted physical 3D cuboid in the Intel
RealSense camera coordinate frame and writes `camera_boxes.json`.

The implementation is
[`export_camera_boxes.py`](export_camera_boxes.py). It does not rerun SAM 3 or
the colour classifier. It consumes their saved JSON metadata and the recording's
camera calibration.

## Pipeline position

```text
SAM 3 + RGB-D top-layer detector
              │
              └── top_boxes.json
                         │
Good/bad classifier ─────┤
              │          │
              └── quality_boxes.json
                         │
                         ▼
             camera-box / pose exporter
                         │
              ┌──────────┴──────────┐
       camera_boxes.json      brick_poses.json
       centres + corners      full 6-DoF metadata
```

## Geometry architecture

For every classified top-layer box:

1. Read the oriented pixel centre, orientation, quality and colour from
   `quality_boxes.json`.
2. Read the fitted support-plane normal, plane offset and selected top height
   from the corresponding segmentation `top_boxes.json`.
3. Read the aligned colour-camera intrinsics from `camera_info.json`.
4. Back-project the pixel centre into a camera ray:

   ```text
   ray = [(u - cx) / fx, (v - cy) / fy, 1]
   ```

5. Intersect that ray with the fitted physical top plane. This gives the exact
   top-face centre in camera coordinates, rather than treating pixels as metric
   positions.
6. Project the detected long image axis into the top plane. Together with the
   outward plane normal, this constructs an orthonormal brick rotation:

   ```text
   +X = brick long axis
   +Y = brick short axis
   +Z = outward top-surface normal
   ```

7. Fit the nominal physical 2x4 brick dimensions around that pose:

   ```text
   length = 31.8 mm
   width  = 15.8 mm
   height =  9.6 mm
   ```

8. Construct four top-face corners and four bottom-face corners in metres.

## Coordinate convention

All `camera_boxes.json` metric fields use the RealSense colour optical frame:

```text
frame_id: camera_color_optical_frame
+x: image/camera right
+y: image/camera down
+z: forward from the camera (depth)
units: metres
```

These are camera-frame coordinates, not Franka base-frame coordinates. Robot
motion planning requires a separate hand-eye transform `T_base_camera`.

## Run

From the repository root, after segmentation and classification results exist:

```bash
conda run -n sam3 python pose_info/export_camera_boxes.py \
  classifier/results_rgbd_quality_v1
```

The exporter searches recursively for `quality_boxes.json`. For every frame it
writes:

- `camera_boxes.json`: compact 3D centres, rotation, size and cuboid corners;
- `brick_poses.json`: full 6-DoF pose, quaternion, approach and pre-grasp data.

It also writes `all_brick_poses.json` at the results root as an index over the
complete dataset.

## `camera_boxes.json` schema

Frame-level fields:

| Field | Meaning |
|---|---|
| `schema` | format version, currently `camera-frame-brick-boxes-v1` |
| `frame_id` | coordinate frame for all metric values |
| `units` | metres |
| `coordinate_convention` | axis definition |
| `dataset`, `recording`, `frame` | source frame identity |
| `bricks` | array containing one record per detected top-layer brick |

Per-brick fields:

| Field | Meaning |
|---|---|
| `id` | detection identifier within the frame |
| `quality`, `colour` | classifier output |
| `classification_confidence` | high/low prototype confidence |
| `center_pixel_xy` | original image centre `[u,v]` |
| `corners_pixel_xy` | original four oriented image corners |
| `top_face_center_camera_m` | graspable top-surface centre `[x,y,z]` |
| `box_center_camera_m` | physical cuboid/volume centre `[x,y,z]` |
| `box_size_m` | `[length,width,height]` |
| `box_rotation_camera` | 3x3 rotation; columns are long, short and outward-normal axes |
| `top_corners_camera_m` | four top-face 3D corners |
| `bottom_corners_camera_m` | four bottom-face 3D corners |
| `box_corners_camera_m` | all eight corners, top face followed by bottom face |
| `corner_order` | explicit semantic order for the eight corners |

The top and bottom corners use this ordering:

```text
0 top(+long,+short)       4 bottom(+long,+short)
1 top(-long,+short)       5 bottom(-long,+short)
2 top(-long,-short)       6 bottom(-long,-short)
3 top(+long,-short)       7 bottom(+long,-short)
```

Example record:

```json
{
  "top_face_center_camera_m": [0.1078238, -0.1032182, 0.3630303],
  "box_center_camera_m": [0.1079379, -0.1035222, 0.3678193],
  "box_size_m": [0.0318, 0.0158, 0.0096],
  "box_corners_camera_m": [
    [0.1157566, -0.0873556, 0.3638482],
    [0.1156864, -0.1190918, 0.3618354],
    [0.0998909, -0.1190808, 0.3622124],
    [0.0999611, -0.0873446, 0.3642252],
    [0.1159848, -0.0879636, 0.3734262],
    [0.1159146, -0.1196997, 0.3714134],
    [0.1001191, -0.1196887, 0.3717904],
    [0.1001893, -0.0879526, 0.3738032]
  ]
}
```

## Examples

- [`examples/dense_camera_boxes.json`](examples/dense_camera_boxes.json): a
  complete dense top layer;
- [`examples/single_camera_box.json`](examples/single_camera_box.json): a
  single remaining top brick.

## Optional robot-base transform

The exporter can additionally transform the full brick pose into a robot base
frame when a calibrated hand-eye file is provided:

```json
{
  "frame": "panda_link0",
  "T_base_camera": [
    ["r00", "r01", "r02", "tx"],
    ["r10", "r11", "r12", "ty"],
    ["r20", "r21", "r22", "tz"],
    [0, 0, 0, 1]
  ]
}
```

```bash
conda run -n sam3 python pose_info/export_camera_boxes.py \
  classifier/results_rgbd_quality_v1 \
  --camera-pose hand_eye.json
```

Do not use an identity or guessed transform. `T_base_camera` must come from a
real hand-eye calibration of the mounted camera and Franka robot.

## Limitations

- The cuboid dimensions are the nominal LEGO 2x4 dimensions, not independently
  estimated for every brick.
- Centre accuracy depends on camera intrinsics, depth alignment and top-plane
  fitting accuracy.
- The exporter describes object geometry; collision checking, tool-centre-point
  calibration, grasp offsets and robot joint constraints remain the motion
  planner's responsibility.
