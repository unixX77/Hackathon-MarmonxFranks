# Separate top-plane segmentation, quality and pose pipeline

`run_modular_pipeline.sh` is independent of the existing `run_pipeline.sh`.
It launches the existing stages in separate processes:

1. `data_collection/capture_d435.py` (optional new camera capture).
2. `segmentation_sam3_bounding_box/segment_lego_rgbd_topplane.py`.
3. `classifier/classify_top_brick_quality.py`.
4. `pose_info/export_camera_boxes.py`.

## New capture

Connect the D435, close Viewer, and run from the repository root:

```bash
./run_modular_pipeline.sh
```

The capture uses `d435_settings.json`. Press Enter to save frames and q to end
capture. The last saved frame is then segmented, classified, and exported.
No detection/classification windows open; their original visualizations are
saved. Capture itself opens its normal RGB-D preview window.

The launcher defaults to `$HOME/miniconda3/envs/sam3/bin/python`; override `PY`
if using another environment containing SAM3 and pyrealsense2. For example:

```bash
PY=/path/to/python ./run_modular_pipeline.sh
```

## Existing recording

```bash
./run_modular_pipeline.sh \
  --recording data_collection/data/extracted/data4/20261024_095657 \
  --frame 20
```

`--recording` takes one session folder containing `rgb/`, aligned `depth/`, and
`camera_info.json`. Sources remain unchanged; the run keeps a symlink to the
existing recording. Keep the original recording available for downstream reruns.

## Outputs

Each run gets a fresh directory under `data/modular_pipeline_results/`:

```text
run_<timestamp>/
  inputs/capture/<recording>/       # new capture, or link to existing recording
  01_segmentation/capture/<recording>/<frame>/
    result.png
    comparison.png
    top_boxes.json
  02_classification/capture/<recording>/<frame>/
    quality_result.png
    quality_comparison.png
    quality_boxes.json
    camera_boxes.json
    brick_poses.json
    centers_camera_m.csv
  pipeline.json                    # paths and frame identity
```

The classifier and exporter also save their usual aggregate summaries. Any
failure stops subsequent stages and writes `INCOMPLETE.json`; old results are
never silently reused. Centers printed at completion and exported in the CSV
are physical box-body centers in camera optical coordinates (x right, y down,
z forward), in metres. Top-face centers are also retained in camera_boxes.json.
The pose exporter uses nominal brick dimensions 31.8 x 15.8 x 9.6 mm; these
are the standalone module's defaults, independent of the other pipeline's
32 x 16 x 9.6 mm prior. Pose/grasp metadata is not a robot motion command.

## Options

- `--out PATH`: choose a separate result root.
- `--settings PATH`: choose a different Viewer settings JSON for capture.
- `--camera-pose PATH`: pass a calibrated transform to the pose exporter.
- `--temporal-frames N`: default 1. Use a larger number only for consecutive
  camera frames with a stationary scene, not Enter snapshots between picks.
- `--segmentation-args ...`: forward options to the segmentation stage; put
  this last. Do not override its input/output routing flags.

For example, to allow objects less than 20 mm above the support surface:

```bash
./run_modular_pipeline.sh --recording data/captures/SESSION --frame 0 \
  --segmentation-args --min-object-height-mm 5
```

Segmentation still needs valid depth, a visible support plane and a usable SAM
seed. Its lattice assumes regular brick geometry. A successful run validates
file compatibility, not detection or robot positioning accuracy.
