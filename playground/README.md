# RGB-D perception playground data

The playground is a static, browser-only visualization of recorded outputs. It does not run SAM 3 or any model in the browser. `manifest.json` lists scenes and their available modes; each scene's `scene.json` holds its matching image dimensions, metadata and annotations. The page fetches a scene's full data only when it is selected.

## Included scenes and provenance

- `data3/20261024_091244`, `data3/20261024_091640`, and `data3/20261024_091956`, frame `000000`: original RGB and aligned depth from `data_collection/extracted_rgbd/extracted/`; oriented boxes from `perception/sam3_detection/results_rgbd_topplane_v2/`; quality and camera-frame cuboids from `classifier/results_rgbd_quality_v1/`. Export checks the dataset, recording, frame, ID sets, box corners, and image dimensions before combining them.
- `rgb_20261023_165040_935110`: a **separate, earlier RGB-only** SAM 3 run from `perception/sam3_detection/results/`, paired with its source image from `data 2/rgb/`. Its saved `instance.png` has four actual candidate masks. This is not a top-layer-v2 result and has no matched depth, top-layer classification, or 3D pose in this playground. Export validates the source image against the saved comparison, mask IDs, and instance pixel counts.

The three RGB-D recordings do not have their original SAM 3 instance masks saved. Their SAM segmentation mode, mask boundaries and opacity controls are unavailable. The fourth scene supports genuine masks and boundaries but not depth, top-layer detection, quality or pose. The UI never combines outputs from different frames or different pipeline versions.

## Geometry and depth

RGB and depth frames are 640×480 and depth is registered to the color frame. Depth colors use the perceptually ordered viridis map, clipped to the 2nd–98th valid-depth percentiles separately for each scene. Scale endpoints are displayed in metres and black represents invalid depth. The displayed depth is the saved single frame; the top-layer pipeline used a temporal median of several depth frames.

Bricks use the original oriented image-space corners and centers. Saved 3D cuboid corners are projected using the matching aligned camera intrinsics and the same pinhole convention as `classifier/visualize_brick_poses.py`: `u = fx*x/z + cx`, `v = fy*y/z + cy`. The exporter enables projection only when saved distortion coefficients are present and zero. Coordinates are in the camera optical frame, in metres: +x right, +y down, +z forward. The top-face center and cuboid center are distinct. Robot-base coordinates require a separate hand-eye transform.

Quality values and color distances are saved model outputs, not ground truth or calibrated probabilities. No instance segmentation area is reported when a mask was not saved. The earlier RGB-only scene exposes its actual mask area and saved SAM candidate score; it is not scored as a top-layer detection result.

## Export more recorded scenes

Run from the website project directory, using a Python environment with NumPy, Pillow, matplotlib, and OpenCV:

```bash
python -m pip install numpy pillow matplotlib opencv-python
python tools/export_playground.py \
  --project /path/to/Hackathon-MarmonxFranks \
  --scene data3:20261024_091244 \
  --scene data3:20261024_091640 \
  --scene data3:20261024_091956 \
  --rgb-sam-scene rgb_20261023_165040_935110 \
  --output playground
```

Repeat `--scene dataset:recording` for additional recordings with matching RGB, depth, `camera_info.json`, `top_boxes.json`, `quality_boxes.json`, and `camera_boxes.json`; `--frame` selects a frame shared by all listed recordings. `--rgb-sam-scene` optionally adds a standalone RGB SAM 3 run that has an `instance.png` and `instances.json` at the paths used by the exporter. A scene export fails if the source frame, geometry or ID sets do not match. Inspect and commit the new files and manifest to the website branch, then redeploy.

To add SAM masks for the RGB-D top-layer-v2 scenes, the original masks must first be recovered or re-exported from SAM 3 for the *same recorded frames*. The lattice boxes and SAM proposal IDs are different concepts: verify any association explicitly. The current exporter does not invent masks, derive them from bounding boxes, or automatically associate SAM proposals with lattice detections.

To preview locally, run `python -m http.server` from the website directory and open the local HTTP address it prints. Opening `index.html` directly as a file can block the browser from fetching the scene manifest.
