#!/usr/bin/env bash
# End to end: capture with d435_settings.json -> SAM3 top-layer bricks -> grasp poses (x y z angle).
# Capture window: Enter saves a frame, q quits. Detection runs on the last saved frame.
# Usage: ./run_pipeline.sh [extra top_layer_bricks.py options, e.g. --top-min-support 1]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/sam3/bin/python}

"$PY" data_collection/capture_d435.py --settings d435_settings.json --output data/captures
SESSION=$(ls -td data/captures/*/ | head -1)
FRAME=$(ls "$SESSION/rgb" 2>/dev/null | tail -1 | cut -d. -f1)
if [ -z "$FRAME" ]; then
    echo "No frame saved in $SESSION (press Enter in the capture window before q)"; exit 1
fi

"$PY" perception/sam3_detection/top_layer_bricks.py "$SESSION" --frame "$FRAME" \
    --out data/top_layer_bricks --no-show "$@"
RESULT=data/top_layer_bricks/$(basename "$SESSION")/$FRAME

echo "Poses above: x y z = top-face centre in camera_color_optical_frame (m),"
echo "angle = long side in the image from +x (right) toward +y (down), deg in (-90, 90]."
echo "Results: $RESULT (bricks.json, poses.csv, overlay.png, preview.png)"
