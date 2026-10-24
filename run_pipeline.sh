#!/usr/bin/env bash
# End to end: capture with d435_settings.json -> SAM3 top-layer bricks -> brick centres for MoveIt.
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

"$PY" - "$RESULT/bricks.json" <<'EOF'
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

bricks = [b for b in json.load(open(sys.argv[1]))['bricks'] if b['id']]
print(f'\n{len(bricks)} bricks, camera colour optical frame (x right, y down, z forward), metres')
print('top-face centre = grasp position, quaternion x y z w = gripper z pointing down into the brick')
print(f'{"id":>3} {"colour":<14}{"quality":<8}{"x":>8}{"y":>8}{"z":>8}  {"qx":>8}{"qy":>8}{"qz":>8}{"qw":>8}')
for b in bricks:
    long_, short, up = np.array(b['box_rotation_cam']).T
    q = R.from_matrix(np.column_stack([long_, -short, -up])).as_quat()
    print(f'{b["id"]:>3} {b["colour"]:<14}{b["quality"]:<8}' + ''.join(f'{v:8.4f}' for v in b['centre_cam_m'])
          + '  ' + ''.join(f'{v:8.4f}' for v in q))
EOF
echo "Results: $RESULT (bricks.json, poses.csv, overlay.png, preview.png)"
