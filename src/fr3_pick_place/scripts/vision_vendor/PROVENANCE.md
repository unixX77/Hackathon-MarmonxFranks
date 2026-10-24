# Teammate perception source

These three files were copied unchanged from the user-supplied
`Hackathon-MarmonxFranks-main.zip`, archive revision
`dc72e47995f62f5b114f90b7fec1b8b2536151e6`, directory
`perception/sam3_detection/`:

- top_layer_bricks.py
- top_layer_depth.py
- segment_lego_sam3_single_prompt.py

The ROS adapter imports their functions and keeps the model loaded. It joins
the original detection/depth quality metrics back onto the fitted brick records.
No bundled example JSON is used for robot targets. Coordinate and collision
handling live outside these files. No model weights are bundled.
