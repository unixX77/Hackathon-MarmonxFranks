# Synthetic brick scenes (Isaac Sim)

Generates RGB-D scenes of the real setup — grey studded LEGO baseplate with black tape, 2x4
bricks (31.8 x 15.8 x 9.6 mm + studs) in the five project colours, loose or stacked — and
writes each one like a real D435 capture, so the normal pipeline runs on it unchanged.
Ground truth comes from the renderer and the scene description (no SAM).

Run with the Isaac Sim pip environment (Isaac Sim 6.0.1). The standalone 6.1.0 install
crashes at startup because of its bundled PyTorch.

```bash
# ~2 s per image on an RTX 4060 laptop GPU
~/env_isaacsim/bin/python sim/generate_brick_scenes.py --out data/sim/v2 --scenes 300 --views 2 --seed 100

# train the colour model on sim, score it on real labelled bricks only
python sim/train_colour_from_sim.py data/sim/v2 --out classifier/colour_model_sim_v2.json \
  --real-capture data/top_layer_bricks/20261026_124136/000000/bricks.json 1:red 2:reddish_brown 3:red 4:medium_nougat

# run the real pipeline on a simulated scene
./run_pipeline_classifier.sh --recording data/sim/v2/scene_00001 --frame 0
```

## Output per scene (`data/sim/<run>/scene_NNNNN/`)

| Path | Content |
|---|---|
| `rgb/000000.png` | colour image, 1920 x 1080 |
| `depth/000000.png` | uint16 depth in mm, aligned to colour (perfect: no D435 noise or holes) |
| `camera_info.json` | real D435 colour intrinsics, copied from `--camera-info` |
| `instance/000000.png` | uint16 brick id per pixel from the renderer (0 = background) |
| `labels/000000.json` | per brick: colour, layer, top-layer flag, top-face centre (world and camera optical frame, m), image angle (deg, +x toward +y, -90..90], centre pixel, visible pixels; camera pose; lamps; applied camera response |

## What is randomized

- Layouts: 1-10 loose bricks, or interlocked stacks of 2-4 layers (alternating direction,
  partial top layer); colours uniformly random.
- Lighting: dome light plus 1-3 lamps (position, size 2-15 cm, brightness, colour temperature).
- Brick glossiness (roughness 0.15-0.55), producing real glare.
- Camera 30-45 cm above the plate, slight tilt, image up roughly along +x.
- Camera response: emulated auto white balance (plate made roughly neutral) and auto
  exposure (plate brightness 140-180), like the D435 with both auto modes on.

## Known limits

- Brick colours are LEGO's official values, not measured on this camera. Medium nougat
  renders too yellow (hue 78 deg vs 62 deg on the real camera), so a sim-trained model calls real
  nougat dark orange. Calibrate the colours against real baseplate photos before relying on it.
- No D435 depth noise, no robot arm, lamps are round (no ring light).
- Validation so far: the pipeline on sim scenes finds every top brick with 0.35 mm median
  centre error and 0.6 deg median angle error; the sim-trained colour model scores 92 % on held-out
  sim scenes.
