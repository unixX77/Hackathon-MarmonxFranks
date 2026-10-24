# SAM 3 LEGO brick detection

Instance and colour segmentation of 2x4 LEGO bricks, with a good / bad label per brick: [`segment_lego_sam3_single_prompt.py`](segment_lego_sam3_single_prompt.py). For RGB-D recordings of a stack, see [Top layer from depth](#top-layer-from-depth-with-the-brick-size-prior).

![Five touching bricks, one per colour](results_single_prompt/test_set/rgb_20261023_165012_439406/preview.png)

*Left: camera RGB. Middle: one mask per brick, with its id and SAM 3 score. Right: each brick filled with its colour class.*

## Prompt

SAM 3 is prompted with a single text prompt:

```
lego brick
```

The prompt only finds the bricks. Each brick's colour class comes from its measured colour, because SAM 3 does not reliably tell close shades apart (red, dark red, reddish brown). The good / bad label follows from the colour class.

## Brick classes

| Class | LEGO element | Quality |
|---|---|---|
| `red` | 300121 | good |
| `dark_red` | 6117418 | good |
| `dark_orange` | 6487946 | good |
| `medium_nougat` | 6135191 | bad |
| `reddish_brown` | 4211201 | bad |

## How it works

1. Run SAM 3 (`facebook/sam3`, via Hugging Face `transformers`) with the prompt and keep detections scoring at least 0.5.
2. Filter the detections:
   - drop a mask when more than half of it is already covered by a higher-scoring mask;
   - drop masks smaller than 300 px;
   - drop masks that are not rectangular (mask area below 0.7 of its minimum-area rectangle);
   - drop slivers smaller than 0.4 of the median brick area, such as parts of lower-layer bricks seen through gaps.
3. Measure each brick's Lab colour inside the mask, away from its edges, shadows and highlights.
4. Match all bricks of an image to the reference colours in `PALETTE` together. A global Lab offset is estimated for the image at the same time, which absorbs white-balance and exposure changes. Bricks still far from every reference colour are rejected.
5. Label each brick good or bad from its colour class.

## Run

Set up the `sam3` environment and Hugging Face access as described in the [perception README](../README.md#sam-3-setup). Then run on a folder of PNGs or a single image:

```bash
python segment_lego_sam3_single_prompt.py path/to/rgb --out results
```

| Option | Default | Effect |
|---|---|---|
| `--prompt` | `lego brick` | text prompt |
| `--threshold` | 0.5 | minimum SAM 3 score |
| `--max-overlap` | 0.5 | drop a mask if this fraction is already covered by better ones |
| `--min-area` | 300 | minimum mask area in pixels |
| `--min-rect` | 0.7 | minimum rectangularity |
| `--min-rel-area` | 0.4 | minimum area as a fraction of the median brick area |
| `--max-colour-dist` | 35 | rejection distance from the nearest reference colour, after the offset |

## Output

One folder per input image `<name>.png`:

| File | Content |
|---|---|
| `semantic.png` | uint8 class per pixel: 0 = background, 1–5 = class in the table above |
| `instance.png` | uint16 brick id per pixel: 0 = background |
| `instances.json` | per brick: id, colour, class id, quality, score, box, area, centroid, minimum-area rectangle (centre, size, angle), rectangularity, `touches_border`, Lab colour, colour distance. Also the image's Lab colour offset and the rejected detections with the reason |
| `preview.png` | input, instances, colour classes side by side |

`summary.json` in the output folder has the brick count per colour and the number of rejected detections per image.

## Results

[`results_single_prompt/`](results_single_prompt/) holds the output for three images from each dataset.

**Test set** (640x480, the five colours the palette was calibrated on): `test_set/`

| Image | Bricks | Good | Bad | Rejected |
|---|---|---|---|---|
| rgb_20261023_164946_396640 | 5 | 3 | 2 | 1 |
| rgb_20261023_165012_439406 | 5 | 3 | 2 | 0 |
| rgb_20261023_165239_744023 | 6 | 6 | 0 | 0 |

All bricks are found and classed correctly, including touching bricks of the same colour:

![Two chains of touching bricks](results_single_prompt/test_set/rgb_20261023_165239_744023/preview.png)

**Brick stack** (1920x1080, a different camera setup): `stack/`

| Image | Bricks | Good | Bad | Rejected |
|---|---|---|---|---|
| rgb_20261023_185622_491366 | 61 | 35 | 26 | 20 |
| rgb_20261023_185951_544446 | 62 | 34 | 28 | 15 |
| rgb_20261023_190231_851195 | 54 | 27 | 27 | 29 |

![Brick stack](results_single_prompt/stack/rgb_20261023_185622_491366/preview.png)

The instances are good: in the image above, every top-layer brick is found as one mask. The colour classes are not reliable yet on this camera:

- The estimated colour offset is very large (about +55 in lightness), so the palette does not fit these images.
- The five brick types are grouped consistently, but under the wrong names. Orange comes out as `red`, reddish brown as `dark_red`, and grey-brown as `reddish_brown`. `dark_orange` and `medium_nougat` look right.
- Some slivers of the layer below, visible in the gaps, are still kept. A higher `--min-rel-area` (about 0.6) or `--threshold` (about 0.85) removes most of them.

To fix the colours, recalibrate `PALETTE` on this camera. Take the reference colours from the `lab` field in `instances.json`, with the camera's white balance and exposure fixed.

## Top layer from depth, with the brick-size prior

[`top_layer_bricks.py`](top_layer_bricks.py) finds the individual bricks of the top layer of a stack in RealSense RGB-D recordings, for picking. It uses the segmentation above and adds depth and a prior on the brick size (30 x 15 x 10 mm).

![Top layer with fitted brick footprints](results_top_layer/20261024_102430/000017/preview.png)

*Top left: camera RGB. Top right: all SAM 3 bricks, filled with their colour class. Bottom left: the top layer selected from depth. Bottom right: the bricks after the size prior, with the fitted 30 x 15 mm footprint of each brick in yellow.*

### How it works

1. Segment the RGB frame with SAM 3 and the "lego brick" prompt, and give each brick a colour class, as described above.
2. Back-project the depth (aligned to colour) to 3D and fit the table plane with RANSAC to the depth around the bricks.
3. Measure each brick's height above the table (80th percentile of its eroded mask) and keep the bricks within half a brick height of the highest one. This step is in [`top_layer_depth.py`](top_layer_depth.py), which can also be run on its own.
4. Project each kept mask onto the table plane at its height and measure it in millimetres:
   - about 30 x 15 mm: one brick;
   - about 15 mm wide and n x 30 mm long: n touching bricks merged into one mask, split into n bricks;
   - about 15 mm wide but clearly shorter than 30 mm, or much narrower than 15 mm: a partly hidden brick or a sliver, rejected;
   - anything else: kept and flagged `irregular`.
5. Fit a 30 x 15 mm footprint to each brick and report its centre in the camera frame, its yaw on the table and its layer.

### Run

The input is one extracted recording folder with `rgb/`, `depth/` (aligned to colour, 16-bit) and `camera_info.json` (with `aligned_depth_intrinsics` and `depth_scale_m`):

```bash
python top_layer_bricks.py path/to/recording --frame 000020
```

A window shows the preview; add `--no-show` to skip it. Without a display the preview is only saved. `--brick-size-mm L W H` and `--size-tol` (default 0.3, i.e. ±30%) set the prior.

Output in `data/top_layer_bricks/<recording>/<frame>/`:

| File | Content |
|---|---|
| `final_instances.png` | uint16 brick id per pixel after splitting, 0 = background |
| `final_semantic.png` | uint8 colour class per pixel |
| `overlay.png` | top-layer bricks as coloured patches with white boundaries and the fitted footprints (yellow; magenta for `irregular`) |
| `preview.png` | the four panels shown above |
| `bricks.json` | per brick: status (`brick`, `split`, `partial`, `irregular`), colour, quality, measured size, height, layer, centre in the camera frame (m), yaw on the table, footprint in pixels, and the 3D box in the camera frame (centre, size, rotation, 8 corners) |
| `poses.csv` | one row per brick: box centre `cx, cy, cz`, top-face centre, yaw, size, and the 8 box corners `c0`–`c7` (top face 0–3, bottom face 4–7), all in the camera frame in metres |

### Results

[`results_top_layer/`](results_top_layer/) holds three frames recorded with a RealSense D435 about 42 cm above the table.

| Recording / frame | Top-layer masks | Bricks | Of which split | Rejected |
|---|---|---|---|---|
| 20261024_095657 / 000020 | 33 | 28 | 7 | 9 |
| 20261024_100345 / 000030 | 21 | 18 | 6 | 7 |
| 20261024_102430 / 000017 | 32 | 30 | 2 | 3 |

- The table plane fits to about 0.9 mm (median residual).
- The size prior fits the data well: single top-layer bricks measure about 31 x 16 mm on the table.
- In `20261024_100345` only the left half of the stack has a top layer, and only those bricks are kept.

Known issues:

- Where SAM 3 returns a top-layer brick only as thin slivers, the size prior rejects it, so that brick is missing (top of the left column in `20261024_095657`).
- The colour palette was calibrated on the webcam images, so the colour names are not reliable on the RealSense yet. The masks and the top-layer selection do not depend on them.
