# SAM 3 LEGO brick detection

Instance and colour segmentation of 2x4 LEGO bricks, with a good / bad label per brick: [`segment_lego_sam3_single_prompt.py`](segment_lego_sam3_single_prompt.py).

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
