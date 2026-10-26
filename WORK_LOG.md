# Brick perception — work log (25–26 Oct 2026)

What was investigated, built and measured, the commands to run it, and what is still open.
Commits are on branch `colour-model-and-batch-compare`: `73f7cc2`, `653c4b3`, and the commit that
adds the simulation scripts (section 7), this log and the lone-brick threshold change.

---

## 1. Diagnosis: why bad bricks were missed

- The old classifier (`classifier/classify_top_brick_quality.py`) does draw bad bricks red; they were
  missing because nothing was labelled bad.
- Root cause: its fixed Lab colour references did not match this camera, and a per-frame Lab offset
  pushed reddish-brown bricks onto dark red.

## 2. Learned colour classifier (commit `73f7cc2`)

`classifier/colour_model.py` → `classifier/colour_model.json`

- Trained on the 224-brick single-brick dataset (`data/brick_dataset_extracted`, white table).
- Each brick: median CIELAB after white balance from the table, reduced to 3 features —
  **hue angle, lightness ÷ table lightness, chroma ÷ table lightness**.
- Linear discriminant analysis (5 class means + shared covariance), equal priors; probability per
  colour, argmax wins, `?` (low confidence) below 0.7.
- Frames with ≥ 10 bricks get a bounded per-frame hue / lightness / chroma correction for
  auto-exposure drift.
- Accuracy: ~90 % random split (optimistic), ~52 % with a lighting condition held out (pessimistic:
  lighting is confounded with colour in the dataset). On real stacks ~35–38 of ~40 bricks looked right.
- Used by `run_pipeline_classifier.sh` by default; `--colour-model none` restores the old references.

Colour separation (dataset medians): dark red hue 27°, red 35°, reddish brown 45°, dark orange 49°,
medium nougat 54°. **Nougat vs dark orange is the closest pair** (hue ~5° apart, same lightness).

## 3. Detector fixes (commits `73f7cc2`, `653c4b3`) — `perception/sam3_detection/`

- **Width limit** (`--max-width-factor`, default 1.4 × brick width = 22.4 mm): raised and stack-edge
  bricks measure ~20 mm wide because their side walls are visible, and were rejected by the old
  19.5 mm limit. 1.5 was too loose (a 23.8 mm mask covering part of a brick on top replaced the
  real top brick).
- **Lone top brick** (`--lone-top-brick`, default on): the top layer needs ≥ 2 masks, but a single
  mask one brick height higher is accepted if its depth shows one clean, flat brick (≥ 80 % valid
  depth, ≥ 70 % flat, ≤ 3 mm spread, 32 × 16 mm top-face footprint from depth points).
- Effect on the 51-capture batch: raised-brick captures went from 29 bricks on the wrong layer to
  the 4 real top bricks; stack-edge nougat bricks recovered; small towers now return the real top brick.
- Known edge case: a loose brick measured only 4.7 mm above the table is dropped (bad depth reading).
- Lone-brick flat-surface minimum lowered from 80 % to 70 % (26 Oct): a brick lying across two
  lower bricks measured 78 % (its edges blend into the bricks below) and was missed, so the
  pipeline returned the two bricks underneath. Replayed over 56 saved captures, only two results
  change, both to the correct top brick.

## 4. Grasp pose output (commit `653c4b3`)

Every run prints one line per brick, in the format the robot side asked for:

```
#3 reddish_brown bad x=0.108904 y=0.057324 z=0.263015 angle=-87.2
```

- `x y z`: **top-face centre** in `camera_color_optical_frame` (x right, y down, z forward), metres.
- `angle`: long side in the image from +x toward +y (clockwise on screen), degrees in (-90, 90].
- Centre from the top-face depth pixels when the face is seen whole (`centre_source: top_face_depth`),
  otherwise from the mask; median shift 0.35 mm, max 2.6 mm on the batch.
- Angle from the mask outline: on stack frames its error is ~1/3 of a fit to top-face points
  (median 0.36° vs 1.1°).
- Brick prior default changed to **32 × 16 × 9.6 mm**.
- `bricks.json` / `poses.csv` gained `angle_img_deg`, `centre_source`, `top_face_mm`.
- Don't send `yaw_deg_table` or the quaternion: different convention.
- The robot side's hand-eye worked example was re-computed and matches.

## 5. Tools

| Command | What it does |
|---|---|
| `./run_pipeline_classifier.sh` | capture (Enter saves, q quits) → detector → learned classifier → pose lines |
| `./run_all_captures.sh` | runs `run_pipeline_classifier` and `run_modular_pipeline` on every saved capture; writes `data/all_captures_results/batch_<ts>/index.html` with side-by-side comparisons |
| `python compare_pipeline_results.py <batch>` | rebuilds comparisons for an existing batch |
| `python pipeline_classifier/reclassify.py <geometry> --colour-model classifier/colour_model.json` | re-classify an existing result, no SAM |
| `python classifier/colour_model.py` | retrain the colour model on the dataset |
| `./run_pipeline_classifier.sh --second-opinion` | optional Claude vision check of final colours (needs `pip install anthropic` + an API key; API billing is separate from the Claude Code subscription; untested end to end) |

## 6. Camera, lighting and setup findings

- Colour stream **1920 × 1080 is the best mode** (native; 69.2° × 42.4°, 27.8 px/deg). 640 × 480 crops
  the sides (54.7° horizontal) and has less than half the detail. Depth 848 × 480 is the D435's best
  depth mode. Keep both.
- Keep the camera ≥ 28–30 cm above the top layer (closer → depth holes; one capture was at 26 cm).
- Glare washes out light glossy bricks (nougat) and is the main colour-error cause: use diffuse side
  lighting, keep table visible around the stack.
- Colour auto exposure / auto white balance are on, and the model was trained that way; changing
  colour settings needs retraining. Safe changes: power-line frequency 50 Hz, higher laser power.
- **Grey LEGO baseplate setup (26 Oct):** colours shift one class (brown → nougat, nougat → dark
  orange) because bricks measure ~2.5× lighter and ~7° warmer relative to the plate than in the
  white-table training photos. A single global correction (hue −5°, L × 0.40, chroma × 0.69) fixed
  all 4 test bricks → a per-setup colour calibration is the fix.
- `run_pipeline.sh` still uses the old built-in colour references; use `run_pipeline_classifier.sh`.

## 7. Isaac Sim synthetic data — `sim/` (see `sim/README.md`)

Runs with the pip install: `~/env_isaacsim/bin/python` (Isaac Sim 6.0.1). The standalone 6.1.0 in
`~/Downloads` crashes at startup (its bundled PyTorch fails to load).

`sim/generate_brick_scenes.py` — grey studded baseplate + black tape on a desk, 2×4 bricks with
studs in the 5 official LEGO colours, loose layouts and interlocked stacks (alternating layers,
partial top layer), 1–3 random lamps (position, size, brightness, colour temperature), random
glossiness (glare), camera 30–45 cm up with slight tilt and the **real D435 1920 × 1080 intrinsics**,
emulated auto white balance / auto exposure. ~2 s per image.

Each scene is written like a real capture, so the real pipeline runs on it unchanged:
`rgb/`, `depth/` (uint16 mm), `camera_info.json`, plus ground truth `instance/` (from the renderer,
no SAM) and `labels/` (colour, layer, top-layer flag, top-face centre in camera frame, image angle).

```bash
~/env_isaacsim/bin/python sim/generate_brick_scenes.py --out data/sim/v2 --scenes 300 --views 2 --seed 100
python sim/train_colour_from_sim.py data/sim/v2 --out data/sim/v2_colour_model.json \
  --real-capture data/top_layer_bricks/20261026_124136/000000/bricks.json 1:red 2:reddish_brown 3:red 4:medium_nougat
```

Bugs found and fixed: brick colours went stale (materials must be bound on body/stud prims, not the
parent); principal-point offset signs; camera near-clip at 1 m; plate colour cast without AWB emulation.

**Results**

| | result |
|---|---|
| Pipeline on sim scenes (vs exact ground truth) | 20/20 top bricks, centre error median 0.35 mm, angle 0.6° (sim depth is perfect) |
| Sim-trained colour model, held-out sim scenes | 92 % |
| … on real white-table dataset (224) | 30 % — different surface, not a fair test |
| … on real grey-plate capture (4 bricks) | 3/4 (current model 2/4) |

Sim vs real colour features on the grey plate: **red matches almost exactly** (L 0.61 vs 0.61,
chroma 0.43 vs 0.43), reddish brown close; **medium nougat renders too yellow** (hue 78° vs 62°).
No real grey-plate measurements exist for dark red and dark orange.

## 8. Open items / next steps

1. **Real baseplate photos** (about 20 frames, no manual labelling), with
   `python data_collection/capture_d435.py --output data/real_calib/<name>`:
   - Part A: one session per colour (`red`, `dark_red`, `dark_orange`, `medium_nougat`,
     `reddish_brown`), 3-5 bricks of only that colour, 4 frames each; the folder name is the label.
   - Part B (`rows`): 5 rows top to bottom red, dark red, dark orange, nougat, reddish brown,
     2 bricks each, 3 frames; the row order is the label. Test set only.
   Then: calibrate the simulated colours, train sim / real / sim+real models, score all on Part B,
   keep the winner, and add a recalibration script.
2. The sim-trained model `classifier/colour_model_sim_v2.json` is experimental: it calls real
   nougat dark orange until the simulated colours are calibrated.
3. Glare: record a per-brick glare measure in sim labels; optionally mark glare-heavy bricks `?`.
4. Sim realism still missing: D435 depth noise/holes, the user's actual ring light / lamp positions,
   the Franka arm.

Parallel work (not part of this log): Oklab colour-space experiment in `classifier/` and `colour_setups/`.
