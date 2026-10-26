# Normal pipeline with standalone classification

This pipeline changes **only the final colour/good-bad classifier**. It uses
normal `top_layer_bricks.py` geometry (the same detector as `run_pipeline.sh`),
then applies the standalone `classifier/classify_top_brick_quality.py` with the
learned colour model to its fitted footprints. There is no lattice recovery or
dimension change in this pipeline itself.

## New capture

```bash
conda activate sam3
./run_pipeline_classifier.sh
```

Capture uses `d435_settings.json`: Enter saves, q finishes. The latest saved frame
is processed. The detector and classifier save previews without opening windows.

## Existing recording

```bash
./run_pipeline_classifier.sh \
  --recording data_collection/data/extracted/data4/20261024_095657 \
  --frame 20
```

Results are isolated in `data/pipeline_classifier_results/run_<timestamp>/`:

- `00_capture/`: new capture when requested.
- `00_normal_geometry/<recording>/<frame>/`: original normal-pipeline results,
  including its original classification for comparison.
- `01_box_adapter/capture/<recording>/<frame>/top_boxes.json`: identical fitted
  footprints expressed in the standalone classifier's input format.
- `02_classification/capture/<recording>/<frame>/`: standalone classification,
  colour distances/confidence, `quality_result.png` and `quality_comparison.png`.
- `03_final/capture/<recording>/<frame>/`: reclassified `bricks.json`,
  `poses.csv`, instance/semantic masks, `overlay.png` and six-panel `preview.png`.

The final preview retains the normal pipeline's first five panels and replaces
only panel 6. All final boxes have centre dots. XYZ, rotations, corners, instance
IDs and instance masks are preserved; pose CSV colour/quality columns change.
Coordinates remain camera-color optical coordinates, in metres. The default
physical prior remains 30 x 15 x 10 mm, configurable with `--brick-size-mm`.

The original integrated classifier still executes in the unmodified geometry
stage, but its final predictions are replaced by the standalone classifier. This
keeps the detector exactly reproducible and saves the original result for an A/B
comparison. Early SAM filtering is unchanged, including its original colour
settings. Classification cannot restore a brick dropped by earlier filters.

## Reclassify an existing normal result without rerunning SAM3

```bash
python pipeline_classifier/reclassify.py \
  data/top_layer_bricks/20261024_204741/000000 \
  --out data/pipeline_classifier_results/reclassification
```

Use a fresh `--out` path to keep experiments separate. The source RGB recording
referenced by bricks.json must still exist. Run from the repository root.
Standalone colour parameters are available through this command's `--help`.

## Learned colour model

By default the launcher classifies with `classifier/colour_model.json`, trained
on the single-brick dataset in `data/brick_dataset_extracted`. Retrain after
adding or correcting samples:

```bash
python classifier/colour_model.py --dataset data/brick_dataset_extracted \
  --out classifier/colour_model.json
```

Training prints accuracy for a random split (optimistic) and with each lighting
condition held out (pessimistic), with confusion matrices. Features are hue,
lightness and chroma relative to the table. For frames with at least 10 boxes,
a bounded per-frame hue shift and lightness/chroma scale is fitted to absorb
auto-exposure drift; it is recorded as `scene_feature_transform`. Boxes below
the `threshold` probability (default 0.7) are marked low confidence.

Use `--colour-model none` for the fixed Lab prototypes, or pass
`--colour-model classifier/colour_model.json` to `reclassify.py` to apply the
model to an existing result.

## Run every capture and compare the pipelines

```bash
./run_all_captures.sh --dry-run    # list the captures that will be processed
./run_all_captures.sh              # run them all
```

This finds every saved D435 capture (`data/captures`,
`data/manual_pipeline/captures` and the captures stored inside earlier
pipeline, modular and temporal result folders, deduplicated through symlinks)
and, for the latest saved frame of each, runs `run_pipeline_classifier.py` and
`run_modular_pipeline.py`. The geometry stage of the former is exactly
`run_pipeline.sh`'s result. Everything goes to
`data/all_captures_results/batch_<timestamp>/`:

| Path | Content |
|---|---|
| `index.html` | table of bricks / good / bad per pipeline, then every comparison; open in a browser |
| `comparisons/<capture>.png` | RGB / `run_pipeline.sh` / `run_modular_pipeline.sh` / `run_pipeline_classifier.sh`, all drawn the same way |
| `<capture>/run_*/` | normal `run_pipeline_classifier` result |
| `<capture>/modular/run_*/` | normal `run_modular_pipeline` result |
| `summary.csv`, `summary.json` | counts and errors for all three pipelines |
| `logs/` | full output of every run |

In the comparisons, green outline = good, red = bad; labels R red, DR dark
red, O dark orange, N medium nougat, B reddish brown; `?` = low confidence.
A failed pipeline shows its error. The summary and page are rewritten after
every capture, so an interrupted batch keeps its results.

Options: `--limit N` (quick test), `--all-frames` (every saved frame),
`--no-modular`, `--colour-model none` (fixed prototypes for an A/B run),
`--roots <globs>` (other recording folders, e.g.
`data_collection/data/extracted/data3`), `--detector-args ...` (last).

Rebuild the comparisons and page of an existing batch without rerunning:

```bash
python compare_pipeline_results.py data/all_captures_results/batch_<timestamp>
```

Additional geometry options may be passed last to the launcher:

```bash
./run_pipeline_classifier.sh --detector-args --threshold 0.4
```

A classification change does not demonstrate an accuracy improvement. Inspect
both original and standalone outputs on the same frames. The adapter checks that
all geometric fields and pose CSV geometric values remain identical.
