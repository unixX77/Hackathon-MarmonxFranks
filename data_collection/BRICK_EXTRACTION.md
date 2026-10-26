# Labeled brick extraction

Run in the sam3 environment:

```bash
conda activate sam3
python data_collection/extract_brick_dataset.py
```

Inputs: `data/brick_dataset/{good,bad}/{color}/*.{png,jpg,jpeg}`.
Outputs: **`data/brick_dataset_extracted/`**, preserving quality/color labels:

- `crops/good/red/...png`: tight original RGB crops, with 12% padding.
- `masked/good/red/...png`: the same crops with outside-mask pixels set to neutral gray.
- `masks/good/red/...png`: full-resolution binary masks in source-image coordinates.
- `crop_masks/good/red/...png`: binary masks matching each crop.
- `metadata/good/red/...json`: source image, folder labels, crop bounds, SAM candidate
  scores/shapes and review status.
- `review/{good,bad}/{color}/page_XX.jpg`: all samples in contact sheets, with source
  image/selected outline and masked crop side by side.
- `manifest.json`, `summary.json`: full index and counts.

`--resume` skips metadata-complete samples and rebuilds the manifest/review pages;
use a new `--output PATH` if model/threshold settings or source images change.
`--limit N` is useful for testing. Originals are never modified.

SAM3 uses the text "lego brick"; the existing colour classifier is deliberately
not used to choose masks or labels. Shape bounds reject huge background masks,
small fragments and implausible aspect ratios. The best remaining candidate is
ranked by SAM score times rectangularity. Multiple candidates, low confidence,
border contact and missing masks are flagged. Automatic candidates are not
certified labels: **every sample still needs visual crop review before training**.
A no-mask sample has metadata but no crop and must be corrected/reprocessed or
excluded explicitly; the full input is never silently used as the brick crop.

Labels come from the folders, not SAM3. Keep source filenames and metadata for
provenance. Some original red filenames encode the capture session; later flat
filenames encode click time, not a verified session identifier. Define capture/
lighting groups before making train/validation/test splits, and keep near-duplicate
images together. Evaluate on real pipeline stack crops in addition to isolated
brick photographs.
