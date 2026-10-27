# Classify physical bricks — Approach 17

This folder packages the frozen physical-brick classifier used in the live
baseline. SAM3 segments each visible brick; Approach 17 classifies each mask.
It is separate from the RGB-D LEGO top-layer and robot-pose pipelines.

## Architecture

For each full-frame brick mask, Approach 17 computes two decision scores:

1. **Approach 5:** 44 LAB color histogram values plus eight image texture
   values, standardized and passed to a trained linear SVM.
2. **DINOv2:** the masked crop with its background filled by the median brick
   color is passed through the frozen DINOv2-small encoder. The mean, standard
   deviation, and 90th percentile of patch tokens form 1,152 features, followed
   by the saved PCA, scaler, and trained linear SVM.

The fused `score_bad` is the equal average of the two SVM decision values.
The current live baseline uses `review_band_175_120`:

| Fused score | Label | Viewer color |
|---|---|---|
| ≤ −1.75 | good | green |
| −1.75 < score ≤ −1.20 | uncertain / review | amber |
| > −1.20 | bad | red |

Scores are SVM margins, not probabilities. The `baseline_margin`,
`zero_binary`, and `minus_two_binary` historical policies are also selectable.
The classifier does not calculate robot poses or issue robot commands.

## Files and checkpoints

- `classify_phy_bricks.py`: inference from an RGB image and a supplied binary
  mask, with a callable Python API and optional image overlay.
- `sam3_inference.py`: SAM3 detection, classification, masks, and annotated
  output for a single image; `--selection multi` supports up to ten instances.
- `prepare_dataset_masks.py` and `evaluate_dataset.py`: reproduce an offline
  run on the labeled `Brick_N_good` / `Brick_N_bad` photo folders.
- `train_classifier.py`: extract the two feature sets and retrain both SVM
  branches, with optional leave-one-physical-brick-out evaluation.
- `remote_server.py` and `remote_client.py`: persistent SSH inference on the
  lab computer with a two-panel Mac camera viewer.
- `checkpoint/`: the two small, trained Approach 17 SVMs and original training
  metadata. These checkpoint files are included in GitHub.

The large encoders are **not** included. They are loaded from a local cache or
downloaded once from Hugging Face:

| Encoder | Pinned model and revision |
|---|---|
| DINOv2 | `facebook/dinov2-small`, `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| SAM3 | `facebook/sam3`, `3c879f39826c281e95690f02c7821c4de09afae7` |

The pinned DINOv2 `model.safetensors` is about 85 MB and has SHA-256
`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
SAM3 is gated; the user running it needs the required Hugging Face access and
authentication. See [SAM3](https://huggingface.co/facebook/sam3) and
[DINOv2-small](https://huggingface.co/facebook/dinov2-small).

## Environment

Python 3.12 and the package versions in `requirements.txt` match the tested
environment, including the scikit-learn version used to load the joblib
checkpoints. Install a PyTorch build matching the computer's CPU/CUDA setup;
the tested lab environment used PyTorch 2.10.0 + CUDA 12.8. Configure a cache
with enough space before the first model download:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Classify_phy_bricks/requirements.txt
export HF_HOME=/path/to/large/cache/huggingface
```

The code chooses CUDA when available; `--device cpu` or `--device cuda` can
override it. If the exact weights already exist locally, pass `--sam3-model`
and `--encoder-path` as appropriate, and optionally `--local-files-only`.

## Classify a photo using SAM3

Run from the repository root:

```bash
python Classify_phy_bricks/sam3_inference.py \
  --image /path/to/brick_photo.png \
  --output-dir results/physical_bricks \
  --policy review_band_175_120
```

Results are `annotated.png`, `predictions.json`, and one binary PNG mask per
detected brick in `masks/`. Each JSON detection contains its score, label,
component scores, SAM3 confidence, and image-coordinate box. With no reliable
mask the result has zero detections; it does not force a good/bad label.
`--selection multi` is the default; `--selection dominant` reproduces the
single-centered-brick selection used by the offline photo evaluation. SAM3
may miss or merge touching bricks, so inspect the contour before relying on
the result.

## Classify an existing mask

The mask must have the same width and height as the RGB image; nonzero pixels
belong to **one** brick. Other segmentation systems can call the classifier
without loading SAM3:

```bash
python Classify_phy_bricks/classify_phy_bricks.py \
  --image /path/to/frame.png --mask /path/to/brick_mask.png \
  --output results/prediction.json --overlay results/annotated.png
```

```python
from Classify_phy_bricks.classify_phy_bricks import PhysicalBrickClassifier

model = PhysicalBrickClassifier(encoder_path="/path/to/dinov2_small")
prediction = model.classify(rgb_uint8, one_brick_mask)
print(prediction["label"], prediction["score_bad"])
```

Create the model once and call `classify` for each independently segmented
brick. Invalid or empty masks raise an error instead of being labeled bad.

## Reproduce the 258-photo offline run

The dataset is not included in the repository. Arrange the same labeled images
under `images/Brick_1_good/`, ..., `images/Brick_10_bad/`. To create masks:

```bash
python Classify_phy_bricks/prepare_dataset_masks.py \
  --images /path/to/images --output-dir results/sam3_masks
```

This uses the original **dominant** mask-selection heuristic and writes
`results/sam3_masks/Brick_N_label/masks/<image-stem>.png`. A manifest records
failures; missing masks are not silently labeled. Then run:

```bash
python Classify_phy_bricks/evaluate_dataset.py \
  --images /path/to/images --masks results/sam3_masks \
  --output-dir results/approach17_evaluation \
  --policy review_band_175_120
```

You may instead supply an existing matching mask folder. Add
`--save-overlays` to store classified images. `predictions.csv` contains each
score; `summary.json` reports good, bad, and uncertain counts by folder.

## Retrain on a labeled dataset

The following trains **new** SVM checkpoints in a separate output directory.
It uses the supplied masks and a frozen DINOv2 encoder. The feature cache can
be large, so put it on a spacious disk. Do not use the existing `checkpoint/`
folder as the output if you want to preserve the supplied baseline.

```bash
python Classify_phy_bricks/train_classifier.py \
  --images /path/to/images --masks results/sam3_masks \
  --output-dir results/retrained_approach17 \
  --features-cache /path/to/large/cache/brick_features.npz \
  --evaluate-heldout
```

Each physical brick must have its own `Brick_N_good` or `Brick_N_bad` folder.
The optional held-out evaluation leaves one entire brick out and selects SVM
regularization on training bricks. For comparison with the historical 212/258
evaluation, it reports a binary threshold at zero; it is not an evaluation of
the review band. New masks or photos produce new model weights, so retraining
is not expected to reproduce the supplied frozen checkpoint. Recalibrate and
test the decision threshold on separate data for the new model and camera.
Use `--checkpoint results/retrained_approach17` with
`classify_phy_bricks.py` or `evaluate_dataset.py` to load it.

## Mac camera → photolab → live viewer

The model and weights remain on the remote machine. The Mac needs only Python,
OpenCV, NumPy, SSH, and its camera. Clone this repository on the Mac or copy
`remote_client.py` from the remote checkout. Run the following from the Mac,
replacing the example remote host, user, and paths with your own:

```bash
python -m pip install numpy opencv-python

python Classify_phy_bricks/remote_client.py \
  --host your-lab-host --user your-lab-user \
  --camera 1 --backend avfoundation --interval 3 --save-frames \
  --remote-repo '/path/on/lab/Hackathon-MarmonxFranks' \
  --remote-python '/path/on/lab/env/bin/python' \
  --remote-hf-home '/path/on/lab/model_cache' --remote-local-files-only
```

Try `--camera 0` if camera 1 is unavailable. Press `q` or Esc to quit. The
Mac saves frames, `predictions.csv`, and per-frame detection JSON under
`live_results/<timestamp>/`. For a single known brick, optionally use
`--truth good` or `--truth bad`; omit it for mixed bricks. Omit
`--remote-local-files-only` if the remote computer needs to download the
models after Hugging Face authentication.

## Validation and limits

Using the saved checkpoint and the 258 existing SAM3 masks, the packaged code
reproduced the original scores to within 5e-7 and gave **194 correct out of
215 decided photos**, with **43 uncertain**. These are the same physical
bricks used to fit the saved model, and the review band was selected on these
scores: this is an in-sample reproduction, not new-brick or live accuracy.

The historical leave-one-physical-brick-out result was **212/258 (82.2%)**
using the original automatic masks and binary decision rule. It is not an
evaluation of the current SAM3 review policy. See `validation_summary.json`
for the reproduced offline per-folder counts. New camera conditions and
new bricks require separate evaluation.
