# Physical brick classification

Hackathon submission: segment physical bricks with SAM3 and classify each one as **good** or **bad** with the frozen Approach 17 classifier. This package is independent of the repository's LEGO/RGB-D pose pipeline.

## Pipeline

1. SAM3 produces an individual whole-brick mask for each detected brick (up to ten in a frame).
2. Approach 5 computes 44 LAB color-histogram and eight texture features from the masked brick, then a trained linear SVM produces a score.
3. Frozen DINOv2-small produces patch-token features; mean, standard deviation and 90th percentile are reduced by PCA and passed to a second trained linear SVM.
4. The two SVM decision values are averaged. The **minus_two_binary** policy returns **good / green** for a fused score ≤ −2.0 and **bad / red** for a score > −2.0. There is no review category in this policy.

The score is a decision value, not a probability. A crack does not by itself define a bad brick: labels for this dataset are based on color, although the trained model uses color, texture, and learned visual features. This package does not calculate robot poses or command a robot.

## Included files

- `classify_phy_bricks.py` — classify one brick from an RGB image and a binary mask.
- `sam3_inference.py` — segment and classify bricks in a photo; save masks, annotations and JSON.
- `remote_server.py`, `remote_client.py` — live Mac/remote-computer viewer over SSH.
- `prepare_dataset_masks.py`, `evaluate_dataset.py`, `train_classifier.py` — prepare masks, evaluate photos and retrain separately.
- `checkpoint/` — the two trained SVM checkpoints, PCA, and training metadata.
- `validation_summary_minus_two.json` — saved in-sample photo results for the default policy.

The small trained classifiers are included. The SAM3 and DINOv2 model weights are **not** included and must be downloaded or supplied separately. SAM3 requires Hugging Face access to its gated model and authentication. The pinned models are [`facebook/sam3`](https://huggingface.co/facebook/sam3), revision `3c879f39826c281e95690f02c7821c4de09afae7`, and [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small), revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Use a cache with sufficient storage.

## Setup

From the repository root on the inference computer:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Classify_phy_bricks/requirements.txt
export HF_HOME=/path/to/large/cache/huggingface
hf auth login
```

The listed package versions match the environment used to load the saved checkpoints. Select a PyTorch build suitable for the machine's CPU/CUDA environment. After the first download, `--local-files-only` avoids network access. For existing local weights, use `--sam3-model /path/to/sam3` and `--encoder-path /path/to/dinov2_small`.

## Run on a photograph

```bash
python Classify_phy_bricks/sam3_inference.py \
  --image /path/to/brick_photo.png \
  --output-dir results/physical_bricks \
  --policy minus_two_binary
```

The output folder contains `annotated.png`, `predictions.json` and one binary mask per detected brick under `masks/`. The JSON includes the score, label, SAM3 confidence and image-coordinate box. A missing reliable mask produces no detection. SAM3 can miss or merge touching bricks; inspect the masks.

To classify an existing full-frame mask without running SAM3:

```bash
python Classify_phy_bricks/classify_phy_bricks.py \
  --image /path/to/frame.png --mask /path/to/brick_mask.png \
  --policy minus_two_binary \
  --output results/prediction.json --overlay results/annotated.png
```

## Live camera over SSH

Run the server and model on the inference computer. On the Mac, clone this branch or copy `Classify_phy_bricks/remote_client.py`, then install the lightweight camera dependencies:

```bash
python -m pip install -r Classify_phy_bricks/requirements-mac.txt
python Classify_phy_bricks/remote_client.py \
  --host YOUR_LAB_HOST --user YOUR_LAB_USER \
  --remote-repo /path/to/Hackathon-MarmonxFranks \
  --remote-python /path/to/env/bin/python \
  --remote-hf-home /path/to/large/cache/huggingface \
  --camera 1 --backend avfoundation --interval 3 \
  --policy minus_two_binary --save-frames
```

The SSH user must have access to the remote checkout and model cache. When the weights are already present, add `--remote-local-files-only`. Try `--camera 0` if necessary. Press **q** or **Esc** to quit. Frames, per-frame JSON and a CSV are saved in `live_results/<timestamp>/` on the Mac. The live pane and most recent prediction are different points in time.

## Evaluation

The dataset contains **258 photos of 10 physical bricks**: four good and six bad. On the saved photos and SAM3 masks, using the frozen classifier and −2 policy:

| Actual | Predicted good | Predicted bad |
|---|---:|---:|
| Good photos | 67 | 42 |
| Bad photos | 9 | 140 |

**207/258 photos (80.2%)** were correct. These physical bricks were used to train the saved model, so this is an **in-sample** result, not accuracy on new bricks or a live camera. See `validation_summary_minus_two.json` for per-brick counts.

A separate historical leave-one-physical-brick-out evaluation obtained **212/258 (82.2%)** image accuracy and **9/10** bricks correct after aggregating multiple photos. It used the original masks and a zero threshold, not this SAM3/−2 pipeline. The dataset is small; test the submitted policy on additional unseen bricks and live camera conditions before relying on it. The earlier review-band results are retained in `validation_summary_review_band.json` for reference.
