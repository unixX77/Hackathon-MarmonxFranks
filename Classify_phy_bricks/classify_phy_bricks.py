#!/usr/bin/env python3
"""Standalone frozen Approach 17 physical-brick quality classifier.

The caller supplies an RGB frame and a full-frame binary mask for one brick.
SAM3 segmentation is intentionally external to this classifier package.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "checkpoint"
DINO_MODEL = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
POLICIES = ("review_band_175_120", "minus_two_binary", "zero_binary", "baseline_margin")
COLORS = {"good": (0, 180, 0), "bad": (0, 0, 225), "uncertain": (0, 190, 255)}


def decide(score: float, policy: str = "minus_two_binary") -> str:
    """Convert the fused SVM decision value to a label using a saved policy."""
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("Classifier score must be finite")
    if policy == "review_band_175_120":
        return "good" if score <= -1.75 else "bad" if score > -1.20 else "uncertain"
    if policy == "minus_two_binary":
        return "good" if score <= -2.0 else "bad"
    if policy == "zero_binary":
        return "good" if score < 0 else "bad"
    if policy == "baseline_margin":
        return "uncertain" if abs(score) < .3 else "bad" if score >= 0 else "good"
    raise ValueError(f"Unknown policy: {policy}")


def normalize_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a height x width x 3 uint8 RGB array")
    if mask.ndim != 2 or mask.shape != rgb.shape[:2]:
        raise ValueError("mask must be a 2D full-frame mask with the same width and height as rgb")
    if not np.isfinite(mask).all():
        raise ValueError("mask contains nonfinite values")
    selected = (mask > 0).astype(np.uint8)
    if not selected.any():
        raise ValueError("mask contains no brick pixels")
    return selected


def appearance_features(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exactly match the original 44 LAB histogram + 8 texture features."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    usable = mask.astype(bool) & (value > 18) & (value < 248)
    if usable.sum() < 500:
        raise ValueError("Too few usable masked material pixels (need at least 500)")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    pixels = lab[usable]
    parts = []
    for channel, bins, limits in ((0, 12, (0, 100)), (1, 16, (-80, 80)), (2, 16, (-80, 80))):
        histogram, _ = np.histogram(pixels[:, channel], bins=bins, range=limits)
        parts.append(histogram.astype(float) / max(1, histogram.sum()))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)[usable]
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))[usable]
    texture = np.r_[np.percentile(gradient, [25, 50, 75, 90]),
                    np.percentile(laplacian, [50, 75, 90]), float(np.std(gray[usable]))]
    return np.r_[np.concatenate(parts), texture]


def masked_crop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match the original masked DINO crop and median-color background."""
    ys, xs = np.nonzero(mask)
    margin = round(.04 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    crop = rgb[y0:y1, x0:x1].copy()
    selected = mask[y0:y1, x0:x1].astype(bool)
    neutral = np.median(crop[selected], axis=0).astype(np.uint8)
    crop[~selected] = neutral
    return crop


class PhysicalBrickClassifier:
    """Load the frozen Approach 17 SVMs and frozen DINOv2 encoder once."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, encoder_path: str | Path | None = None,
                 device: str = "auto", local_files_only: bool = False):
        self.checkpoint = Path(checkpoint)
        self.approach5 = joblib.load(self.checkpoint / "approach5_svm.joblib")
        self.dino_svm = joblib.load(self.checkpoint / "dinov2_patch_svm.joblib")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        source = str(encoder_path) if encoder_path is not None else DINO_MODEL
        revision = None if encoder_path is not None else DINO_REVISION
        self.processor = AutoImageProcessor.from_pretrained(source, revision=revision,
                                                             local_files_only=local_files_only)
        self.encoder = AutoModel.from_pretrained(source, revision=revision,
                                                  local_files_only=local_files_only).to(self.device).eval()

    @torch.inference_mode()
    def dino_features(self, crop: np.ndarray) -> np.ndarray:
        pixels = self.processor(images=[crop], return_tensors="pt")["pixel_values"]
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self.device.startswith("cuda")):
            hidden = self.encoder(pixel_values=pixels.to(self.device)).last_hidden_state
        patches = hidden.float().cpu().numpy()[0, 1:]
        return np.r_[patches.mean(0), patches.std(0), np.percentile(patches, 90, axis=0)]

    def classify(self, rgb: np.ndarray, mask: np.ndarray,
                 policy: str = "minus_two_binary") -> dict:
        mask = normalize_mask(rgb, mask)
        a5 = float(self.approach5.decision_function(appearance_features(rgb, mask)[None])[0])
        dino = float(self.dino_svm.decision_function(self.dino_features(masked_crop(rgb, mask))[None])[0])
        score = (a5 + dino) / 2
        return {"label": decide(score, policy), "score_bad": score,
                "approach5_score": a5, "dinov2_score": dino, "threshold_policy": policy,
                "mask_area_px": int(mask.sum())}


def read_image_and_mask(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, normalize_mask(rgb, mask)


def save_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray, prediction: dict) -> None:
    """Draw a thin contour; colors denote model decisions, not ground truth."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = COLORS[prediction["label"]]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, color, 2, cv2.LINE_AA)
    x, y, _, _ = cv2.boundingRect(cv2.findNonZero(mask))
    text = f"{prediction['label'].upper()} {prediction['score_bad']:+.2f}"
    cv2.putText(bgr, text, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                .65, color, 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Could not write overlay: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True, help="binary mask for one brick in image coordinates")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder-path", default=os.environ.get("DINO_V2_PATH"),
                        help="local DINOv2 directory; otherwise fetch pinned Hugging Face revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--policy", choices=POLICIES, default="minus_two_binary")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--overlay", type=Path, help="optional annotated image output path")
    args = parser.parse_args()
    rgb, mask = read_image_and_mask(args.image, args.mask)
    classifier = PhysicalBrickClassifier(args.checkpoint, args.encoder_path,
                                         args.device, args.local_files_only)
    prediction = classifier.classify(rgb, mask, args.policy)
    result = {"image": str(args.image), "mask": str(args.mask), **prediction}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.overlay:
        save_overlay(args.overlay, rgb, mask, prediction)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
