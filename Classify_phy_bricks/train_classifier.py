#!/usr/bin/env python3
"""Retrain the two Approach 17 SVM branches using labeled brick masks.

Each Brick_N_good/bad folder represents one physical brick. Feature scaling,
PCA, and C selection stay inside the physical-brick training folds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from transformers import AutoImageProcessor, AutoModel

from classify_phy_bricks import (DINO_MODEL, DINO_REVISION, appearance_features,
                                 masked_crop, read_image_and_mask)
from evaluate_dataset import EXTENSIONS, FOLDER

C_GRID = (.01, .03, .1, .3, 1.0, 3.0)
FEATURE_VERSION = "approach17-mask-features-v1"


def discover(images: Path, masks: Path) -> list[dict]:
    folders = sorted((p for p in images.iterdir() if p.is_dir() and FOLDER.match(p.name)),
                     key=lambda p: int(FOLDER.match(p.name).group(1)))
    rows = []
    for folder in folders:
        label = FOLDER.match(folder.name).group(2).lower()
        for image in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS):
            roots = (masks / folder.name / "masks", masks / folder.name)
            candidates = [root / name for root in roots for name in (image.name, image.stem + ".png")]
            mask = next((p for p in candidates if p.is_file()), None)
            if mask is None:
                raise FileNotFoundError(f"No mask for {image}")
            rows.append({"image": image, "mask": mask, "brick": folder.name,
                         "label": int(label == "bad"), "truth": label})
    if not rows:
        raise ValueError("No labeled Brick_N_good / Brick_N_bad images found")
    return rows


def fingerprints(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    paths, signatures = [], []
    for row in rows:
        image, mask = row["image"], row["mask"]
        a, b = image.stat(), mask.stat()
        paths.append(str(image.resolve()))
        signatures.append(f"{a.st_mtime_ns}:{a.st_size}:{b.st_mtime_ns}:{b.st_size}")
    return np.asarray(paths), np.asarray(signatures)


def extract(rows: list[dict], encoder_path: str | None, device: str,
            local_only: bool, cache: Path | None) -> tuple[np.ndarray, np.ndarray]:
    paths, signatures = fingerprints(rows)
    if encoder_path:
        source_dir = Path(encoder_path)
        if not source_dir.is_dir():
            raise ValueError(f"Local encoder directory does not exist: {source_dir}")
        source_parts = [str(source_dir.resolve())]
        for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
            file = source_dir / name
            if file.exists():
                stat = file.stat(); source_parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
        source_id = "|".join(source_parts)
    else:
        source_id = f"{DINO_MODEL}@{DINO_REVISION}"
    if cache and cache.is_file():
        with np.load(cache) as saved:
            if (str(saved["version"]) == FEATURE_VERSION and str(saved["encoder_source"]) == source_id
                    and np.array_equal(saved["paths"], paths)
                    and np.array_equal(saved["signatures"], signatures)):
                print(f"Using validated feature cache {cache}", flush=True)
                return saved["approach5"], saved["dino"]
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    actual_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if actual_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    source, revision = (encoder_path, None) if encoder_path else (DINO_MODEL, DINO_REVISION)
    processor = AutoImageProcessor.from_pretrained(source, revision=revision, local_files_only=local_only)
    encoder = AutoModel.from_pretrained(source, revision=revision,
                                         local_files_only=local_only).to(actual_device).eval()
    a5, dino = [], []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            rgb, mask = read_image_and_mask(row["image"], row["mask"])
            a5.append(appearance_features(rgb, mask))
            pixels = processor(images=[masked_crop(rgb, mask)], return_tensors="pt")["pixel_values"]
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=actual_device.startswith("cuda")):
                hidden = encoder(pixel_values=pixels.to(actual_device)).last_hidden_state
            patches = hidden.float().cpu().numpy()[0, 1:]
            dino.append(np.r_[patches.mean(0), patches.std(0), np.percentile(patches, 90, axis=0)])
            if i % 25 == 0 or i == len(rows):
                print(f"Extracted features: {i}/{len(rows)}", flush=True)
    a5, dino = np.asarray(a5), np.asarray(dino)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, approach5=a5, dino=dino, paths=paths, signatures=signatures,
                            version=FEATURE_VERSION, encoder_source=source_id)
    return a5, dino


def fit_branch(x: np.ndarray, y: np.ndarray, groups: np.ndarray, indices: np.ndarray,
               pca_components: int | None):
    steps = ([PCA(n_components=pca_components, whiten=True, random_state=7)] if pca_components else [])
    model = make_pipeline(*steps, StandardScaler(), SVC(kernel="linear", class_weight="balanced"))
    train_x, train_y, train_groups = x[indices], y[indices], groups[indices]
    folds = list(LeaveOneGroupOut().split(train_x, train_y, train_groups))
    search = GridSearchCV(model, {"svc__C": C_GRID}, scoring="accuracy", cv=folds,
                          n_jobs=1, error_score="raise")
    search.fit(train_x, train_y)
    return search.best_estimator_, float(search.best_params_["svc__C"])


def held_out(a5: np.ndarray, dino: np.ndarray, rows: list[dict], output: Path) -> dict:
    y = np.asarray([r["label"] for r in rows]); groups = np.asarray([r["brick"] for r in rows])
    scores = np.empty(len(rows)); selected = []
    for train, test in LeaveOneGroupOut().split(a5, y, groups):
        a5_model, a5_c = fit_branch(a5, y, groups, train, None)
        dino_model, dino_c = fit_branch(dino, y, groups, train, 32)
        scores[test] = (a5_model.decision_function(a5[test]) + dino_model.decision_function(dino[test])) / 2
        selected.append({"held_out_brick": groups[test][0], "approach5_C": a5_c, "dinov2_C": dino_c})
        print(f"Held out {groups[test][0]}: {int(np.sum((scores[test] >= 0) == y[test]))}/{len(test)}", flush=True)
    pred = scores >= 0
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    brick_rows = []
    for group in np.unique(groups):
        use = groups == group; score = float(np.mean(scores[use])); truth = int(y[use][0])
        brick_rows.append({"brick_id": group, "truth": "bad" if truth else "good", "images": int(use.sum()),
                           "correct_images": int(np.sum(pred[use] == y[use])), "mean_score_bad": score,
                           "prediction": "bad" if score >= 0 else "good"})
    result = {"method": "leave-one-physical-brick-out; nested group-C selection; historical binary threshold zero",
              "n_images": len(rows), "n_bricks": len(np.unique(groups)), "correct_images": int(np.sum(pred == y)),
              "image_accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "good_recall": float(matrix[0, 0] / matrix[0].sum()), "bad_recall": float(matrix[1, 1] / matrix[1].sum()),
              "confusion_matrix_good_bad": matrix.tolist(),
              "brick_correct": sum(r["truth"] == r["prediction"] for r in brick_rows)}
    def save(path, values):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    save(output / "held_out_image_predictions.csv", [
        {"brick_id": row["brick"], "image": row["image"].name, "truth": row["truth"],
         "score_bad": float(scores[i]), "prediction": "bad" if pred[i] else "good",
         "correct": int(pred[i] == y[i])} for i, row in enumerate(rows)])
    save(output / "held_out_brick_predictions.csv", brick_rows)
    save(output / "held_out_parameters.csv", selected)
    (output / "held_out_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for trained SVMs")
    parser.add_argument("--encoder-path", default=os.environ.get("DINO_V2_PATH"))
    parser.add_argument("--features-cache", type=Path, help="optional feature-cache file, preferably on large storage")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluate-heldout", action="store_true", help="also run nested physical-brick evaluation")
    args = parser.parse_args()
    rows = discover(args.images, args.masks)
    labels = np.asarray([r["label"] for r in rows]); groups = np.asarray([r["brick"] for r in rows])
    group_counts = {label: len(np.unique(groups[labels == label])) for label in (0, 1)}
    if min(group_counts.values()) < (3 if args.evaluate_heldout else 2):
        parser.error("Need at least two physical bricks per class for grouped training, or three per class with held-out evaluation")
    sizes = sorted((int(np.sum(groups == group)) for group in np.unique(groups)), reverse=True)
    removed = 2 if args.evaluate_heldout else 1
    smallest_training_fold = len(rows) - sum(sizes[:removed])
    if smallest_training_fold < 32:
        parser.error(f"PCA(32) needs at least 32 photos in every training fold; the smallest can have {smallest_training_fold}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a5, dino = extract(rows, args.encoder_path, args.device, args.local_files_only, args.features_cache)
    indices = np.arange(len(rows))
    with threadpool_limits(limits=1):
        evaluation = held_out(a5, dino, rows, args.output_dir) if args.evaluate_heldout else None
        a5_model, a5_c = fit_branch(a5, labels, groups, indices, None)
        dino_model, dino_c = fit_branch(dino, labels, groups, indices, 32)
    joblib.dump(a5_model, args.output_dir / "approach5_svm.joblib")
    joblib.dump(dino_model, args.output_dir / "dinov2_patch_svm.joblib")
    metadata = {"approach": "Approach 17: equal-weight fusion of trained A5 and DINOv2 SVMs",
                "n_training_images": len(rows), "n_physical_bricks": len(np.unique(groups)),
                "selected_C": {"approach5": a5_c, "dinov2": dino_c},
                "encoder": args.encoder_path or DINO_MODEL,
                "encoder_revision": None if args.encoder_path else DINO_REVISION,
                "warning": "Thresholds must be recalibrated and tested for this retrained checkpoint and camera."}
    (args.output_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.output_dir), "selection": metadata["selected_C"],
                      "held_out": evaluation}, indent=2))


if __name__ == "__main__":
    main()
