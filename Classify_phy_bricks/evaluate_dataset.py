#!/usr/bin/env python3
"""Evaluate the frozen classifier on Brick_N_good/bad folders and SAM3 masks."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from classify_phy_bricks import (DEFAULT_CHECKPOINT, POLICIES, PhysicalBrickClassifier,
                                 read_image_and_mask, save_overlay)

FOLDER = re.compile(r"^Brick_(\d+)_(good|bad)$", re.IGNORECASE)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="root containing Brick_N_good and Brick_N_bad image folders")
    parser.add_argument("--masks", type=Path, required=True,
                        help="root containing matching folder/masks/image.png SAM3 masks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--policy", choices=POLICIES, default="minus_two_binary")
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()
    folders = [(p, FOLDER.match(p.name)) for p in args.images.iterdir() if p.is_dir()]
    folders = sorted(((p, m) for p, m in folders if m), key=lambda item: int(item[1].group(1)))
    if not folders:
        raise ValueError("No Brick_N_good or Brick_N_bad folders found")
    jobs = []
    for folder, match in folders:
        for image in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS):
            roots = (args.masks / folder.name / "masks", args.masks / folder.name)
            options = [root / name for root in roots for name in (image.name, image.stem + ".png")]
            mask = next((candidate for candidate in options if candidate.is_file()), None)
            if mask is None:
                raise FileNotFoundError(f"No matching mask for {image}")
            jobs.append((folder.name, match.group(2).lower(), image, mask))
    if not jobs:
        raise ValueError("No images found in brick folders")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = PhysicalBrickClassifier(args.checkpoint, args.encoder_path, args.device, args.local_files_only)
    rows = []
    for index, (folder, truth, image, mask_path) in enumerate(jobs, 1):
        rgb, mask = read_image_and_mask(image, mask_path)
        result = model.classify(rgb, mask, args.policy)
        rows.append({"folder": folder, "image": image.name, "truth": truth, **result,
                     "correct": int(result["label"] == truth)})
        if args.save_overlays:
            save_overlay(args.output_dir / folder / image.name, rgb, mask, result)
        if index % 25 == 0 or index == len(jobs):
            print(f"Processed {index}/{len(jobs)}", flush=True)
    with (args.output_dir / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_folder = defaultdict(list)
    for row in rows:
        by_folder[row["folder"]].append(row)
    def summary(values):
        counts = Counter(row["label"] for row in values)
        decided = len(values) - counts["uncertain"]
        correct = sum(row["correct"] for row in values)
        return {"n_images": len(values), "correct": correct, "predictions": dict(counts),
                "decided": decided, "uncertain": counts["uncertain"],
                "accuracy_on_decided": correct / decided if decided else None,
                "correct_fraction_of_all": correct / len(values)}
    report = {"threshold_policy": args.policy, "overall": summary(rows),
              "folders": {folder: {"truth": values[0]["truth"], **summary(values)}
                          for folder, values in by_folder.items()},
              "evaluation_note": "This inference run does not retrain or hold out physical bricks."}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
