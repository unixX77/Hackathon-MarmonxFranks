#!/usr/bin/env python3
"""Create one dominant SAM3 mask per image for the labelled photo dataset."""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import cv2

from sam3_inference import BrickSegmenter, SAM3_MODEL

FOLDER = re.compile(r"^Brick_(\d+)_(good|bad)$", re.IGNORECASE)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3-model", default=os.environ.get("SAM3_PATH", SAM3_MODEL))
    parser.add_argument("--prompt", default="brick")
    parser.add_argument("--sam-threshold", type=float, default=.35)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit-per-folder", type=int, help="limit each folder for a smoke test")
    args = parser.parse_args()
    if args.limit_per_folder is not None and args.limit_per_folder < 1:
        parser.error("--limit-per-folder must be positive")
    folders = sorted((p for p in args.images.iterdir() if p.is_dir() and FOLDER.match(p.name)),
                     key=lambda p: int(FOLDER.match(p.name).group(1)))
    jobs = []
    for folder in folders:
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if args.limit_per_folder is not None:
            images = images[:args.limit_per_folder]
        if len({p.stem for p in images}) != len(images):
            raise ValueError(f"Duplicate image stems in {folder}; mask files would overwrite")
        jobs.extend((folder.name, image) for image in images)
    if not jobs:
        raise ValueError("No Brick_N_good or Brick_N_bad images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segmenter = BrickSegmenter(args.sam3_model, args.device, args.local_files_only)
    rows = []
    for index, (folder, image) in enumerate(jobs, 1):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = segmenter.segment(rgb, args.prompt, args.sam_threshold, selection="dominant")
        mask_file = ""
        if detections:
            mask_dir = args.output_dir / folder / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            path = mask_dir / (image.stem + ".png")
            if not cv2.imwrite(str(path), detections[0]["mask"] * 255):
                raise OSError(f"Could not save mask: {path}")
            mask_file = str(path)
        rows.append({"folder": folder, "image": image.name,
                     "status": "ok" if detections else "no_reliable_sam3_brick",
                     "sam_score": detections[0]["sam_score"] if detections else "",
                     "mask_file": mask_file})
        if index % 25 == 0 or index == len(jobs):
            print(f"Segmented {index}/{len(jobs)}", flush=True)
    with (args.output_dir / "mask_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    missing = sum(row["status"] != "ok" for row in rows)
    print(f"Found {len(rows) - missing}/{len(rows)} brick masks; manifest: {args.output_dir / 'mask_manifest.csv'}")
    if missing:
        raise SystemExit("Some masks were not found. Inspect the manifest before evaluation.")


if __name__ == "__main__":
    main()
