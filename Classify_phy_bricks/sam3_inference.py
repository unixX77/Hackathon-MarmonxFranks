#!/usr/bin/env python3
"""SAM3 whole-brick masks -> standalone Approach 17 for one RGB image."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from classify_phy_bricks import COLORS, POLICIES, PhysicalBrickClassifier

SAM3_MODEL = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


class BrickSegmenter:
    """Reproduce the live whole-brick SAM3 selection and geometry filters."""

    def __init__(self, model_path: str, device: str = "auto", local_files_only: bool = False):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        revision = None if Path(model_path).is_dir() else SAM3_REVISION
        self.processor = Sam3Processor.from_pretrained(model_path, revision=revision,
                                                        local_files_only=local_files_only)
        self.model = Sam3Model.from_pretrained(model_path, revision=revision,
                                               torch_dtype=self.dtype,
                                               local_files_only=local_files_only).to(self.device).eval()

    @torch.inference_mode()
    def segment(self, rgb: np.ndarray, prompt: str = "brick", threshold: float = .35,
                max_bricks: int = 10, selection: str = "multi") -> list[dict]:
        if selection not in ("multi", "dominant"):
            raise ValueError(f"Unknown selection: {selection}")
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        output = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            output, threshold=threshold, mask_threshold=.5, target_sizes=[image.size[::-1]])[0]
        masks = result["masks"].cpu().numpy().astype(bool)
        scores = result["scores"].float().cpu().numpy()
        if len(scores) == 0:
            return []
        # Match the source pipeline's descending-confidence overlap filtering.
        covered = np.zeros(rgb.shape[:2], bool)
        candidates = []
        for index in np.argsort(-scores):
            mask = masks[index]
            area = int(mask.sum())
            if not area or np.logical_and(mask, covered).sum() / area > .5:
                continue
            covered |= mask
            if area < 1200:
                continue
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            (_, _), (w, h), _ = cv2.minAreaRect(contour)
            if area / max(w * h, 1.0) < .55:
                continue
            candidates.append((mask, float(scores[index])))
        # The existing segmenter writes non-overlapping instance IDs in score
        # order before selecting the largest connected component of each mask.
        occupied = np.zeros(rgb.shape[:2], bool)
        selected = []
        height, width = rgb.shape[:2]
        for mask, confidence in candidates:
            remaining = (mask & ~occupied).astype(np.uint8)
            occupied |= remaining.astype(bool)
            count, components, stats, _ = cv2.connectedComponentsWithStats(remaining, 8)
            if count <= 1:
                continue
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            part = (components == component).astype(np.uint8)
            area = int(part.sum())
            area_fraction = area / (height * width)
            x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(part))
            centre_distance = np.hypot((x + bw / 2 - width / 2) / width,
                                       (y + bh / 2 - height / 2) / height)
            # Match existing record border detection before component selection.
            ys_all, xs_all = np.nonzero(remaining)
            border = bool(xs_all.min() == 0 or ys_all.min() == 0 or
                          xs_all.max() == width - 1 or ys_all.max() == height - 1)
            rounded_score = round(confidence, 3)
            if selection == "dominant":
                if not .02 <= area_fraction <= .75 or centre_distance > .48:
                    continue
                if border and not (area_fraction >= .08 and centre_distance <= .35 and rounded_score >= .65):
                    continue
                rank = area * (1 - centre_distance) * max(.1, rounded_score)
            else:
                if not .008 <= area_fraction <= .75 or min(bw, bh) < 35:
                    continue
                if border:
                    if not (area_fraction >= .05 and centre_distance <= .55 and rounded_score >= .65):
                        continue
                elif centre_distance > .71:
                    continue
                rank = area * max(.1, rounded_score)
            selected.append({"mask": part, "sam_score": rounded_score, "touches_border": border,
                             "box_xyxy": [x, y, x + bw - 1, y + bh - 1],
                             "centroid_xy": [round(float(xs_all.mean()), 1), round(float(ys_all.mean()), 1)],
                             "rank": rank})
        selected.sort(key=lambda item: item["rank"], reverse=True)
        selected = selected[:1 if selection == "dominant" else max_bricks]
        selected.sort(key=lambda item: (item["centroid_xy"][1], item["centroid_xy"][0]))
        return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3-model", default=os.environ.get("SAM3_PATH", SAM3_MODEL))
    parser.add_argument("--encoder-path", default=os.environ.get("DINO_V2_PATH"))
    parser.add_argument("--prompt", default="brick")
    parser.add_argument("--sam-threshold", type=float, default=.35)
    parser.add_argument("--max-bricks", type=int, default=10)
    parser.add_argument("--selection", choices=("multi", "dominant"), default="multi",
                        help="multi for the live viewer; dominant for the single-brick offline dataset")
    parser.add_argument("--policy", choices=POLICIES, default="review_band_175_120")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_bricks <= 10:
        parser.error("--max-bricks must be between 1 and 10")
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    segmenter = BrickSegmenter(args.sam3_model, args.device, args.local_files_only)
    classifier = PhysicalBrickClassifier(encoder_path=args.encoder_path, device=args.device,
                                         local_files_only=args.local_files_only)
    detections = segmenter.segment(rgb, args.prompt, args.sam_threshold, args.max_bricks, args.selection)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    annotated = frame.copy()
    records = []
    for index, detection in enumerate(detections, 1):
        mask = detection.pop("mask")
        detection.pop("rank")
        prediction = classifier.classify(rgb, mask, args.policy)
        mask_name = f"brick_{index:03d}.png"
        if not cv2.imwrite(str(mask_dir / mask_name), mask * 255):
            raise OSError(f"Could not save mask: {mask_name}")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = COLORS[prediction["label"]]
        cv2.drawContours(annotated, contours, -1, color, 2, cv2.LINE_AA)
        x, y = detection["box_xyxy"][:2]
        text = f"#{index} {prediction['label'].upper()} {prediction['score_bad']:+.2f}"
        cv2.putText(annotated, text, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, color, 2, cv2.LINE_AA)
        records.append({"id": index, **detection, **prediction, "mask_file": f"masks/{mask_name}"})
    if not cv2.imwrite(str(args.output_dir / "annotated.png"), annotated):
        raise OSError("Could not save annotated image")
    report = {"image": str(args.image), "selection": args.selection, "policy": args.policy, "brick_count": len(records),
              "status": "ok" if records else "no_reliable_sam3_brick", "detections": records}
    (args.output_dir / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
