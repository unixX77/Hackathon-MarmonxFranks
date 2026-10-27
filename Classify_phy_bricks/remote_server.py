#!/usr/bin/env python3
"""Persistent JSON-lines server for SAM3 + Approach 17 live classification."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np

from classify_phy_bricks import COLORS, POLICIES, PhysicalBrickClassifier, decide
from sam3_inference import BrickSegmenter, SAM3_MODEL


def send(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def annotate(frame: np.ndarray, mask: np.ndarray, label: str, score: float, number: int) -> np.ndarray:
    """Same thin soft boundary used by the existing live SAM3 viewer."""
    color = np.asarray(COLORS[label], np.float32)
    selected = mask.astype(bool)
    edge = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, contours, -1, 255, 3, cv2.LINE_AA)
    soft = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.5)
    alpha = np.maximum(soft.astype(np.float32) / 255 * .45, edge.astype(np.float32) / 255 * .65)
    alpha[selected] = np.maximum(alpha[selected], .055)
    out = (frame.astype(np.float32) * (1 - alpha[..., None]) +
           color[None, None, :] * alpha[..., None]).clip(0, 255).astype(np.uint8)
    x, y, _, _ = cv2.boundingRect(cv2.findNonZero(mask))
    text = f"#{number} {label.upper()} {score:+.2f}"
    scale = max(.55, min(frame.shape[1] / 1600, .85))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    tx = min(x, max(0, frame.shape[1] - tw - 12))
    ty = y - 9 if y >= th + baseline + 13 else y + th + 10
    cv2.rectangle(out, (tx, ty - th - 5), (tx + tw + 8, ty + baseline + 4), (20, 20, 20), -1)
    cv2.putText(out, text, (tx + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLORS[label], 2, cv2.LINE_AA)
    return out


def process_request(request: dict, segmenter: BrickSegmenter,
                    classifier: PhysicalBrickClassifier, policy: str) -> dict:
    payload = base64.b64decode(request["image_b64"], validate=True)
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or max(frame.shape[:2]) > 4096:
        raise ValueError("Invalid JPEG or frame exceeds 4096 pixels")
    start = time.monotonic()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    segments = segmenter.segment(rgb)
    truth = request.get("truth_label") or ""
    result = {"type": "prediction", "frame_id": request.get("frame_id"),
              "captured_at": request.get("captured_at", ""), "pipeline": "sam3",
              "threshold_policy": policy, "truth_label": truth, "correct": "", "label": "",
              "score_bad": "", "approach5_score": "", "dinov2_score": "",
              "baseline_margin_label": "", "zero_binary_label": "", "minus_two_binary_label": "",
              "review_band_175_120_label": "", "status": "no_reliable_sam3_brick",
              "sam_score": "", "sam_touches_border": "", "mask_area_px": "",
              "brick_count": 0, "good_count": 0, "bad_count": 0, "uncertain_count": 0,
              "detections": [], "box_x": "", "box_y": "", "box_width": "", "box_height": "",
              "frame_width": frame.shape[1], "frame_height": frame.shape[0], "error": ""}
    annotated = frame.copy()
    for number, detection in enumerate(segments, 1):
        mask = detection["mask"]
        prediction = classifier.classify(rgb, mask, policy)
        score, label = prediction["score_bad"], prediction["label"]
        policy_labels = {name + "_label": decide(score, name) for name in POLICIES}
        record = {"id": number, **prediction, **policy_labels,
                  "sam_score": detection["sam_score"], "sam_touches_border": detection["touches_border"],
                  "box_xyxy": detection["box_xyxy"], "centroid_xy": detection["centroid_xy"]}
        result["detections"].append(record)
        result[label + "_count"] += 1
        annotated = annotate(annotated, mask, label, score, number)
    result["brick_count"] = len(result["detections"])
    if result["brick_count"]:
        result["status"] = "ok"
    if result["brick_count"] == 1:
        detection = result["detections"][0]
        x0, y0, x1, y1 = detection["box_xyxy"]
        for key in ("label", "score_bad", "approach5_score", "dinov2_score", "mask_area_px", "sam_score",
                    "sam_touches_border", "baseline_margin_label", "zero_binary_label",
                    "minus_two_binary_label", "review_band_175_120_label"):
            result[key] = detection[key]
        result.update({"box_x": x0, "box_y": y0, "box_width": x1 - x0 + 1,
                       "box_height": y1 - y0 + 1,
                       "correct": int(detection["label"] == truth) if truth in ("good", "bad") else ""})
    if not result["brick_count"]:
        cv2.putText(annotated, "NO RELIABLE SAM3 BRICK", (25, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 190, 255), 2, cv2.LINE_AA)
    result["inference_seconds"] = round(time.monotonic() - start, 3)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("Could not encode annotated frame")
    result["annotated_b64"] = base64.b64encode(encoded).decode("ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-model", default=os.environ.get("SAM3_PATH", SAM3_MODEL))
    parser.add_argument("--encoder-path", default=os.environ.get("DINO_V2_PATH"))
    parser.add_argument("--policy", choices=POLICIES, default="review_band_175_120")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    print("Loading SAM3 and frozen Approach 17...", file=sys.stderr, flush=True)
    segmenter = BrickSegmenter(args.sam3_model, args.device, args.local_files_only)
    classifier = PhysicalBrickClassifier(encoder_path=args.encoder_path, device=args.device,
                                         local_files_only=args.local_files_only)
    send({"type": "ready", "model": "SAM3 + Approach 17", "device": segmenter.device,
          "pipeline": "sam3", "threshold_policy": args.policy})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = process_request(request, segmenter, classifier, args.policy)
        except Exception as error:
            response = {"type": "error", "frame_id": None,
                        "error": f"{type(error).__name__}: {error}"}
        try:
            send(response)
        except BrokenPipeError:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
