"""Quick test of a single SAM 3 text prompt.

Runs one text prompt (default "lego brick") on an image or a folder of PNGs and
saves, per image, a split view: input on the left, every detected instance in
its own colour on the right, labelled with its id and score. No colour classes
and no good / bad labels; use this to see what a prompt finds.

For every image `<stem>.png` the output folder gets:
  <stem>_preview.png   input | instance masks
  <stem>.json          per instance: id, score, box

Run in the `sam3` conda env (needs Hugging Face access to facebook/sam3).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def colour(i):
    """Distinct RGB colour per instance id."""
    hsv = np.uint8([[[(i * 47) % 180, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def preview(rgb, masks, scores):
    out = rgb.copy()
    order = np.argsort([-m.sum() for m in masks])  # large first, so small masks stay visible
    for i in order:
        out[masks[i]] = (0.4 * rgb[masks[i]] + 0.6 * np.array(colour(i + 1))).astype(np.uint8)
    scale = max(rgb.shape[0] / 480, 1.0)
    for i in order:
        contours, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), max(1, int(2 * scale)))
        ys, xs = np.nonzero(masks[i])
        org = (int(xs.mean()) - int(20 * scale), int(ys.mean()))
        text = f'#{i + 1} {scores[i]:.2f}'
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 0, 0), int(3 * scale))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), max(1, int(scale)))
    cv2.putText(out, f'{len(masks)} instances', (int(10 * scale), int(30 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0), int(4 * scale))
    cv2.putText(out, f'{len(masks)} instances', (int(10 * scale), int(30 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), max(1, int(2 * scale)))
    return np.hstack([rgb, out])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='a PNG image or a folder of PNGs')
    parser.add_argument('--prompt', default='lego brick')
    parser.add_argument('--out', type=Path, default=Path('data/test_prompt'))
    parser.add_argument('--model', default='facebook/sam3')
    parser.add_argument('--threshold', type=float, default=0.5, help='instance score threshold')
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    paths = sorted(args.input.glob('*.png')) if args.input.is_dir() else [args.input]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'[sam3] {len(paths)} images, prompt={args.prompt!r}, device={device}')
    for n, path in enumerate(paths, 1):
        image = Image.open(path).convert('RGB')
        inputs = processor(images=image, text=args.prompt, return_tensors='pt').to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=args.threshold, mask_threshold=args.mask_threshold,
            target_sizes=[image.size[::-1]])[0]
        scores = result['scores'].float().cpu().numpy()
        if len(scores):
            masks = result['masks'].cpu().numpy().astype(bool)
            boxes = result['boxes'].float().cpu().numpy()
        else:  # empty results are not resized to the image
            masks, boxes = np.zeros((0, image.height, image.width), bool), np.zeros((0, 4))

        Image.fromarray(preview(np.array(image), masks, scores)).save(args.out / f'{path.stem}_preview.png')
        (args.out / f'{path.stem}.json').write_text(json.dumps({
            'image': str(path), 'prompt': args.prompt, 'threshold': args.threshold,
            'instances': [{'id': i + 1, 'score': round(float(s), 3), 'box_xyxy': [round(float(v), 1) for v in b]}
                          for i, (s, b) in enumerate(zip(scores, boxes))]}, indent=2))
        print(f'[sam3] {n}/{len(paths)} {path.name}: {len(scores)} instances')
    print(f'[sam3] wrote {args.out}')


if __name__ == '__main__':
    main()
