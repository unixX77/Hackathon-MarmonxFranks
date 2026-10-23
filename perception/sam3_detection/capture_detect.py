"""Interactively capture webcam frames and detect good / bad LEGO bricks with SAM 3.

The window shows the live camera on the left and the detection of the last
captured frame on the right. Press Enter to save the current frame and run
SAM 3 on it; q (or Escape) quits.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from segment_lego_sam3 import BrickSegmenter, add_detection_args, save_result, write_summary


def placeholder(shape, lines):
    panel = np.full(shape, 40, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (20, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory: frames go to <output>/rgb, detections to <output>/results "
             "(default: data beside this script)",
    )
    add_detection_args(parser)
    args = parser.parse_args()
    rgb_dir, results_dir = args.output / "rgb", args.output / "results"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM 3 ...")
    segmenter = BrickSegmenter(args)
    camera = cv2.VideoCapture(args.camera)
    window = "SAM 3 brick capture - Enter: capture + detect | q: quit"
    summary, result = [], None
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}. Try --camera 1.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print(f"Saving to {args.output.resolve()}")
        print("Focus the preview window. Press Enter to capture and detect; q to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            live = frame.copy()
            cv2.putText(live, f"Enter: capture + detect | q: quit | Saved: {len(summary)}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            right = result if result is not None and result.shape == frame.shape else placeholder(
                frame.shape, ["Press Enter to capture", "and detect bricks"])
            cv2.imshow(window, np.hstack([live, right]))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = rgb_dir / f"rgb_{stamp}.png"
                # Save the original frame without the preview text.
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save {path}")
                cv2.imshow(window, np.hstack([live, placeholder(frame.shape, ["Detecting ..."])]))
                cv2.waitKey(1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                semantic, instance, records, rejected = segmenter(rgb)
                entry = save_result(results_dir / path.stem, path, rgb, semantic, instance, records,
                                    rejected, args)
                summary.append(entry)
                write_summary(results_dir, summary)
                preview = cv2.imread(str(results_dir / path.stem / "preview.png"))
                result = preview[:, frame.shape[1]:]  # right half: the detection
                print(f"Saved {path.name}: {entry['bricks']} bricks "
                      f"({entry['good']} good, {entry['bad']} bad) {entry['per_colour']}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, cv2.error) as error:
        raise SystemExit(str(error)) from error
    except KeyboardInterrupt:
        pass
