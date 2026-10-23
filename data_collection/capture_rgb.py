"""Interactively save color webcam frames as PNG images."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "data" / "rgb",
        help="Output directory (default: data/rgb beside this script)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    window = "RGB capture - Enter: save | q: quit"
    count = 0
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}. Try --camera 1.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print(f"Saving to {args.output.resolve()}")
        print("Focus the preview window. Press Enter to save; q to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            preview = frame.copy()
            cv2.putText(preview, f"Enter: save | q: quit | Saved: {count}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = args.output / f"rgb_{stamp}.png"
                # OpenCV accepts BGR and encodes it as a standard color PNG.
                # Save the original frame without the preview instructions.
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save {path}")
                count += 1
                print(f"Saved {path}")
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
