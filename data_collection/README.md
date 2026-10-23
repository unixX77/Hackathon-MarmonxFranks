# Webcam RGB capture

Install and run from a terminal on the computer with the webcam:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python capture_rgb.py
```

For the Logitech C920 connected to this system, run
`python capture_rgb.py --camera 2`.

Click the preview window to focus it. Press **Enter** to save the displayed
frame, and **q** (or Escape) to quit. Closing the window also quits.

Images are saved as timestamped, full-resolution color PNGs in `data/rgb/`
beside the script. Preview text is not included in saved images.

For a different camera or destination:

```bash
python capture_rgb.py --camera 1 --output data/rgb
```

Requires a graphical desktop and the GUI-enabled `opencv-python` package
(not `opencv-python-headless`).
