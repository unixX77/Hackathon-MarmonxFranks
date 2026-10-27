"""The window of sort_pile.py: the picture of the last scan with every brick outlined in the colour of what happens to it.

  yellow  to do this round          cyan (thick)  the one the arm is on now      green  on its colour stack / in the row
  purple  in the bin (bad)          red           failed: the run stopped there
  orange  no stack for its colour   grey          lower layer, or on a stack: not picked this round

The picture is the frame the pipeline used (the recording's rgb/<frame>.png), the outlines are the bricks' footprints from the result
file whose ids the pipeline printed (bricks_reviewed.json, bricks_second_opinion.json or bricks.json). Everything runs in its own thread,
so the robot never waits for the window; closing the window only hides it. Without a display there is simply no window.
"""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

STYLE = {     # status: (BGR colour, line width, legend text)
    'todo': ((0, 220, 255), 2, 'to do'),
    'current': ((255, 255, 0), 5, 'now'),
    'done': ((0, 200, 0), 3, 'stacked'),
    'bin': ((200, 0, 200), 3, 'in the bin'),
    'failed': ((0, 0, 255), 4, 'failed'),
    'skipped': ((0, 140, 255), 2, 'no stack'),
    'later': ((140, 140, 140), 1, 'not this round'),
}
WIDTH = 1280                    # window width (pixels)
HEADER = 70


def result_files(final):
    """(the clean rgb frame, the bricks file) of a pipeline result folder (03_final/capture/<recording>/<frame>)."""
    final = Path(final)
    bricks = next((final / name for name in ('bricks_reviewed.json', 'bricks_second_opinion.json', 'bricks.json')
                   if (final / name).is_file()), None)
    rgb = None
    try:
        info = json.loads((final.parents[3] / 'pipeline.json').read_text())
        candidate = Path(info['recording']) / 'rgb' / f'{info["frame"]}.png'
        rgb = candidate if candidate.is_file() else None
    except (OSError, ValueError, KeyError, IndexError):
        pass
    if rgb is None and (final / 'overlay.png').is_file():
        rgb = final / 'overlay.png'
    return rgb, bricks


class Board:
    def __init__(self, title='sort_pile'):
        self.title = title
        self.lock = threading.Lock()
        self.image, self.outlines, self.status = None, {}, {}
        self.lines = ['starting']
        self.dirty, self.alive, self.shown = True, True, True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- what the program tells it
    def text(self, *lines):
        with self.lock:
            self.lines, self.dirty = [str(line) for line in lines], True

    def scan(self, final, statuses):
        """A new picture: `final` is the pipeline's result folder, statuses {brick id: status} for the bricks to show."""
        rgb, bricks = result_files(final) if final else (None, None)
        image = cv2.imread(str(rgb)) if rgb else None
        outlines = {}
        if bricks:
            try:
                for brick in json.loads(bricks.read_text())['bricks']:
                    if brick.get('id') and brick.get('footprint_px'):
                        outlines[int(brick['id'])] = np.round(np.array(brick['footprint_px'])).astype(np.int32)
            except (OSError, ValueError, KeyError):
                outlines = {}
        with self.lock:
            self.image, self.outlines, self.status, self.dirty = image, outlines, dict(statuses), True

    def mark(self, ident, status):
        with self.lock:
            self.status[ident], self.dirty = status, True

    def close(self, wait=0.0):
        time.sleep(wait)
        self.alive = False
        self.thread.join(timeout=2.0)

    # ---------------------------------------------------------------- drawing, in its own thread
    def _compose(self):
        with self.lock:
            image = None if self.image is None else self.image.copy()
            outlines, status, lines = dict(self.outlines), dict(self.status), list(self.lines)
        if image is None:
            image = np.full((720, 1280, 3), 40, np.uint8)
        order = sorted(status, key=lambda i: STYLE.get(status[i], STYLE['later'])[1])     # the thick ones on top
        for ident in order:
            colour, width, _ = STYLE.get(status[ident], STYLE['later'])
            if ident not in outlines:
                continue
            cv2.polylines(image, [outlines[ident]], True, colour, width, cv2.LINE_AA)
            x, y = outlines[ident].mean(axis=0).astype(int)
            cv2.putText(image, str(ident), (x - 12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(image, str(ident), (x - 12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        scale = WIDTH / image.shape[1]
        image = cv2.resize(image, (WIDTH, int(image.shape[0] * scale)))
        header = np.full((HEADER, WIDTH, 3), 30, np.uint8)
        for k, line in enumerate(lines[:2]):
            cv2.putText(header, line[:110], (12, 28 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        counts = {}
        for value in status.values():
            counts[value] = counts.get(value, 0) + 1
        x = 12
        legend = np.full((34, WIDTH, 3), 30, np.uint8)
        for key, (colour, _, name) in STYLE.items():
            label = f'{name} {counts.get(key, 0)}'
            cv2.rectangle(legend, (x, 10), (x + 16, 26), colour, -1)
            cv2.putText(legend, label, (x + 22, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            x += 32 + 11 * len(label)
        return np.vstack([header, image, legend])

    def _loop(self):
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        except cv2.error:
            return                                       # no display: no window, the program goes on
        while self.alive:
            if self.dirty and self.shown:
                self.dirty = False
                cv2.imshow(self.title, self._compose())
            cv2.waitKey(50)
            try:
                if self.shown and cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    self.shown = False                   # closed by hand: stays closed, the robot goes on
            except cv2.error:
                self.shown = False
        cv2.destroyAllWindows()
