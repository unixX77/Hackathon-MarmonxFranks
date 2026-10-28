"""Pallet of bricks: where every placed brick goes, which slot is next, and the state kept between runs.

No ROS here: pure logic, unit tested. pose_pick.py --pallet places each brick in the next free slot of a grid
and records it as soon as the hand has let go, so the next run goes to the next slot by itself.

The grid (config/pallet.yaml), all lengths in metres in the robot base frame (fr3_link0):
    origin      [x, y]   centre of slot 1
    step_along  [dx, dy] from one slot to the next one in a row
    step_row    [dx, dy] from one row to the next row
    per_row     slots in a row         rows   number of rows
    yaw_deg     angle of the long side of every placed brick (0 = along base x)
    layer       1 = on the baseplate / table, 2 = on top of one brick, ...
    area        optional [xmin, xmax, ymin, ymax]: every brick must lie inside (e.g. the baseplate)
Slot k (0-based) is in row k // per_row at place k % per_row:  origin + place * step_along + row * step_row.
"""
import json
import math
import os
import tempfile
import time
from pathlib import Path

import yaml

MAX_SLOTS = 400
DEFAULTS = {'yaw_deg': 0.0, 'layer': 1, 'min_gap': 0.004, 'state_file': 'pallet_state.json'}


# --------------------------------------------------------------------------- the grid
def _pair(value, name):
    if not (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise ValueError(f'pallet {name} must be two numbers [x, y] in metres')
    return (float(value[0]), float(value[1]))


def normalise(raw):
    """Check a pallet mapping (from YAML) and fill in the defaults. Raises ValueError with a clear message."""
    if not isinstance(raw, dict):
        raise ValueError('The pallet file must be a YAML mapping')
    missing = [k for k in ('origin', 'step_along', 'step_row', 'per_row', 'rows') if k not in raw]
    if missing:
        raise ValueError('The pallet file is missing: ' + ', '.join(missing))
    p = dict(DEFAULTS)
    p.update(raw)
    p['needs_setup'] = bool(raw.get('needs_setup', False))
    p['origin'], p['step_along'], p['step_row'] = (_pair(p['origin'], 'origin'), _pair(p['step_along'], 'step_along'),
                                                    _pair(p['step_row'], 'step_row'))
    for key in ('per_row', 'rows', 'layer'):
        if isinstance(p[key], bool) or not isinstance(p[key], int) or p[key] < 1:
            raise ValueError(f'pallet {key} must be a whole number, 1 or more')
    if p['per_row'] * p['rows'] > MAX_SLOTS:
        raise ValueError(f'pallet has {p["per_row"] * p["rows"]} slots: more than {MAX_SLOTS}')
    if max(abs(v) for v in p['step_along'] + p['step_row']) > 0.5:
        raise ValueError('pallet steps are in METRES (0.04 is 4 cm); more than 0.5 m is surely a typo')
    if p['per_row'] > 1 and p['step_along'] == (0.0, 0.0):
        raise ValueError('pallet step_along is zero but there is more than one slot in a row')
    if p['rows'] > 1 and p['step_row'] == (0.0, 0.0):
        raise ValueError('pallet step_row is zero but there is more than one row')
    p['yaw_deg'] = float(p['yaw_deg'])
    p['min_gap'] = float(p['min_gap'])
    if p.get('area') is not None:
        area = p['area']
        if not (isinstance(area, (list, tuple)) and len(area) == 4 and all(isinstance(v, (int, float)) for v in area)
                and area[0] < area[1] and area[2] < area[3]):
            raise ValueError('pallet area must be [xmin, xmax, ymin, ymax] in metres')
        p['area'] = tuple(float(v) for v in area)
    return p


def load_pallet(path):
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    except OSError as problem:
        raise ValueError(f'Cannot read the pallet file {path}: {problem}') from None
    return normalise(raw)


def slot_count(p):
    return p['per_row'] * p['rows']


def slot_row_place(p, index):
    """(row, place) of a 0-based slot index, both 0-based."""
    if not 0 <= index < slot_count(p):
        raise ValueError(f'slot {index + 1} does not exist (the pallet has {slot_count(p)})')
    return index // p['per_row'], index % p['per_row']


def slot_xy(p, index):
    """Centre (x, y) of a 0-based slot, metres, base frame."""
    row, place = slot_row_place(p, index)
    return (p['origin'][0] + place * p['step_along'][0] + row * p['step_row'][0],
            p['origin'][1] + place * p['step_along'][1] + row * p['step_row'][1])


def slot_label(p, index):
    row, place = slot_row_place(p, index)
    return f'slot {index + 1} of {slot_count(p)} (row {row + 1}, place {place + 1})'


# ---------------------------------------------------------------- geometry checks
def _axes(yaw_deg):
    a = math.radians(yaw_deg)
    return (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))        # along the long side, across it


def brick_gap(d, length, width, yaw_deg):
    """Distance between two congruent bricks (length x width, long side at yaw_deg) whose centres differ by d;
    negative when they overlap."""
    u, v = _axes(yaw_deg)
    du, dv = abs(d[0] * u[0] + d[1] * u[1]), abs(d[0] * v[0] + d[1] * v[1])
    if du < length and dv < width:
        return -min(length - du, width - dv)
    return math.hypot(max(du - length, 0.0), max(dv - width, 0.0))


def finger_conflict(d, length, width, yaw_deg, open_clearance, tip_thickness, tip_width, margin=0.001):
    """True when, releasing a brick, a finger tip would touch a neighbour whose centre differs by d. The fingers
    stand beside the brick across its width: from open_clearance to open_clearance + tip_thickness outside its
    long edges, and tip_width wide along the long side."""
    u, v = _axes(yaw_deg)
    du, dv = abs(d[0] * u[0] + d[1] * u[1]), abs(d[0] * v[0] + d[1] * v[1])
    return (du < (tip_width + length) / 2 + margin
            and open_clearance - margin < dv < width + open_clearance + tip_thickness + margin)


def check_pallet(p, c):
    """Everything that can be checked before a brick moves. Raises ValueError listing what is wrong; returns a list
    of notes (things that were skipped or are tight) to print.
    c: the pose_pick config (brick size, allowed window, hand widths and the optional finger tip size)."""
    notes, errors = [], []
    if p.get('needs_setup'):
        raise ValueError('The pallet file is still the template (needs_setup: true). Put the real position of slot 1 in '
                         'origin, choose the steps and counts, then set needs_setup: false.')
    length, width = c['block_size'][0], c['block_size'][1]
    wx, wy = c['workspace_x'], c['workspace_y']
    half_u, half_v = length / 2, width / 2
    u, v = _axes(p['yaw_deg'])
    reach_x = abs(u[0]) * half_u + abs(v[0]) * half_v                  # half extent of a brick along base x / y
    reach_y = abs(u[1]) * half_u + abs(v[1]) * half_v
    outside = []
    for i in range(slot_count(p)):
        x, y = slot_xy(p, i)
        if not (wx[0] <= x <= wx[1] and wy[0] <= y <= wy[1]):
            outside.append(f'slot {i + 1} at x={x:.3f} y={y:.3f}')
        elif p.get('area') and not (p['area'][0] <= x - reach_x and x + reach_x <= p['area'][1]
                                    and p['area'][2] <= y - reach_y and y + reach_y <= p['area'][3]):
            outside.append(f'slot {i + 1} at x={x:.3f} y={y:.3f} (brick leaves the pallet area)')
    if outside:
        errors.append('Slots outside the allowed window / area: ' + '; '.join(outside[:4]) +
                      (f' and {len(outside) - 4} more' if len(outside) > 4 else ''))
    smallest = None
    shifts = [(m, n) for m in range(-(p['per_row'] - 1), p['per_row']) for n in range(-(p['rows'] - 1), p['rows'])
              if (m, n) != (0, 0)]
    for m, n in shifts:
        d = (m * p['step_along'][0] + n * p['step_row'][0], m * p['step_along'][1] + n * p['step_row'][1])
        gap = brick_gap(d, length, width, p['yaw_deg'])
        smallest = gap if smallest is None else min(smallest, gap)
    if smallest is not None and smallest < p['min_gap']:
        errors.append(f'Bricks in neighbouring slots would be {smallest * 1000:.1f} mm apart (or overlap): at least '
                      f'{p["min_gap"] * 1000:.0f} mm is needed (min_gap). Increase step_along / step_row.')
    elif smallest is not None:
        notes.append(f'Closest gap between two bricks on the pallet: {smallest * 1000:.1f} mm.')
    tip_t, tip_w = c.get('finger_tip_thickness'), c.get('finger_tip_width')
    if tip_t is None or tip_w is None:
        notes.append('NOTE: finger_tip_thickness / finger_tip_width are not set in the config, so the room for the '
                     'fingers between neighbouring bricks was NOT checked.')
    else:
        open_clearance = (c['open_width'] - c.get('hand_zero_width', 0.0) - width) / 2
        worst = [(m, n) for m, n in shifts
                 if finger_conflict((m * p['step_along'][0] + n * p['step_row'][0],
                                     m * p['step_along'][1] + n * p['step_row'][1]),
                                    length, width, p['yaw_deg'], open_clearance, tip_t, tip_w)]
        if worst:
            errors.append(f'A finger tip ({tip_t * 1000:.1f} mm thick, {tip_w * 1000:.0f} mm wide) would hit a neighbouring '
                          f'brick when a brick is released: {len(worst)} neighbour positions conflict, e.g. {worst[0]} '
                          '(steps in slots). Increase the spacing across the brick width, or place along the long side.')
    if errors:
        raise ValueError(' | '.join(errors))
    return notes


# ------------------------------------------------------------------------ the state
def state_path(p, pallet_file):
    """The state file lives next to the pallet file (absolute, so it does not depend on the folder you run from)."""
    path = Path(p['state_file'])
    return path if path.is_absolute() else Path(pallet_file).resolve().parent / path


class PalletState:
    """Which slots are filled, kept in a small JSON file between runs."""

    def __init__(self, path):
        self.path = Path(path)
        self.filled = {}                       # 0-based index -> {'time': ..., 'how': 'placed' | 'skipped' | 'marked', 'xy': ...}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.filled = {int(k): v for k, v in data.get('filled', {}).items()}
            except (ValueError, OSError, AttributeError) as problem:
                raise ValueError(f'The pallet state file {self.path} is unreadable ({problem}). Fix or delete it '
                                 '(pallet_tool.py reset) - it says which slots hold a brick.') from None

    def next_free(self, p):
        for i in range(slot_count(p)):
            if i not in self.filled:
                return i
        return None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(dir=str(self.path.parent), prefix='.pallet_', suffix='.tmp')
        try:
            with os.fdopen(handle, 'w') as f:
                json.dump({'filled': {str(k): v for k, v in sorted(self.filled.items())}}, f, indent=2)
            os.replace(temp, self.path)
        except BaseException:
            if os.path.exists(temp):
                os.unlink(temp)
            raise

    def mark(self, index, how='placed', xy=None):
        self.filled[index] = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'how': how,
                              'xy': [round(float(xy[0]), 5), round(float(xy[1]), 5)] if xy else None}
        self.save()

    def unmark(self, index):
        self.filled.pop(index, None)
        self.save()

    def reset(self):
        self.filled = {}
        self.save()


def choose_slot(p, state, requested=None):
    """The 0-based slot to fill: the requested 1-based slot (refused if it already holds a brick), else the first free one."""
    if requested is not None:
        index = requested - 1
        slot_row_place(p, index)                                      # raises if it does not exist
        if index in state.filled:
            raise ValueError(f'{slot_label(p, index)} is already filled ({state.filled[index]["how"]} at '
                             f'{state.filled[index]["time"]}). Free it first: pallet_tool.py unmark {requested}')
        return index
    index = state.next_free(p)
    if index is None:
        raise ValueError(f'The pallet is full ({slot_count(p)} of {slot_count(p)} slots). Clear the bricks, then '
                         'pallet_tool.py reset')
    return index


def render(p, state, highlight=None):
    """The grid as text lines: # filled, s skipped, > the slot about to be filled, . free."""
    lines = [f'Pallet: {p["per_row"]} per row x {p["rows"]} rows = {slot_count(p)} slots, bricks at yaw '
             f'{p["yaw_deg"]:.0f} deg, layer {p["layer"]}    (# filled  s skipped  > next  . free)']
    for row in range(p['rows']):
        cells = []
        for place in range(p['per_row']):
            i = row * p['per_row'] + place
            if i == highlight:
                cells.append('>')
            elif i in state.filled:
                cells.append('s' if state.filled[i]['how'] == 'skipped' else '#')
            else:
                cells.append('.')
        x0, y0 = slot_xy(p, row * p['per_row'])
        x1, y1 = slot_xy(p, (row + 1) * p['per_row'] - 1)
        lines.append(f'  row {row + 1}: {" ".join(cells)}    from x={x0:.3f} y={y0:.3f} to x={x1:.3f} y={y1:.3f}')
    done = sum(1 for i in state.filled if i < slot_count(p))
    lines.append(f'  {done} of {slot_count(p)} slots filled')
    return lines
