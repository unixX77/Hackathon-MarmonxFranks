# Requested red lattice task

Use `src/lego_sorter/config/lattice_red.yaml` for the requested mock geometry.
The original `sorter.yaml` remains the historical 60-brick demonstration fixture.

The source contains 300 nominal 2 x 4 bricks: 10 centred layers, each with six rows
of five touching bricks. Rows have a two-stud air gap. Bottom rows run along base
Y (yaw 90 degrees); adjacent layers alternate by 90 degrees. At 8 mm stud pitch,
the bottom footprint is 176 mm in X by 160 mm in Y; the next footprint is
160 by 176 mm. These are the 22 x 20 and 20 x 22 stud orientations, giving an
8 mm overhang at each row end. Identical layer parity has identical XY positions.

High-confidence red bricks go to `selected`, which uses the same lattice geometry
at a separate illustrative origin. Other colours and low-confidence detections go
to a single `discard` drop-box target, without arranging them in a grid. The mock
uses repeated release coordinates and a distinct receipt for each disposal. Its
300-drop capacity is an illustrative count, not a measured physical box capacity.
It does not simulate settling, bouncing, fill height, or accumulated collisions.

Destination capacity is 300 red bricks, not a requirement to obtain 300 red bricks
from a mixed source. The mission ends when the source is independently verified
empty. For example, the default synthetic colour cycle produces 60 red placements
(two complete destination layers) and 240 disposal receipts. Real input colour
counts may leave a partial last layer. The current allocator scans bottom-up and
accepts the first supported available cell; it does not require a whole lower layer
to be complete before using a supported cell above it.

The nominal geometric overlap at each interface is 50% for interior upper rows
and 25% for outer upper rows, split equally between two supporting rows. Contacts
can span seams between adjacent lower bricks. This config uses a 25% overlap
threshold to admit the requested outer rows. It is not a proof of physical stability,
stud seating, or extraction feasibility. Support must come from verified PLACED
cells, so missing support under a red brick is never assumed present.

## Physical work still required

`source.stud_connected` and `selected.stud_connected` explicitly describe the real
task. Hardware startup rejects these layouts: straight lifting/release does not
implement validated stud detachment, reaction-force handling, or stud seating.
The mock ignores those forces. Trash-box drops are also mock-only until measured
box geometry, release clearance, fill handling and verification are implemented.
Do not remove these markers or set commissioning flags to bypass these limits.

`hardware.yaml` is a disabled preparation template of this task. Its origins,
finger measurements, tray/drop dimensions, robot motion, camera detection and
hand-eye calibration still require commissioning. No robot motion was executed.

## Preview and simulation

```bash
cd /home/tanay/Documents/Hackathon/project-planner
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src/lego_sorter"
/usr/bin/python3 -s -m lego_sorter.cli preview --config src/lego_sorter/config/lattice_red.yaml
# Use a new runtime directory for a fresh simulated world.
/usr/bin/python3 -s -m lego_sorter.cli mock --config src/lego_sorter/config/lattice_red.yaml --runtime "$PWD/runtime/my-red-lattice"
```

The preview draws release footprints for the trash target, not the contents of a
filled box. A simulated completion does not validate the physical assembly task.
