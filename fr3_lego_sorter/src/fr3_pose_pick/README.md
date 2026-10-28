# fr3_pose_pick: the robot side of the LEGO sorter

The application is `sort_pile.py` (see the [repository README](../../README.md) and the
[technical documentation](../../DOCUMENTATION.md)). This file is the package's development notes:
the pose-input pick tool the application is built on, and how each setting was found on the real robot,
newest sections in the middle (dated). Values quoted in older sections may since have changed: the
config is the authority.

`pose_pick.py`: you give the poses, the arm does the rest. It does not look for the brick itself; the
camera pipeline is in `vision/` of this repository.

Motion goes through MoveIt, but not its path planner: every move is a straight, collision-checked
Cartesian line (`compute_cartesian_path`), executed with `execute_trajectory` at low speed
(`smooth_motion.py`). A move that cannot be done as a clean straight line is refused, not improvised.

## Poses

All in `fr3_link0` (the robot base), metres. +x is away from the robot, +y to its left.

| Option | Meaning |
|---|---|
| `--pose X Y [Z]` | brick **centre** to pick. Z left out = the brick rests on the table |
| `--yaw DEG` or `--quat X Y Z W` | angle of the brick's long side from base X (a quaternion: only its yaw is used) |
| `--place X Y [Z]` | brick centre to put it down at. Z left out = on the table; give Z to stack |
| `--place-offset DX DY` | put it down this far from where it was picked (base frame, metres): `0 0.05` = 5 cm to the robot's left. Follows any correction made at the inspection stop; on the table |
| `--place-yaw DEG` | angle of the placed brick (default: the angle it was picked at) |
| `--pose-is-tcp`, `--place-is-tcp` | the given Z is the hand (TCP) height, not the brick centre |

A pose outside `workspace_x/y`, one that would push the fingertips into the table, or a place below the
table is refused before anything moves. Settings (brick size, table, limits) are in `config/pose_pick.yaml`.
Check `block_size` against the real brick.

## What it does

open the hand, hover above the pick pose, **stop 4 cm above the brick for inspection**, lower, close,
lift, carry at hover height to the place pose, lower, open, back off straight up. Without `--place` it
stops after the lift with the brick in the hand. Every arrival is compared with the target (warn above
3 mm, stop above 10 mm).

At the inspection stop look at the fingers against the brick and type small corrections:
`x 0.002`, `y -0.001 yaw 5` (at most 10 mm / 15 degrees at a time, 30 mm in total), `ok` to continue,
`q` to stop. The total correction is printed and appended to `pose_pick_offsets.json`, which tells your
teammate how far off the given pose was.

## Pick and place 5 cm away

```bash
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --camera-xyz X Y Z --yaw 0 --place-offset 0 0.05
```
After the lift it carries the brick at hover height, lowers it until it rests on the table (2 mm margin),
opens the hand and backs off straight up. `--place-offset 0.05 0` is 5 cm further from the robot,
`0 -0.05` is 5 cm to its right. The offset is checked before anything moves, including for the largest
correction you can make at the inspection stop (30 mm), so it never refuses with the brick in the hand.

## Poses given in the camera frame

If the pose comes from a camera picture (`camera_color_optical_frame`: x right, y down, z forward), give it
as the centre of the brick's **top face** and the robot converts it with your hand-eye matrix
(`calibration_file` = `data/calibration/charuco_result.json`, `T_ee_camera`): `p_base = T_base_hand @
T_hand_camera @ p_camera`. Nothing is detected: you type the numbers.

```bash
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --camera-xyz 0.035 0.059 0.144 --camera-quat 0 0 0.26 0.97 \
     --place-camera-xyz 0.24 -0.005 0.46 --ee-pose 0.314 0.0 0.406 1 0 0 0
```
- `--ee-pose X Y Z QX QY QZ QW` is where the hand (`fr3_hand_tcp`) was **when the picture was taken**.
  Leave it out to use the arm's pose right now, which is only right if the arm has not moved since.
- `--camera-quat` / `--place-camera-quat` give the brick orientation in the camera frame; only the
  direction of its long side on the table is used (`--camera-long-axis x|y`). Or use `--yaw` / `--place-yaw`.
- The height of the brick is worked out automatically (see the next section): you do not give a layer count.
- **Brick angle from the image:** `--camera-angle DEG` (and `--place-camera-angle DEG`) takes the angle of the
  brick's long side **in the image**, which is what a segmentation mask gives: degrees from the image's right
  (+x) toward the image's down (+y), i.e. clockwise on the screen, any value (a line repeats every 180 degrees).
  It replaces `--yaw` / `--camera-quat`. Only the direction on the table is used, and the camera's mounting
  rotation and the hand yaw at the time of the picture are taken from the hand-eye result and the arm pose.
  With the hand pointing down at yaw 0: a long side vertical in the image is a brick along base x (yaw 0),
  and 80 degrees (10 degrees clockwise from vertical on the screen) is a brick turned +10 degrees.

```bash
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --camera-xyz 0.035240 0.005403 0.318830 --camera-angle 80 --place-offset 0 0.07
```
  `A` in `--camera-angle A` is the number your teammate sends. It must mean: the angle of the brick's long side in
  the image, from the image's right toward the image's down (clockwise on the screen). If his number means
  something else, fix it once in `config/pose_pick.yaml`: `used = camera_angle_sign * his_angle +
  camera_angle_offset_deg` (counter-clockwise positive: `camera_angle_sign: -1`; the angle of the short side:
  `camera_angle_offset_deg: 90`). Find out with three bricks: long side vertical on the screen should give 90 (or
  -90), horizontal 0, and tilted 10 degrees clockwise from vertical 80. The run prints the converted angle.
- The hand-eye result missed its strict gate (typical error about 3 mm / 0.9 deg), so expect a few mm
  of error from the conversion alone: use the inspection stop to correct it.

## Hand widths (fixed in the config)

`config/pose_pick.yaml` sets the hand once for every open and close: **open 19 mm** (`open_width`, it was 0.017, then 0.018, and went to
0.019 on 26 Oct at 18:57 after a finger landed on the brick) and **close 15 mm** (`grasp_width`), for a 16 mm brick with other bricks
right next to it. Nothing to type on the command line. `min_open_clearance: 0.001` lowers the config check from its default 5 mm margin
to the 3 mm that 19 mm leaves. That is 1.5 mm of room per side, so the script prints a note at each
start, and the hand has to be lined up to about 1 mm sideways and in yaw: use the inspection stop,
and expect the occasional finger landing on the brick. A hand narrower than the brick plus
`min_open_clearance` is still refused.

## Grasp slightly off the centre of the brick: measure it once, store it for good

If picking and placing are accurate but the fingers close a little off the middle of the brick, the real
middle between the fingertips sits a small, fixed distance from the point the arm is told to go to. Placing
stays right because the same offset is in the arm at the place. The cure is `tcp_correction_hand` in
`config/pose_pick.yaml`: metres, `[along the brick's long side, across it]` in the **hand's own frame**
(at hand yaw 0: along = base +x, across = base -y), added to **every** commanded hand position, pick and place.

Measure it, do not guess it:
1. Do 3 to 5 picks with the inspection stop. At `correction>`, nudge (`x 0.002`, `y -0.001`, up to 10 mm per
   command) until the fingers look centred on the brick (a scale helps), then `ok`. Use different spots and angles.
2. From the folder where you ran `pose_pick.py`:
   ```bash
   python3 src/fr3_pose_pick/scripts/measure_grasp_offset.py --last 5
   ```
   It turns every correction into the hand frame (so the brick angle does not matter), averages them and prints
   the spread. Both numbers should be well under 1.5 mm; if the runs disagree, the offset depends on where the
   brick is and one value will not fix it.
3. Store it: `... --last 5 --apply src/fr3_pose_pick/config/pose_pick.yaml` (the old file is kept as `.bak`).
4. Run again. Corrections you still type are added to the stored value, so you can repeat it to refine.

Runs where you pressed `ok` without looking count as "no correction needed": use `--last` to take only the
runs where you really checked. The arrival line shows the signed error (`dx`, `dy`) of the arm itself.

## Brick height from a camera point: automatic (on the table, on other bricks, or anywhere)

Give only the camera xyz. The robot works out the height itself, so a brick lying on the table, on one or
several other bricks, or on anything else is picked the same way:
1. The measured top face is corrected by `camera_height_bias` (3.7 mm). In your real runs the converted top
   face of a brick lying on the table came out 0.7 to 7.1 mm too high, 3.7 mm on average over 8 runs: the
   studs (about 1.7 mm) and a small depth bias. Re-measure it if the camera setup changes.
2. If the corrected height is within `layer_snap_tolerance` (3 mm) of a whole brick layer (the brick height in
   the config is one layer; layer 1 is on the table), the **exact layer height** is used, so the camera noise does
   not move the fingers.
3. Otherwise the corrected measurement is used as it is, and the script says so: look carefully at the fingers
   at the inspection stop.
4. Refused as impossible: a top face below a brick on the table, or higher than `max_brick_top_above_table`
   (12 cm). That almost always means the arm was not in the pose the picture was taken from.

```bash
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --camera-xyz X Y Z --yaw 0 --place-offset 0 0.07
```
Overrides, only if you want them: `--layers N` states the layer (1 = on the table); the corrected height must
then agree within `table_check_tolerance` or it refuses. `--place-layers N` puts the brick down as layer N (with
`--place`, `--place-offset` or `--place-camera-xyz`; a camera place point is automatic like the pick).
`--use-measured-z` always uses the corrected measurement and never snaps.

**Fixing a height that is a little off, by eye:** at the inspection stop type `z 0.002` (up 2 mm, at most 5 mm
at a time and 15 mm in total) as well as `x`, `y` and `yaw`. The total correction is logged with every run.

Limits you should know:
- A layer is 9.6 mm and the camera height is good to a few mm, so a wrong layer can in principle be mistaken for
  a right one. The inspection stop is the real check: look at the fingers' height against the brick.
- The camera needs at least about 28 cm to the brick to measure depth reliably (a RealSense D435). A rule of
  thumb for a brick on the table: the depth in the camera numbers is about hand height + 0.055 m, minus the
  height of what it lies on. If the numbers do not fit this, retake the picture from higher up.
- The bricks of a stack, and the neighbours, are not in the planning scene. The hand comes straight down beside
  the brick, so keep other bricks out of the finger travel on each side.
- Fingertips reach about 9 mm below the hand point, so they grip the upper part of the brick and stay clear of
  the brick below.

## New (printed) gripper tips: measure, then calibrate with the inspection stop

The widths, the fingertip height and the stored grasp centring all belong to the tips they were measured with.
After a change of tips:
1. **Reset the centring** (it was measured for the old tips):
   `sed -i 's/^tcp_correction_hand:.*/tcp_correction_hand: [0.0, 0.0]/' src/fr3_pose_pick/config/pose_pick.yaml`
2. **How far the tips reach below the hand point** (`fingertip_below_tcp`, metres): measure from the hand's
   mounting flange face to the lowest point of the tip, and subtract 103.4 mm (where the hand point sits). The
   stock tips are about 9 mm. `grasp_tcp_height` must be at least this plus the margin (about 3 mm more is good):
   the script refuses to start otherwise, because the tips would go through the bottom of the brick into the
   table, baseplate or the brick underneath.
3. **What the hand reads with the tips touching** (`hand_zero_width`): with nothing between the fingers run
   `ros2 run fr3_pose_pick measure_hand_zero.py` (type YES; it closes gently at 5 N, then opens again). It prints
   `hand_zero_width`, `open_width` and `grasp_width` for the config. Printed tips of 3 mm each read 6 mm, and
   the 16 mm brick is then gripped at 22 mm, not 16.
4. **Size of one tip** for the pallet check: `finger_tip_thickness` (the way the jaws close) and
   `finger_tip_width` (along the brick's long side), from calipers. The pallet check uses them to refuse
   spacings where a tip would hit the neighbouring brick when a brick is released.
5. **Centre the grasp**: pick with the inspection stop and nudge `x`, `y`, `z`, `yaw` until the tips look
   centred, `ok`. After 3 to 5 picks:
   `python3 src/fr3_pose_pick/scripts/measure_grasp_offset.py --last 5 --apply src/fr3_pose_pick/config/pose_pick.yaml`

## Lego baseplate on the table

A flat baseplate raises the surface the bricks stand on. Set `baseplate_thickness` (a standard Lego baseplate is
0.0032 m): the brick heights, the layers and the table in the planning scene all move up by it. Also set
`fingertip_table_margin: 0.0025` (the plate's studs stand 1.7 mm above its surface). Bricks seat on the studs, so
a brick released at its seated height (`place_margin: 0.0`) just touches them; a negative `place_margin` presses it
onto them (down to `-place_max_press`, 3 mm). Expect a seated brick to need a firm pull to lift off again.

## Pallet: bricks placed one after the other in a grid, each with a gap

`config/pallet.yaml` describes the grid (all in metres, 4 cm = 0.04): `origin` (centre of slot 1), `step_along`
(from one slot to the next in a row), `step_row` (from one row to the next), `per_row`, `rows`, `yaw_deg`,
`layer`, and optionally `area` (the baseplate edges). The file ships as a template with `needs_setup: true`;
the robot refuses to use it until you put in the real position of slot 1 (move the hand over its centre and read
`ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp`) and set `needs_setup: false`.

```bash
python3 src/fr3_pose_pick/scripts/pallet_tool.py check     # window, gaps between bricks, room for the finger tips
python3 src/fr3_pose_pick/scripts/pallet_tool.py status    # the grid: # filled  > next  . free
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --camera-xyz X Y Z --camera-angle A --pallet                  # pick what your teammate gave, place in the next slot
```
- Every run places ONE brick in the next free slot (a row, then the next row) and records it **as soon as the hand
  has let go**, so the next run goes to the next slot by itself, even if the back-off after the release fails.
- The state is kept in `config/pallet_state.json`, next to the pallet file (independent of the folder you run
  from). A damaged state file stops the run instead of starting again at slot 1.
- `--pallet-slot N` uses slot N (refused when it already holds a brick). `pallet_tool.py` can `mark N` a slot filled
  by hand, `unmark N` one you emptied, `skip` the next one (a bad spot) and `reset` everything.
- The pick can be anything: a camera point from your teammate (any height, stacks included), or `--pose`.
- Gaps: the check refuses a pallet whose bricks would be closer than `min_gap` (4 mm). Across the brick's width the
  gap must also leave room for a finger tip (`finger_tip_thickness`) next to the brick being released.
- **Accuracy:** the jaws centre the brick across its width, but not along its length. The placed brick lands
  at its slot plus however far off-centre it sat in the fingers along its long side: the pick error along the
  brick (2 to 3 mm from the camera, unless you correct it at the inspection stop). The spacing along a row is only
  as good as that.

## Autonomous version: `pose_pick_auto.py` (no questions, no corrections)

`pose_pick.py` stays the interactive one (inspection stops, step mode, corrections). `pose_pick_auto.py` is a separate script
that goes and picks and places on its own. It does not change `pose_pick.py`: it imports its planning and motion code and the
same config, so every locked value (grasp height, centring, place height, grasp force, baseplate) is used as it is.

    ros2 run fr3_pose_pick pose_pick_auto.py --config src/fr3_pose_pick/config/pose_pick.yaml \
        --camera-xyz X Y Z --camera-angle A          # same options as pose_pick.py, plus --countdown 3..60

- **Same pose by default.** With no place option the brick is put back down exactly where it was picked: same x, y, angle
  and the same layer (`--place-layers same`: a brick taken from on top of another goes back on top of it, a brick from the
  table goes back on the table), released at the locked place height (`place_margin`). The arm lands with the same small
  tracking error at the pick and the place (1 to 2 mm, measured: dx -1.5 dy +1.4 down, dx -1.5 dy +0.6 down at the place), so
  commanding the same pose returns the brick to the same spot within about 0.5 mm; the measured grasp pose is not used instead,
  that would count the error twice.
- **It trusts the gripper, not the pose.** After the hand closes and again after the lift it reads the finger width from
  `franka_gripper/joint_states` and stops when it does not fit a held brick (`grasp_check_tolerance` 3 mm around the brick width;
  `hold_check_tolerance` 1 mm change between the grasp and the end of the lift: a dropped brick lets the fingers close). The
  Franka Hand has no force sensor: its effort reads 0, so there is no force feedback, only the width and the grasp result.
  `pose_pick.py` prints the same widths and only warns. Every width goes to `pose_pick_auto_log.json` (`finger_width_mm`). `--place`, `--place-offset`, `--place-camera-xyz` or `--pallet` put it
  somewhere else instead (`--place-offset 0 0.05` is 5 cm to the robot's left).
- **No questions.** "Type GO" becomes a countdown (default 5 s; Ctrl+C or the e-stop cancels). The inspection stops, step mode
  and correction prompts are always off, `--step`, `--inspect` and `--place-inspect` are refused, and a prompt that still
  appears is never answered: the run stops before it moves.
- **Same safety.** The hand must start pointing down, bad poses and geometry are refused, every move is compared with where the
  hand really is (stop above 10 mm), and a downward move that ends short of its target never closes the hand or releases the
  brick in the air. It still passes the inspection height on the way down, only without stopping.
- **A failure stops and explains** (exit code 1). The arm is not moved to fix it, the hand is not opened or closed by itself,
  and a brick already in the hand stays there.
- **Own log:** every run (done or stopped, with the reason) goes to `pose_pick_auto_log.json` (`auto_log` in the config), never to
  `pose_pick_offsets.json`, so these runs do not dilute `measure_grasp_offset.py`.
- With `--camera-xyz` the hand pose of the picture is "the arm right now": run it right after the picture with the arm still,
  or pass `--ee-pose X Y Z QX QY QZ QW`. One brick per run; with `--pallet` each run takes the next free slot.
- After a fresh checkout run `colcon build` (or `ln -s` the script into `install/fr3_pose_pick/lib/fr3_pose_pick/`); it is in
  the install list of `CMakeLists.txt`.

## Emptying the whole pile: `sort_pile.py` (2026-10-27)

One command, no publisher or listener: look from the recorded `observe` pose, move the top layer (`pose_pick_stacks.py`, or
`--mode row` for the good row), go back to `observe`, look again, until the pile is empty:

    ros2 run fr3_pose_pick sort_pile.py --config src/fr3_pose_pick/config/pose_pick.yaml --review codex

Bricks the camera missed are seen again in the next round; when a layer is gone the next picture shows the one below. It stops DONE
after `--empty-scans` (2) pictures in a row with no brick, and stops at once on any failure, when only colours without a stack are left,
or after `--max-rounds` (40). Bricks seen on a colour stack's footprint (or outside `pile_area`, if set) are never picked. The moves to
`observe` are `go_to_pose.py observe` (also usable alone, with any recorded pose). The layer of each round comes from the pipeline's own
height above the plate (`--layers N` for the pick script), not the base-frame height, which slopes 1-2 deg with the hand-eye result (a
layer-2 brick read 2.73 layers and was grasped above, 16:49). A failed pick is RECOVERED (unless a brick is still in the hand): the
Franka `error_recovery` action clears the red light, then the next round goes back to `observe` and looks again; a brick that fails
twice is left alone, `--max-failures` (3) in a row stops the run. Moves to: straight up first if lower, then one straight line,
at the recorded hand angle itself (the camera is on the hand). `--dry-run` looks once and prints the pick command without picking.

## Sorting by colour into stacks: `pose_pick_stacks.py` / `stack_listener.py` (2026-10-27)

The same as palletizing, but every good brick goes onto the stack of ITS COLOUR instead of a row (bad bricks still go to the bin):

    ros2 run fr3_pose_pick stack_listener.py --config src/fr3_pose_pick/config/pose_pick.yaml       # instead of palletize_listener.py
    ros2 run fr3_pose_pick pose_pick_stacks.py --config src/fr3_pose_pick/config/pose_pick.yaml --status   # or --reset

Each stack's centre is recorded with the arm (hand pointing down over the middle of the stack, then `record_pose.py stack_red --config
...`: its x, y is the centre, the height does not matter) or set in `colour_stacks: {red: [x, y], ...}` (colour names as the pipeline
prints them; a good colour with no stack is left on the pile and said so; a colour in both is refused). A layer is `stack_side_by_side` (3, along base y) x `stack_end_to_end` (2, along
base x) bricks, long sides along base x, `stack_gap` (16 mm, 2 studs) between neighbours both ways: 80 x 80 mm. When it is full the next
layer goes straight on top, up to `stack_max_layers`. Each brick is let go with the hand at `palletize_row_drop_z` + (layer - 1) x
(9.6 mm + `stack_layer_gain`): a dropped brick rests on the studs, so a layer stands a stud higher than the model (gain 1.7 mm). The
stacks are remembered in `pose_pick_stacks.json`; a stack's centre is fixed once a brick is on it. `palletize_listener.py` and
`pose_pick_palletize.py` are unchanged; the stack listener refuses to start while another listener is on the topic.

## A camera error is fixed in the base frame: `camera_offset_base` (2026-10-27 11:40)

`tcp_correction_hand` is in the HAND frame and turns with the hand. The along value had gone -13.0 -> -19.5 mm by eye with the hand
at yaw 0 (6.5 mm toward the robot); the first lattice brick, picked at yaw -89, then missed by ~7 mm toward a finger: the 6.5 mm had
turned with the hand into base +y, and the error it fixed, in base x, came back across the brick. That error is in the camera
conversion (the hand-eye result is rejected, ~3 mm / 0.9 deg), so it went to `camera_offset_base: [-0.0065, 0.0]` (metres, base
frame, added to every camera point) and the hand value back to -13.0. A correction found by eye at ONE hand angle cannot tell the two
apart: check it on bricks at 0 and 90 deg; what stays the same in base x / y belongs in `camera_offset_base`, what turns with the hand
in `tcp_correction_hand`.

11:50: the next brick at 90 deg was grasped at hand yaw +89 (the hand takes whichever of +-89 is nearer), the rest of the hand value
turned the other way and one finger landed on each neighbouring brick, 16.7 mm across from where a base-frame correction puts it. So
ALL of it is now in `camera_offset_base: [-0.0195, -0.0040]` and `tcp_correction_hand: [0, 0]`: at yaw 0 the hand goes exactly where
it went before, and at +-90 the same base offset is used whichever way the hand turns.

## Camera pipeline -> topic -> palletizer (2026-10-27)

`brick_publisher.py` runs `vision/run_pipeline_classifier.sh` and publishes the bricks it finds on `/brick_poses`;
`palletize_listener.py` turns every message into one `pose_pick_palletize.py` run. Three terminals, all from the repository root after
`source env.sh`:

    ros2 launch fr3_pose_pick pose_bringup.launch.py robot_ip:=192.168.3.10                         # 1: robot + MoveIt
    ros2 run fr3_pose_pick palletize_listener.py --config src/fr3_pose_pick/config/pose_pick.yaml   # 2: waits for bricks
    ros2 run fr3_pose_pick brick_publisher.py --config src/fr3_pose_pick/config/pose_pick.yaml --review codex   # 3: per picture

- The publisher reads the hand pose from TF before and after the pipeline; if the arm moved (1 mm, 0.5 deg) nothing is published.
  That pose goes into the message and the listener passes it as `--ee-pose`, so the arm may be anywhere when the run starts.
- Options the publisher does not know go to the pipeline (`--review codex`, `--second-opinion codex`, `--recording DIR --frame N`; an
  old recording needs `--ee-pose`). `--pipeline` is the script (default `vision/run_pipeline_classifier.sh` of this repository).
- The message is a `std_msgs/String` with JSON: `ee_pose` [x y z qx qy qz qw] and `bricks`, each with `camera_xyz`, `angle_deg` and
  `label` G (good) / B (bad) from the classifier's quality. A brick with neither good nor bad is left out and said so.
- The listener does one message at a time and STOPS when a run fails, like `pose_pick_palletize.py`; start it again after looking.
  `--once` does one message, `--dry-run` only prints the command, options after `--` go to `pose_pick_palletize.py` (`-- --countdown 10`).
- Start the listener first: the publisher waits `--wait` (30 s) for it, and a message nobody hears is lost, never replayed.
- Take the next picture with the arm back at a picture pose after the run: the bricks of the stack below need a new picture.

## Closing the hand gently first, and recording what closing does (2026-10-26 20:20)

"One finger seems to dominate and nudges the brick." The Franka Hand's two fingers are driven together and move equally (the real hand
reported 0.0349177 m for each finger at rest). What shoves the brick is the hand landing a millimetre or two off the brick's middle, so
the first finger to touch pushes it, harder since the grip went from 10 N to 40 N. So `Mover.hand(close=True)` now closes in two stages:
slowly (`grasp_speed` 0.01 m/s) with only `grasp_force_first` (8 N), waits `grasp_settle` (0.3 s), then grips with `grasp_force` (40 N).
Opening is unchanged. `grasp_force_first: null` brings back the single stage.

The autonomous scripts also record what closing did, in the printout and in `pose_pick_auto_log.json` (`closing`): how far the HAND moved
while the fingers closed, across the brick and along it (hand frame), in base x and y, and where each finger ended. The fingers move
together, so a sideways shift of the hand means the brick was not in the middle: the sign says which side pushed, the size by how much.
If it is always the same side by about the same amount, that is the number to put into `tcp_correction_hand` (across, positive = base -y at
yaw 0) instead of guessing. Send the lines `Closing: the hand moved ...` of a few picks.

## The pick moved 2 mm toward -y, and the good row drops higher (2026-10-26 18:40)

The fingers hit the brick when picking (the hand landed 3 mm further toward -y than in the earlier picks: `dy=-1.8` at the grasp against
+0.9 to +1.7, and the fingers read 17.2 mm: the brick was caught crooked). The centring `tcp_correction_hand` across went 0.0045 -> 0.0065,
which is base y -2 mm for a hand at yaw 0 (the along part, -0.0130, is unchanged). It is in the config, so it applies to every pick; if a
pick still touches the brick, tell which side and by how much. The bin pose is not affected (the hand goes exactly to the recorded pose).

At the good row (y 0.27 to 0.33) the plate is higher than at the stack: bricks came to rest on the studs with the hand at z=0.0021 to
0.0052 there, against 0.0010 to 0.0018 at y 0 to 0.03 (log 18:14). The drop to z=0.001 at 18:36 hit the plate and the controller aborted
("execution failed (status 6, code -6)"). `palletize_row_drop_z: 0.0065` (highest rest 0.0052 + 1.3 mm) now says where the hand lets go
for a good row (null: 1 mm above the model's seat). A drop that still aborts says so and names the setting; the arm may then need its
errors recovered in Desk. A dropped brick no longer stores its release height as the row's seat height (that would have stacked every
next drop on the one before).

## Bad bricks go to a recorded BIN pose (2026-10-26 18:27)

Move the arm to where the bad bricks should be dropped (hand pointing down), hold it still, and record the pose under the name `bin`:

    ros2 run fr3_pose_pick record_pose.py bin --config src/fr3_pose_pick/config/pose_pick.yaml     # --show lists, --delete NAME forgets

`record_pose.py` only READS the hand pose from TF (about 20 readings over 2 s, refused if they differ by more than 1 mm), and saves the
average (position, quaternion, yaw, tilt) in `config/poses.yaml`, which is in git. The pose recorded on 2026-10-26 18:27 from the real
arm is x=0.3519 y=-0.1058 z=0.1831, yaw 0.22 deg, tilted 0.7 deg from pointing down. It was then nudged 2 mm toward -y on request
(`record_pose.py bin --shift 0 -0.002`): the file keeps the measurement (`xyz_recorded`) and every nudge (`nudges`), so the bin is now
x=0.3519 y=-0.1078 z=0.1831. `--shift DX DY [DZ]` takes metres, at most 2 cm per call, and refuses a nudge that makes the bin unusable.

A brick labelled **B** (`--brick X Y Z A B`, or `--label B`) is picked like any other, then carried to that pose (straight there if the
bin is higher than the carry height; from above and back up the same way if it is lower, `bin_clearance`), the hand opens, and the
run goes on with the next brick; the good ones keep filling the good row with no gap. The hand ends AT the recorded pose (the centring
offset of the grasp is taken off the commanded point). Refused before anything moves: a B brick when no bin is recorded, a bin pose with
the hand tilted more than `max_tilt_deg`, out of reach, within 3 cm of the table or above 0.55 m. `palletize_bad_mode: row` brings back the second
row for the bad bricks. The far moves to and from the bin run at `transit_velocity_scale`, the last bit down at the normal speed.

## Palletizing now drops the brick and sorts good from bad (2026-10-26 18:20)

Seating bricks on the studs was not reliable (the plate's studs are unknown to the robot), so `pose_pick_palletize.py` now puts a brick
down the simple way (`palletize_place_mode: drop`, the default): over the slot, down to 4 cm, nudge x and y by the measured hand
position, straight down to `palletize_drop_height` (1 mm) above the seat, let go, next brick. No press, no search, no blocked check;
a brick may rest on top of its studs. `palletize_place_mode: press` is the stud-seating place of before (first brick searches the stud
cell, later bricks search 2 rings, the anchor mode). The "Place pose ... release with the hand at z=-0.0008" line of the printout is the
press model and does not apply to a drop; the line of the brick header says "drop 1.0 mm above the seat".

Every brick carries a capital letter, **G = good, B = bad**:

    ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml \
        --brick X1 Y1 Z1 A1 G --brick X2 Y2 Z2 A2 B --brick X3 Y3 Z3 A3 G

(the letter is required in a batch, so a forgotten one cannot sort silently; the single form takes `--label G|B`, default G). Good
bricks go to the good row, 40 cm to the robot's left of the first pick (`palletize_start_offset`); bad bricks to the bad row, the same
40 cm and 10 cm closer to the robot (`palletize_bad_start_offset: [-0.10, 0.40]`). Both rows hang from the same reference, the hand's
measured position at the very first grasp, so they line up whichever label comes first. Each row keeps its own state file
(`pose_pick_palletize.json` for G, `pose_pick_palletize_bad.json` for B) and every row grows along +y, 32 mm per brick.
`--status` shows both, `--reset` forgets both, `--shift DX DY --label B` moves one row.

## Incremental palletizing: `pose_pick_palletize.py`

Same autonomous machinery as `pose_pick_auto.py` (no questions, countdown, every safety check, gripper width confirms the grasp), but
the bricks go into a ROW instead of back where they were picked:

    ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml --camera-xyz X Y Z --camera-angle A
    # or several bricks in one command, all from one picture pose: start it with the arm at that pose (it is remembered for
    # every brick), or give it with --ee-pose X Y Z QX QY QZ QW
    ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml \
        --brick X1 Y1 Z1 A1 --brick X2 Y2 Z2 A2 --brick X3 Y3 Z3 A3

- The first brick is set down `--start-offset` from where it was picked (default: `palletize_start_offset` in the config, now `0 0.40`:
  40 cm to the robot's left, along +y, clear of the stack), on the table / baseplate, at the angle it was picked at. The window
  `workspace_y` was widened to 0.55 for it (a slot beyond is refused before the arm moves). The studs of that plate are unknown, so a brick that
  sits on top of them is tried again over the whole 8 mm stud cell (`place_first_search_rings` 4 x `place_first_search_step` 1 mm = 81 spots,
  middle outwards, about 3.5 s each) and the row is taken from where it SEATS: position, angle, seat height. The next row bricks search `place_row_search_rings` (2 = 25 spots, +-2 mm) around where the row says the studs are, and every
  brick that seats refines the row origin (the mean of all seated bricks): the 17:55 run seated slots 1 and 2 after 1 and 3 shifted
  tries and then missed slot 3 by a fraction of a millimetre. A 16 mm gap is two studs, so every next slot is on the grid too. Only a new row reads the offset. Every next brick goes NEXT to the one before, along +y, `--gap` apart (default 16 mm, so the
  centres are 32 mm apart), long side along x (`--row-yaw 0`).
- The row is remembered in `pose_pick_palletize.json` (gitignored): each command, or each `--brick`, takes the next slot. Slot 1 is
  wherever the first brick really ended up (the measured hand position at the release); the others are exact multiples of the pitch from
  it, so the row stays straight even though every brick is picked with the camera's 3 mm of error. The offset and angle are fixed by
  the first brick; the gap can still be changed while at most one brick is down (`--gap 0.017`, only the next slots move); `--reset`
  starts a new row. A brick taken from a stack exposes the next one, which needs a new picture: run one brick per command then.
- `--status` shows the row and the room left, `--shift DX DY` moves the whole row by up to 1 cm (a few mm if the first brick did not
  line up with the baseplate's studs), `--reset` forgets it. None of them moves the arm.
- If `finger_tip_thickness` / `finger_tip_width` are set in the config, the script checks that a finger fits between two bricks `--gap`
  apart; otherwise it says that was not checked.
- A failure stops the batch at that brick and says how many bricks are already in the row; the slot is taken at the release, before
  the back-off, so a failed back-off does not lose it.

## Starting the row from an anchor brick (why the first brick would not seat), speed, countdown

The 17:36 palletizing runs aligned the hand to 0.02 mm and still stopped 2.0 to 2.6 mm above the seat, at nearly the same height after
every sideways retry (z=0.0012 to 0.0018): the brick sat on top of the studs and a shift of 0.7 to 1.0 mm did not find them. The robot
cannot see the studs of the plate the row is on; they repeat every 8 mm (0.098 m is 12.25 of them) and the plate may be turned a degree or
two from the stack's, while the put-back onto the stack worked because it returns to a seat the brick had just left. The sideways
"back and forth in x" is that retry (`place_search_*`: lift, shift 0.7 mm, press again, now 4 tries instead of 6), not a sensor.

The fix does not guess the grid: press ONE brick onto the row's plate by hand, where the row should start, and give its camera numbers:

    ros2 run fr3_pose_pick pose_pick_palletize.py --config src/fr3_pose_pick/config/pose_pick.yaml --anchor X Y Z A

The robot picks it up and puts it back (the seat-to-seat put-back that works) and measures from it: the row origin is the hand position
at the grasp, the row angle is the hand angle at the grasp, `seat_z` is the hand height at the grasp. The anchor is slot 1; the next
bricks go one pitch (brick width + gap) further along +y, at that angle and that seat height (the press goes to `seat_z + place_margin`,
refused if it is more than 4 mm from the model). Needs an empty row (`--reset` first). `--status` shows "measured from the anchor brick".

Also in this version: the far moves (hover, lift, carry, back off) run at `transit_velocity_scale` (0.10, at most 0.2) while the approach,
the alignments and the press keep `velocity_scale` (0.05); the countdown is once per command (the next `--brick` of a command follow at
once); and a press that ends more than `-place_margin + 0.8 mm` (1.8 mm) above its target counts as "on the studs" (it was 2.0 mm, which
let +2.0 pass as seated).

## The place by measured position (why the brick sat on top of the studs)

The 16:23 autonomous run put a layer-2 brick back 1.7 mm high, ON its studs (hand stopped at z=0.0115, seat 0.0098): the arm
had landed (-0.6, +1.1) mm from its target at the pick but (+0.8, +0.3) at the place, because the place came down in one 10 cm line
and the pick in two steps, and a brick needs about 0.5 mm to drop onto its studs. The place in `pose_pick_auto.py` and
`pose_pick_palletize.py` now: records the hand's MEASURED position at the pick's 4 cm stop and at the grasp; at the place comes
down to the same 4 cm stop and nudges x and y (to `place_align_tolerance` 1 mm); then comes down to a SECOND stop 8 mm above the seat
(`place_fine_height`) and nudges again (to `place_fine_tolerance` 0.3 mm), because a long descent drags the hand sideways by a different
amount at every place (0.4 mm at the pick, 2.4 mm 10 cm away: the 17:23 palletizing run lost 2.5 mm that way and the brick sat on the
studs) while the last 8 mm cannot; presses the brick in; and if it still stops short (it sits on the studs) lifts 8 mm and tries again shifted by
`place_search_step` (0.7 mm) in a small spiral, up to `place_search_tries` times, before it stops with the brick still in the hand.
Everything is in the log (`place`: alignment left, drift, tries). Not tried on the real arm yet.

## The place stopped at 95 %: the planning-scene table sits 3.5 mm low (`scene_table_sink`)

A place with `place_margin: -0.001` presses the brick 1 mm onto the studs. MoveIt checks the attached brick against the table box
of the planning scene, called that a collision at the last waypoint of the lowering, and answered "straight line not possible
(reached 95 %)". The scene table is now `scene_table_sink` (3.5 mm, at least `place_max_press`) below the real surface; only the
planning scene changes, every height the code checks still uses `table_top_z`. The same run also showed that "the same pose"
put a brick taken from layer 2 down on the table: fixed with `--place-layers same`, and the smoothing spline no longer fails on
short 5-point moves (it picked degree 4, which scipy refuses, and fell back to the old timing).

## Locked settings (2026-10-26, real robot, new thin tips, two baseplates)

These came from a real, successful pick and place and are stored in `config/pose_pick.yaml`; do not re-tune them by hand:

    baseplate_thickness: 0.0032      the plates under the bricks
    grasp_tcp_height:    0.017       tips touch at hand z = -0.0005, the grasp stops at +0.0002 (was 0.012: +5 mm)
    tcp_correction_hand: [-0.0130, 0.0045]   centring along / across the long side (second run: x -3.0 mm, y -0.5 mm more)
    place_margin:        -0.001      release 1 mm BELOW the seat (pressed onto the studs): -2 mm typed at the stop, then 1 mm more

The release height follows `grasp_tcp_height` (+5 mm with the pick), so the place is: grasp z +0.0002, release z -0.0008.
The pick of the second run (photo taken with the hand at z=0.226) needed 3 mm more along x than the first (hand at z=0.187):
if a later pick is off by a steady amount, take the photo with the hand at the same height as these two (about 0.19 to 0.23)
before changing the centring again. The place may still be pressed 2 mm lower at the stop (total press limit 3 mm).
`measure_grasp_offset.py` and the config check now accept a centring up to 20 mm (it was 10 mm).

    grasp_force:         40.0        was 10 N. A 2x4 seated on the baseplate takes roughly 10-25 N to pull off (about 2 N per
                                     stud); two fingers hold 2 x friction x force, so 40 N has a safe margin. The config check
                                     now accepts up to 50 N (the Franka Hand is rated 70 N). If the printed tips flex or the
                                     brick creaks, go down in steps of 5 N; if the brick still slips while lifting, up to 50 N.

## The hand stops on the way down: it is never closed on, and the brick is never released in the air

The last move before the grasp (and before the release) is a straight vertical line. If the hand ends it more than
`blocked_height` (2 mm) short of the target or pushed more than `blocked_sideways` (3.5 mm) aside, something is in the
way: a finger on top of the brick, the tips on the plate or its studs, a neighbouring brick. Before this check the run
only printed a warning and closed the hand anyway. Now it prints the height where the hand stopped, **does not close**
(or release), lifts straight back to the inspection stop and asks for a correction (3 tries), at the pick and at the place.
Every such event is written to the log (`blocked`).

Finding out what blocks it (one minute, with a ruler): at the inspection stop, before `ok`, measure the vertical gap g
between the lowest point of the tips and the plate (the surface between the bricks). With the stop height T printed in
`stop at z=...` (and `table_top_z` = the plate surface, -0.0168 with `baseplate_thickness: 0.0032`):

    fingertip_below_tcp  L = (T - table_top_z) - g          e.g. T = 0.0352, g = 40 mm  ->  L = 12 mm
    grasp_tcp_height     = L + 0.003                         the fingertips then stop 3 mm above the bottom of the brick

    sed -i 's/^fingertip_below_tcp:.*/fingertip_below_tcp: 0.012/' src/fr3_pose_pick/config/pose_pick.yaml
    sed -i 's/^grasp_tcp_height:.*/grasp_tcp_height: 0.015/' src/fr3_pose_pick/config/pose_pick.yaml

If the tips are on top of the brick instead (look from the side), it is alignment, not height: use `pick_open_width`.

`pick_open_width` (default off): open the hand wider for the pick descent only, e.g. `pick_open_width: 0.026`. The jaws then
land on both sides of the brick with 5 mm to spare instead of 1 mm, and closing to `grasp_width` centres the brick across.
Use it when nothing stands within ~1 cm of the brick's long sides. The hand still opens to `open_width` at the release.

## Correct the place by eye (x, y, z, yaw) before the brick is lowered

Like the pick, the place has an inspection stop. After the brick is carried above the place spot it comes down to
`place_inspect_height` (3 cm) **above where it will be put down** and waits at `place correction>`:

```
z -0.002          2 mm LOWER  (z is up: negative = down)
x 0.002 y -0.001  move it 2 mm away from the robot / 1 mm to the robot's right
yaw 3             turn the brick 3 degrees
ok                lower it, let go and back off      q   stop here: the brick stays in the hand, nothing is released
```
- On a real robot it is on by default (as the pick inspection); `--no-place-inspect` turns it off, `--place-inspect` forces it
  on (also in simulation). It works with `--place`, `--place-offset`, `--place-camera-xyz` and `--pallet`.
- Limits, as at the pick: 10 mm in x and y, 5 mm in z and 15 degrees per command; 30 mm, 15 mm and 45 degrees in total.
- **It will not press the brick into the surface:** the brick may be pressed at most `place_max_press` (3 mm) below its seat
  in total, `place_margin` (shipped: -1 mm, already pressed 1 mm) and the correction together: with the shipped values `z` down
  to -2 mm. If the surface really is lower than planned, fix
  `table_top_z` / `baseplate_thickness` instead. It also refuses a correction that would leave the allowed window.
- The correction applies to this brick only; it is written to `pose_pick_offsets.json` (`place_correction`: dx, dy, dz, yaw).
  A z that every place needs belongs in `place_margin` (signed: positive above the seat, negative pressed onto it).
- The back-off after the release starts from the corrected position.

## Smooth, straight-line motion (no jerks)

Every move is a straight Cartesian line (MoveIt's `compute_cartesian_path`): the hand stays within 0.01 to 0.05 mm of
the line. The jerks came from somewhere else: the planner's inverse kinematics lets the 7-axis arm's elbow wander by
several milliradians from point to point (up to 15 mrad, and 2 to 3 joints reversing direction in the middle of a move)
while the hand stays on the line, and the velocities worked out from that noise made the arm jerk and the hand speed
vary by about +-25 %.

`smooth_motion: true` (default, `config/pose_pick.yaml`) fixes this in `trajectory_smoothing.py`:
1. the hand stays on exactly the planned poses; only the wobble of the elbow is removed (the poses are re-imposed with
   the arm's own kinematics, which match MoveIt's to the sixth decimal);
2. a quintic spline through them (continuous velocity and acceleration);
3. ONE constant hand speed, the fastest that keeps every joint under `velocity_scale` times its limit at the worst point
   of the move, with smooth quintic ramps: speed and acceleration are zero at the start and the end, the jerk is finite;
4. positions, velocities and accelerations every 50 ms, so the controller follows a quintic spline.

Measured on the real MoveIt plans of seven typical moves (0.05 speed): peak jerk 49 to 461 rad/s3 before, 0.3 to 2.0
after; peak acceleration 0.16 to 0.86 before, 0.16 to 0.28 after; the long hover at a constant 37 mm/s (before
53 +- 13 mm/s); the hand 0.01 mm off the line. The long 40 cm hover takes about 12 s instead of 9 s, because it now moves
at one steady speed set by the worst point of the path; short moves take the same time. Faster: raise `velocity_scale`.

Every move is checked before it is sent: the smoothed path must stay within 0.5 mm of the planned line and 0.6 mrad of the
planned orientation (checked between the planned points too), inside the planned range of joint angles, and under the speed
limit. If any check fails, the planner's own points are sent with the old timing and a warning is printed. The step
prompt shows `(smooth, constant N mm/s, on the line within X mm)` for a smoothed move. `smooth_motion: false` switches back.

## Safety: one robot stack per ROS domain

A simulation and the real robot must never run in the same ROS domain (a command can reach either).
`pose_pick.py` refuses to start when it sees two `move_group` / `controller_manager` /
`robot_state_publisher` nodes. For a simulation next to the real robot use another domain
(`export ROS_DOMAIN_ID=78` in every simulation terminal).

## Real robot

1. FCI active, e-stop in reach, hand homed. Tell your teammates: the robot is shared.
2. Terminal 1: `ros2 launch fr3_pose_pick pose_bringup.launch.py robot_ip:=192.168.3.10`
3. Move the arm to a **ready pose**: hand pointing down, above the table, roughly over the brick. The script
   refuses to start if the hand is tilted more than `max_tilt_deg` and never swings the arm there itself.
4. Terminal 2: the same command as above without `--yes`. Type `GO`. Step mode and the inspection stop are
   on: Enter = do the move, `a` = run the rest, `q` = stop. The hand is never opened automatically after
   an error. On a real hand the grasp only reports success if it closed on something.
