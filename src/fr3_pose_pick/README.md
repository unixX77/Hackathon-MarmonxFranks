# fr3_pose_pick: pose-driven pick and place (no vision)

You give the poses, the arm does the rest. Nothing here looks for the brick: no camera, no detection,
no simulated brick. The camera-based pipeline is the sibling package `fr3_visual_pick`.

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

## Simulation (RViz, no brick)

Every terminal: `source ~/hackathon_repo/env.sh` (it sets `ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77`).

```bash
# terminal 1
ros2 launch fr3_pose_pick pose_bringup.launch.py use_fake_hardware:=true
# terminal 2
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --pose 0.45 0.10 --yaw 30 --place 0.40 -0.20 --place-yaw 20 --yes --step --inspect
```
The simulated hand (`sim_gripper.py`) animates the fingers in RViz; there is no brick, so a grasp just
closes to `grasp_width`. RViz shows the pick pose as a green box and the place pose as a blue box. The
carried brick appears in the planning scene while it is in the hand, so the arm avoids hitting the table
with it. In simulation the proof of accuracy is the `arrived: ... mm` line after every move.

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
- The converted top face must be where a brick on the table has it (within 8 mm) or it refuses;
  `--use-measured-z` trusts the measured height instead (for a brick on something).
- The hand-eye result missed its strict gate (typical error about 3 mm / 0.9 deg), so expect a few mm
  of error from the conversion alone: use the inspection stop to correct it.

## Hand widths (fixed in the config)

`config/pose_pick.yaml` sets the hand once for every open and close: **open 17 mm** (`open_width`) and
**close 15 mm** (`grasp_width`), for a 16 mm brick with other bricks right next to it. Nothing to type
on the command line. `min_open_clearance: 0.001` lowers the config check from its default 5 mm margin
to the 1 mm that 17 mm leaves. That is only 0.5 mm of room per side, so the script prints a note at each
start, and the hand has to be lined up to well under 1 mm sideways and in yaw: use the inspection stop,
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

## Bricks lying on other bricks (stacks)

The brick height in the config (`block_size[2]`, 9.6 mm) is one layer. Layer 1 is on the table, layer 2 is on
top of one brick, and so on. Say which layer the brick is in:

```bash
# pick the brick in layer 2 of a stack, put it down 7 cm to the left as layer 3 (on top of two bricks)
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --camera-xyz X Y Z --layers 2 --place-offset 0 0.07 --place-layers 3
```
- `--layers N` (pick) and `--place-layers N` (put down) work with `--camera-xyz` / `--place-camera-xyz`, with
  typed `--pose` / `--place X Y`, and with `--place-offset`. With a typed pose they replace the Z.
- With a camera point, the measured height is only a **cross-check**: it must be within
  `table_check_tolerance` (8 mm) of that layer's top face or the script refuses. The grasp then uses the exact
  layer height, so the 3 to 7 mm of camera height error does not move the fingers.
- Without `--layers` the brick is assumed to be on the table, as before. If the check fails, the message says how
  many layers the measured height corresponds to, so you can try `--layers N`.
- `--layers auto` works the layer out from the camera height, and refuses when the height is not within
  `layer_auto_tolerance` (3 mm) of one layer. A layer is 9.6 mm and the camera height is often 3 to 7 mm off, so
  `auto` often refuses: then say the count yourself.
- **Limit:** an 8 mm check cannot always tell a wrong layer count from camera error (layers are 9.6 mm apart).
  The inspection stop is the real check: look at the fingers' height against the stack before you type `ok`.
- The camera needs at least about 28 cm to the brick to measure depth reliably (a RealSense D435). With a tall
  stack, keep the hand correspondingly higher when the picture is taken. A rule of thumb for a brick on the table:
  the depth in the camera numbers is about hand height + 0.055 m, minus the height of the stack.
- Fingertips reach about 9 mm below the hand point, so they grip the upper part of the brick and stay clear of
  the brick below. The released brick is removed from the planning scene, and the bricks of a stack are not in it:
  the hand comes straight down beside the brick, so keep other bricks out of the 17 mm of finger travel.

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
