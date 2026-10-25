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
