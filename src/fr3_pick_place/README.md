# FR3 measured-position pick and place

For the wrist-mounted RealSense D435 calibration workflow, see [HANDEYE.md](HANDEYE.md).

ROS 2 Humble, this workspace's FR3 MoveIt configuration, and a Franka Hand.
The default mode plans six sequential arm motions without sending any arm or
gripper execution commands. Each planning-only stage starts from the previous
plan's endpoint and models the changing finger width and carried block.

## Configure

Fill in `config/task.yaml`. All lengths are metres, in `fr3_link0` (robot base).
`grasp_width` is set to your tested **0.015 m** jaw opening. `block_size` is
separate: provide approximate measured dimensions for collision checking.
`pick` and `place` are the block's **centre**. Measure the table box centre and
dimensions, including its thickness. The block's X/Y axes must align with the
base X/Y axes; the fingers close across its Y dimension. The tool orientation
is fixed to quaternion `(x=1, y=0, z=0, w=0)` with tool Z pointing down.
The grasp places `fr3_hand_tcp` at the block centre: verify finger depth against
your actual block and table. Additional fixtures and obstacles must already
be represented in the MoveIt planning scene.

The program adds the table and block to each request's planning-scene diff.
Only the fingers may contact the block. The block is attached to the TCP in
the carrying phases and placed at the destination for retreat planning.
It does not permanently add these objects to the shared planning scene.

## Build and preview

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select fr3_pick_place --symlink-install
source install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=unused use_fake_hardware:=true
```

In a second terminal with the same ROS environment:

```bash
ros2 run fr3_pick_place pick_place.py --config src/fr3_pick_place/config/task.yaml
```

MoveIt's planned trajectories are published for RViz. Review each phase.
These are collision-checked pose-goal plans, **not guaranteed straight-line
Cartesian descents**. Inspect the complete paths and clearance. Planning with
fake hardware does not validate real calibration, grip, timing, or dynamics.

## Hardware

Stop the fake-hardware launch. Use the controller's supported FCI setup, with
the workspace's required real-time host configuration. Keep the stop device
accessible and the workspace clear for the first slow trial. The hand should
already be homed; this program does not run homing automatically.

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=YOUR_ROBOT_IP use_fake_hardware:=false
```

Run planning-only against the live state first. To command a physical cycle:

```bash
ros2 run fr3_pick_place pick_place.py --config src/fr3_pick_place/config/task.yaml --execute
```

Execution replans each stage from the live state, checks arm and grasp results,
and stops the sequence on failure. It does not automatically open the hand on
failure, since a block may be held. Inspect the hardware before retrying; there
is no automatic recovery. Action cancellation on timeout/Ctrl-C is best effort
and is not a replacement for the hardware stop. Fake gripper mode supplies
joint states only, so `--execute` requires the real hand action servers.

Default namespaces and link names match this workspace. ROS remapping can
select another node namespace; custom robot/link prefixes require code changes.

## Pose-driven pick (no camera): `pose_pick.py`

A teammate gives the brick pose; the arm hovers above it, comes down to an inspection stop where
you check the offset and correct it in small steps, lowers, closes the hand and lifts straight back
up. Nothing is placed. Settings are in `config/pose_pick.yaml` (brick size, table, accepted
workspace window, grasp height, clearance, reliability limits).

**Real robot, step by step**

1. FCI active, e-stop in reach, hand homed. Launch (starts MoveIt, RViz, the Franka hand, the table):
   `ros2 launch fr3_pick_place lego_bringup.launch.py robot_ip:=192.168.3.10 spawn_brick:=false`
2. Move the arm to a **ready pose**: hand pointing down, above the table, roughly over the brick
   (RViz MotionPlanning, or guiding). The script refuses to start if the hand is tilted more than
   `max_tilt_deg` and never swings the arm there itself.
3. Run it (source `install/setup.bash` first). Either pose input:
   ```bash
   ros2 run fr3_pick_place pose_pick.py --config src/fr3_pick_place/config/pose_pick.yaml \
        --pose 0.45 0.10 --yaw 20          # brick centre x y [z] in fr3_link0 (m); long-side angle (deg)
   ros2 run fr3_pick_place pose_pick.py --config src/fr3_pick_place/config/pose_pick.yaml \
        --camera-xyz 0.035 0.059 0.144 --yaw 20   # top-face point in the camera frame, from the picture
   ```
   `--camera-xyz` converts with your hand-eye file (`charuco_result.json`) and the arm's current
   pose, so the arm must still be where the picture was taken. The converted top face must agree
   with the table model (8 mm) or it refuses. `--quat X Y Z W` can replace `--yaw`.
4. Type `GO`. Every move is shown first; Enter does it, `a` runs the rest, `q` stops. After each move
   the real hand position is compared with the target (warn above 3 mm, stop above 10 mm).
5. At the **inspection stop** (4 cm above the grasp) look at the fingers against the brick and type
   corrections, e.g. `x 0.002`, `y -0.001 yaw 5` (base axes: +x away from the robot, +y to its left;
   at most 10 mm / 15 degrees at a time, 30 mm in total). `ok` lowers and grasps, `q` stops there.
6. It opens the hand early, grasps, lifts, and prints and logs the correction it needed
   (`pose_pick_offsets.json`): that is how much the given pose was off.

A pose outside `workspace_x/y`, one that would push the fingertips into the table, or any move that
is not a clean straight line or turns a joint more than `max_joint_swing` is refused instead of
improvised. RViz shows the (corrected) brick as the green `BrickEstimate` box. Other pose options:
`--pose-is-tcp` (z is the hand height), `--use-measured-z` (camera mode, brick not on the table),
`--no-inspect`, `--no-step`. Simulation (private ROS domain, add `--yes --inspect --step`):
`ros2 launch fr3_pick_place lego_bringup.launch.py use_fake_hardware:=true sim_world:=true
spawn_brick:=false`.

### Pick **and place** without vision: `--place`

Add `--place X Y [--place-yaw DEG]` and after the lift the hand carries the brick at hover height to
above the place pose, lowers it until the brick touches the table (`place_margin` above it), opens,
and backs off straight up. RViz shows the pick pose (green) and place pose (blue). In simulation the
script then reads the brick's true pose from `/sim_brick_marker` and prints `SIM CHECK ... PASS/FAIL`
(limits `sim_check_xy_tol` / `sim_check_yaw_tol_deg`); exit code 2 means FAIL.
`scripts/sim_truth.py` prints the exact simulated brick pose as `--pose X Y --yaw DEG`.
Write tiny numbers as `-0.00001`, not `-1e-05` (argparse reads the latter as a flag).

```bash
ros2 run fr3_pick_place pose_pick.py --config src/fr3_pick_place/config/pose_pick.yaml \
     --pose 0.51844 -0.01604 --yaw -43.43 --place 0.40 -0.20 --place-yaw 20 --yes --no-step --no-inspect
```

### Getting the pose from a camera click: `pixel_to_base.py`

Click the brick centre in the RGB picture, read the depth there, and get the brick pose in
`fr3_link0` using your hand-eye file (`charuco_result.json`): `T_base_ee (FK) @ T_ee_camera @
p_camera`. The full 4x4 chain is used, so a tilted camera is fine. The arm must stand still from the
picture until you read the answer. The RealSense needs `align_depth.enable:=true`.

```bash
ros2 run fr3_pick_place pixel_to_base.py --config src/fr3_pick_place/config/pose_pick.yaml
#   SPACE = freeze, click the brick centre (then the end of the long side for the yaw), Enter = accept
ros2 run fr3_pick_place pixel_to_base.py --config ... --pixel 640 360 --yaw 20     # no window
ros2 run fr3_pick_place pixel_to_base.py --config ... --pixel 640 360 --depth 0.42 \
     --intrinsics 910 905 645 358 --ee-pose 0.4 0 0.45 1 0 0 0                      # offline numbers
```

It prints the top-face point, the brick centre (half a brick height lower), and a **table check**:
a brick lying on the table must have its top face at `table_top_z + brick height` (-0.0104 m).
More than 8 mm off means calibration, intrinsics, depth or arm movement is wrong. It ends with the
matching `pose_pick.py` command.

## Autonomous Lego pick and place (camera watching continuously)

`lego_auto.py` searches for the brick from several scan poses, then steps down through hover
heights, re-aiming at the brick from the camera stream at each stop, grasps, carries it to
`--place-xy`, releases and checks where it landed. Settings are in `config/lego.yaml`
(`scan_poses`, `servo_heights`, ...). It asks you to type `GO` before any motion.

Arm motion is straight Cartesian lines only (`smooth_motion.py`): collision checked, smooth speed
profile, and the wrist always turns the short way (the jaws are symmetric, so a brick angle and
that angle + 180 degrees are the same grasp). A move that cannot be done as a clean line is refused.
In RViz the red box is the true simulated brick, the green box is where the camera thinks it is.

**Simulation (no robot, no camera).** `sim_world.py` provides a fake wrist camera, a fake hand and
a movable brick; the brick shows in RViz as the `SimBrick` marker, the camera's estimate as the green `BrickEstimate` box, the camera picture in the
`Camera` panel, and the fingers open and close. The brick sits at a random place and angle that only
`sim_world` knows, with decoys (a red wrong-size object, a blue brick) that must be ignored. The simulated camera is deliberately a few mm off its calibration
(`sim_cal_error`), so the result is not perfect. Use a private ROS domain so it cannot reach a real
robot or other people's sessions:

```bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77
ros2 launch fr3_pick_place lego_bringup.launch.py use_fake_hardware:=true sim_world:=true
# second terminal, same two exports:
ros2 run fr3_pick_place lego_auto.py --config src/fr3_pick_place/config/lego.yaml --place-xy 0.45 -0.20 --yes
```

`--yes` skips the GO prompt only when `sim_world` is running. The launch also swaps the fake arm
to a position controller (`config/sim_arm_controller.yaml`), because the stock effort controller
does not move the fake hardware.

**Step mode.** On a real robot `lego_auto.py` asks before every arm move and hand action: it plans
the move without executing it, prints the line length, hand yaw, biggest joint turn and duration,
and waits. Enter = do it, `a` = run the rest automatically, `q` = stop (the arm stays where it is,
the hand is not touched). `--no-step` turns it off; `--step` turns it on in simulation.

**Real robot.** Camera running, then
`ros2 launch fr3_pick_place lego_bringup.launch.py spawn_brick:=false` and the same
`lego_auto.py` command without `--yes`.
