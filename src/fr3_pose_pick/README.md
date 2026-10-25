# fr3_pose_pick: pose-driven pick and place (no vision)

A pose for the brick comes in (typed, from the perception team, or from a camera click via
`pixel_to_base.py`), the arm picks it and, with `--place`, puts it down at another pose.
No detection runs here. The camera-based pipeline is the sibling package `fr3_visual_pick`.

## Pose-driven pick: `pose_pick.py`

A teammate gives the brick pose; the arm hovers above it, comes down to an inspection stop where
you check the offset and correct it in small steps, lowers, closes the hand and lifts straight back
up. Nothing is placed. Settings are in `config/pose_pick.yaml` (brick size, table, accepted
workspace window, grasp height, clearance, reliability limits).

**Real robot, step by step**

1. FCI active, e-stop in reach, hand homed. Launch (starts MoveIt, RViz, the Franka hand, the table):
   `ros2 launch fr3_pose_pick lego_bringup.launch.py robot_ip:=192.168.3.10 spawn_brick:=false`
2. Move the arm to a **ready pose**: hand pointing down, above the table, roughly over the brick
   (RViz MotionPlanning, or guiding). The script refuses to start if the hand is tilted more than
   `max_tilt_deg` and never swings the arm there itself.
3. Run it (source `install/setup.bash` first). Either pose input:
   ```bash
   ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
        --pose 0.45 0.10 --yaw 20          # brick centre x y [z] in fr3_link0 (m); long-side angle (deg)
   ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
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
`ros2 launch fr3_pose_pick lego_bringup.launch.py use_fake_hardware:=true sim_world:=true
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
ros2 run fr3_pose_pick pose_pick.py --config src/fr3_pose_pick/config/pose_pick.yaml \
     --pose 0.51844 -0.01604 --yaw -43.43 --place 0.40 -0.20 --place-yaw 20 --yes --no-step --no-inspect
```

### Getting the pose from a camera click: `pixel_to_base.py`

Click the brick centre in the RGB picture, read the depth there, and get the brick pose in
`fr3_link0` using your hand-eye file (`charuco_result.json`): `T_base_ee (FK) @ T_ee_camera @
p_camera`. The full 4x4 chain is used, so a tilted camera is fine. The arm must stand still from the
picture until you read the answer. The RealSense needs `align_depth.enable:=true`.

```bash
ros2 run fr3_pose_pick pixel_to_base.py --config src/fr3_pose_pick/config/pose_pick.yaml
#   SPACE = freeze, click the brick centre (then the end of the long side for the yaw), Enter = accept
ros2 run fr3_pose_pick pixel_to_base.py --config ... --pixel 640 360 --yaw 20     # no window
ros2 run fr3_pose_pick pixel_to_base.py --config ... --pixel 640 360 --depth 0.42 \
     --intrinsics 910 905 645 358 --ee-pose 0.4 0 0.45 1 0 0 0                      # offline numbers
```

It prints the top-face point, the brick centre (half a brick height lower), and a **table check**:
a brick lying on the table must have its top face at `table_top_z + brick height` (-0.0104 m).
More than 8 mm off means calibration, intrinsics, depth or arm movement is wrong. It ends with the
matching `pose_pick.py` command.

