# Wrist-mounted D435: eye-in-hand calibration

Recommended first run: **collect manually positioned, stationary poses; calculate
the transform offline.** `collect` and `solve` never create motion clients.
No calibration is possible from the statement that the camera points in the
gripper's direction: both rotation and translation must be estimated.

The camera and bracket must be rigid relative to the selected `ee_frame`.
Attach to the hand body or wrist, not a moving finger; a finger-mounted camera
changes its transform as the jaws move. Secure the USB cable with enough slack
for the intended wrist rotations. Set the correct added payload/centre of mass
using the robot manufacturer's procedure. Keep the hardware stop accessible.
These scripts cannot guarantee collision-free hardware operation.

## 1. Build, camera stream and board

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select fr3_pick_place --symlink-install
source install/setup.bash
```

This machine's user NumPy 2 installation conflicts with system OpenCV.
The calibration executable uses `/usr/bin/python3 -s`, which successfully loads
the installed system OpenCV 4.5.4 / NumPy 1.21.5 and ROS vision dependencies.
For direct invocation use `/usr/bin/python3 -s`; avoid changing global packages.
If dependencies are missing on another machine, install the ROS Humble
`cv_bridge`, `tf2_ros` and `sensor_msgs` packages plus distribution packages
`python3-opencv` (with `cv2.aruco`), `python3-numpy`, `python3-scipy`, `python3-yaml`.

The RealSense ROS driver must be installed separately. A typical current driver
launch is:

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true
ros2 topic list
```

Set the actual color `image_raw` and `camera_info` topics in
`config/handeye.yaml`. Defaults are `/camera/camera/color/image_raw` and
`/camera/camera/color/camera_info`; older driver releases may omit a `camera`
namespace. Use raw color and its matching intrinsics, not depth intrinsics.
Depth is not needed for calibration. Match `camera_mount_frame` to the RealSense
root (usually `camera_link`); the driver must publish its internal optical TFs.
Do not add an approximate wrist-to-camera TF just to make calibration work.
The collector only needs base-to-EE and camera-root-to-optical TF chains, which
can remain disconnected until calibration is complete.

Generate a target (a copy is included as `config/aruco_board.png`):

```bash
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml board --output /tmp/aruco_board.png
```

The included PNG design has 5 by 4 ArUco `DICT_4X4_50` markers, IDs 0–19, black marker edge
35 mm, gap 10 mm. The black grid extent is **215 by 170 mm**; with the white
border the image should print **235 by 190 mm** (254 dpi). Use A4 landscape,
disable fit-to-page, and **measure the printed black edges and gaps** with a
ruler. Update the two metric lengths in YAML if necessary. Mount the print
flat on a rigid surface and secure it to the table. Do not move the board,
robot base, or camera mount during a dataset.

For your current print, the measured black edge is **34 mm**. YAML now records
`marker_length: 0.034`; `marker_separation` is intentionally unset until measured.
If the print was scaled uniformly, the expected gap is `10 * 34 / 35 = 9.714 mm`,
but check the print rather than assuming this. Measure the full grid horizontally
and vertically between its outer black edges, excluding the white border:
`horizontal gap = (grid width - 5 * marker width) / 4`,
`vertical gap = (grid height - 4 * marker height) / 3`.
Check several marker edges in both directions too. The board model assumes square
markers and equal horizontal/vertical spacing; reprint if scaling is visibly
unequal. Fill in the measured average gap in metres before collecting. The
original PNG has not been regenerated. Board generation also requires a complete
board configuration and uses its current dimensions, not the original design.

## 2. Position and capture without automatic motion

Provide live `/joint_states` and base-to-EE TF from actual measured hardware
joint states; do not use fake hardware for a real calibration. Default parent:
`fr3_hand_tcp`; `fr3_link8` is another possible choice, but use the same frame
for the entire dataset and later integration. Its pose must come from forward
kinematics, not a previous guessed camera transform.

Use the robot's supported hand-guiding/teaching mode **only with a compatible
read-only state/TF setup**. Do not push against an active position/trajectory
controller or assume that running the MoveIt launch enables hand-guiding.
FCI mode and Desk teaching are not interchangeable. If live state capture is
unavailable in guiding mode, move through individually reviewed slow MoveIt
poses instead, then capture after each move has stopped. Never run two motion
controllers competing for the arm.

```bash
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml collect --output handeye_samples.json
```

At each stationary pose press Enter. Capture waits for a one-second joint
stability window, detects the board, uses the image timestamp for the robot TF,
and saves robot pose, board pose, intrinsics, joint positions and a grayscale
image. It rejects stale images, mismatched frames/resolution, more than 2 mrad
joint drift, image/joint timestamp separation over 50 ms, too few markers,
ambiguous planar PnP and reprojection error over the configured pixel threshold.
A rejected capture does not send any robot commands. Type `q` when finished.
Use a new output filename for each dataset; overwriting is refused.

Aim for **20–30 usable views** (minimum 12). Keep the board large and in focus
in the image. Vary position, distance and **rotation about at least two axes**:
use modest pitch, roll and yaw changes within a clearly unobstructed workspace,
not large wrist sweeps. Translations with a fixed downward orientation cannot
determine the complete transform. Near-frontoparallel board views may be
rejected as ambiguous; a modest tilt helps. More poses do not compensate for
poor pose diversity or an incorrectly printed board scale.

## 3. Solve and validate

```bash
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml solve --input handeye_samples.json --output handeye_result.json
```

OpenCV's Park hand-eye method solves for `T_ee_camera`, mapping camera optical
coordinates into EE coordinates:

```text
p_ee   = T_ee_camera @ p_camera
p_base = T_base_ee @ T_ee_camera @ p_camera

T_base_target[i] = T_base_ee[i] @ T_ee_camera @ T_camera_target[i]
```

The last expression should stay constant because the board is fixed. The
solver checks rotation diversity and fits a separate estimate while holding
out every fifth sample. It reports held-out and all-sample position/orientation
consistency. Default maximum deviations are 5 mm / 1 degree; these are screening
thresholds, **not guarantees of absolute accuracy**. Incorrect board scale,
intrinsics, clock offsets or common systematic errors can still produce a
plausible fit. Rejected results exit with code 2 and supply no TF command.

Before using the result for robot motion, capture a **second independent set**
with the board unchanged and check that its reconstructed base-frame board pose
stays constant under the first result. Capture at least five diverse new poses,
then evaluate the original transform without refitting:

```bash
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml collect --output validation_samples.json
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml validate --input validation_samples.json --calibration handeye_result.json --output validation_report.json
```

Verify a known physical board dimension
and a measured base-frame target position as well. Recalibrate after any
mount movement. Review saved images if residuals are high; do not increase the
thresholds simply to get an accepted result.

The result stores translation (metres), quaternion **xyzw**, the 4x4 matrix and
quality statistics. It also calculates `T_ee_mount` for the RealSense root via
the driver's known root-to-color-optical transform. A passing result contains
a `mount_tf_command_after_validation`; nothing publishes TF automatically.
Publish only **EE → camera root**, preserving the driver's root → optical
subtree. Remove any previous competing parent/static transform first: publishing
EE → color optical directly would conflict with RealSense's existing TF parent.

For future depth use, ensure points really are expressed in the calibrated
color optical frame. Raw depth uses a different optical frame; transform it
through the driver's depth/color extrinsics, or use the appropriate aligned
depth stream and matching projection model.

## 4. Optional replay of taught poses

This is an advanced option after the manual collection works. **No random
poses or automatic workspace exploration are generated.** It reuses joint
poses from a previously collected dataset and always plans the full sequence
before any execution. Reaching a taught endpoint does not imply that the path
between endpoints is safe. The initial path from the current pose also matters.

Measure conservative axis-aligned collision envelopes in YAML for the mounted
camera/bracket (EE frame), table and board/support (base frame). Include margins.
If a box is too conservative, reposition the setup or improve the geometry;
do not broadly disable robot collision checking. `camera_touch_links` should
list only the actual mounting contact link. These boxes cannot model cable
snagging, people, or every real obstacle. Add all other fixtures to MoveIt too.

With your normal hardware MoveIt stack running:

```bash
# Changes only the planning scene; does not move hardware.
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml scene
# Plans only. Inspect camera/table/board geometry and EVERY path in RViz.
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml replay --input handeye_samples.json --output replay_samples.json
```

Both replay modes refuse to plan if the named attached-camera/table/board
collision geometry is absent. Only after checking the full swept paths, start
from a known clear pose, keep people out of the workspace, and supervise:

```bash
ros2 run fr3_pick_place handeye_calibrate.py --config src/fr3_pick_place/config/handeye.yaml replay --input handeye_samples.json --output replay_samples.json --execute
```

`--execute` enables real arm motion. There are no gripper commands. It plans at
3% velocity/acceleration scaling, executes the exact plans it just computed,
checks the measured start joints before each trajectory, waits for settling,
then captures. On planning, action or capture failure it aborts the sequence
without retrying or changing collision/safety thresholds. The board must be
visible before the first move; intermediate visibility is checked at capture,
not continuously during motion. Planning-only previews and execution runs may
produce different paths, since the latter replans before starting. Static
geometry is assumed unchanged during a run. Ctrl-C requests cancellation of an
accepted active goal on a best-effort basis; the hardware stop remains necessary.
Do not treat software cancellation or low scaling as a certified safety measure.
Run `solve` on `replay_samples.json` afterwards to calculate its transform.

## Validation performed in this workspace

Offline tests recover a known synthetic transform and the camera-root conversion,
reject single-axis/translation-only datasets, reject an inconsistent target,
and recover board pose from a rendered image. Build and ROS message construction
checks are also run. These do not validate real camera accuracy, guiding mode,
physical collision geometry, or hardware trajectory execution.

References: [OpenCV hand-eye calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html),
[RealSense ROS driver](https://github.com/realsenseai/realsense-ros),
[Franka FCI overview](https://frankarobotics.github.io/docs/overview.html).
