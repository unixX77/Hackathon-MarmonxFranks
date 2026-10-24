# One separated brick: stopped refinement

This is a preparation workflow with a synthetic demo and a deliberately incomplete
real configuration. It does not change `cartesian_jog.py`, `task.yaml`, or the
existing six-stage runner. There is no continuous velocity/visual-servo loop.

`vision.launch.py` starts MoveIt (optional) and an observation service. The
interactive `vision_pick_place.py` runner is started in another terminal so its
confirmation prompts can read keyboard input. Neither the launch nor worker
sends arm/gripper goals. The runner defaults to preview only; `--execute` is
required for a cycle. Preview mode stops after the observation-pose preview
because subsequent detections require actual/simulated motion.

## Offline demo

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1
colcon build --packages-select fr3_pick_place franka_fr3_moveit_config
source install/setup.bash
ros2 launch fr3_pick_place vision.launch.py
```

Stop other robot/MoveIt launches first. Defaults are fake hardware and mock
perception. No camera, FCI activation, model weights, or camera calibration is
needed. `vision_demo.yaml` contains synthetic scene/pose values, not measurements
for a real arm. In another sourced terminal:

```bash
ros2 run fr3_pick_place vision_pick_place.py \
  --config src/fr3_pick_place/config/vision_demo.yaml --execute
```

Review each preview in RViz and press an empty Enter to execute; `q` exits.
The simulated arm approaches the mock brick, obtains repeated observations,
descends, models a grasp/attachment, transfers, models release and retreats.
Mock gripper events do not actuate a physical hand and do not simulate contact
or verify that an object was actually grasped. For an unattended mock smoke
test only, add `--auto-confirm`; this is rejected on a real arm.

The mock observation reports a stationary known object in base coordinates, with
a 6 mm bias on the first observation to exercise one correction. It
tests orchestration and geometry, not SAM3, image quality or calibration. Use
`use_rviz:=false` for a headless launch, or `start_moveit:=false` when a compatible
MoveIt stack is already running. Do not start duplicate controller managers.

## Before connecting to real hardware

Fill in `config/vision_real.yaml` from measurements. Null values intentionally
prevent startup. Required items:

- Brick dimensions, table box and attached camera/bracket box. The camera box
  is expressed in `fr3_hand_tcp`; it includes the mounting bracket and margins.
- A taught observation TCP pose and a taught release TCP pose, including their
  quaternion orientations. Both use `fr3_link0`.
- `object_tcp_grasp`: the desired TCP pose expressed in the object's coordinates
  (object X=long side, Y=short side, Z=up). Tool Z must point down. Its translation
  is a measured grasp-depth offset, not automatically zero.
- `camera_extrinsic`: maps optical-frame coordinates into its named wrist parent:
  `p_parent = T_parent_optical @ p_optical`. Supply the exact optical frame used by
  RGB/aligned depth/CameraInfo, not `camera_link`. Compose the driver's internal
  transform if your supplied calibration is for the camera root. This worker
  uses the transform directly; it does not publish a competing optical TF parent.
- Validate the extrinsic independently before setting `calibration_validated:
  true`. This flag records your validation; software cannot establish accuracy.
- Gripper settings, adequate transfer clearance, and measured camera depth limits.
  Confirm the release pose leaves the held object's bottom above the support
  surface with suitable release clearance; the code does not drive into contact.

Version 1 requires a horizontal table, one separated regular unsplit brick, and
a vertical tool. Refinement corrects only X/Y and yaw at the established height;
it does not lower between observations. It requires several stable estimates
before a final descent bounded by `final_descent_max_m`. Ensure valid depth and
visibility at the refinement height; it must remain outside the camera's close
range limit. If it cannot observe there, change the physical setup/observation
strategy rather than bypassing measurement gates.

Start the RealSense ROS driver separately with aligned depth enabled. Only one
process should own the camera. Check the exact topic names, optical frame,
16UC1 depth encoding, depth scale and intrinsics. This adapter preserves the
teammate geometry's rectified/zero-distortion requirement. Its RGB, aligned depth
and CameraInfo are synchronized within 30 ms. No new RealSense driver launch is
included because its installed package/version is not assumed.

The workspace's user-site NumPy 2 conflicts with its system OpenCV build. The
demo uses the ROS system packages via `PYTHONNOUSERSITE=1`; set this in the
runner terminal too. Do not mix incompatible NumPy/OpenCV/cv_bridge builds in
the SAM3 environment.

The SAM3 worker needs a Python environment with ROS modules (`rclpy`, tf2,
cv_bridge, message_filters), NumPy/OpenCV/SciPy, PyTorch, Pillow and a transformers
version supporting `Sam3Model`/`Sam3Processor`. Model access/weights may require
your team's existing Hugging Face setup. No dependencies or model weights are
automatically installed. Use `perception_python:=/absolute/path/to/python` to
select that environment. The model is loaded once; robot control stays in the
normal ROS Python environment.

With FCI activated and no other hardware stack running:

```bash
ros2 launch fr3_pick_place vision.launch.py \
  config:=/absolute/path/to/config/vision_real.yaml \
  use_fake_hardware:=false robot_ip:=YOUR_ROBOT_IP \
  perception_python:=/absolute/path/to/sam3/python
```

In another terminal, first preview:

```bash
ros2 run fr3_pick_place vision_pick_place.py \
  --config src/fr3_pick_place/config/vision_real.yaml
```

Only after verifying the hardware, geometry and poses, repeat with `--execute`.
The runner detects real/mock hardware from controller_manager's model and
refuses mock perception on a real arm. Hardware motion requires a measured
table/camera scene, spline interpolation and fresh joint measurements. Real
gripper homing must already be completed. Other fixtures must be represented
in the shared MoveIt scene. Do not move the scene or operate another controller
while a stage is planned/approved/executed. Keep the physical stop accessible.

## Perception and control contract

The worker exposes `vision/observe` (`std_srvs/Trigger`). A real request waits
for a synchronized capture newer than the request. It saves the capture-time
base-to-camera transform before running inference. The robot waits stationary
through inference; the runner checks freshness and measured joint drift.

Successful responses contain JSON with `schema: 1`, `backend`, `frame_id`,
`capture_stamp_ns`, and one object containing `xyz_m`, `quaternion_xyzw`,
`size_m`, `colour`, and diagnostic `source_id`. Real detections include score,
depth and plane `metrics`. Position is the estimated box centre, not top-face
centre. Instance IDs are not tracked across frames.
Association uses base-frame position, colour and bounded displacement;
180-degree rectangular yaw ambiguity is resolved against the previous pose.
Ambiguity/loss, invalid depth, bad plane fit, partial/split/irregular detections,
border truncation or iteration exhaustion abort before another move.

The three files in `scripts/vision_vendor` come unchanged from the teammate ZIP;
see `PROVENANCE.md`. The adapter calls `segment`, `filter_top`, and
`apply_size_prior`, then restores the score/depth metrics the original JSON
export omitted. `quality` remains a colour-derived label and is never used as
pose confidence. No old bundled result JSON supplies motion targets.

World object and TCP poses remain separate. After a successful grasp, attachment
uses the measured TCP-to-object transform. On release, the object is returned
to the world at the predicted held-object pose. Finger/object contact is allowed.
Brick/table contact is permitted only during the validated upward lift from its
support, after checking the estimated brick does not penetrate the table by more
than 1 mm. That pair is revoked after the lift or a lift failure. Other robot/table
contacts remain collision-checked. These objects persist for inspection; after an abort there is no
automatic release/recovery. Inspect any remaining attachment before restarting.

Every arm plan is previewed before execution. Vertical segments reuse the
existing timed-spline vertical validation. Transfers and refinement also sample
timed trajectories for joint limits and collision validity, with an additional
horizontal/tool-down corridor for corrections. The current scene is checked
again after approval, and start/end measurements are checked. These are sampled
model checks, not physical clearance, contact or grasp guarantees.

## Validation

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
/usr/bin/python3 -m unittest discover -s src/fr3_pick_place/test -p 'test_vision.py' -v
```

Tests cover transform composition, top/box centre semantics in the actual
teammate fitter, grasp offset, bounded corrections, stable-observation gates,
association, yaw ambiguity, failure sequencing and missing measurements.
The mock demo is an integration test; real camera inference and physical grasps
require separate testing with your team and robot. Run the isolated headless
full-cycle check with:

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 src/fr3_pick_place/test/vision_mock_smoke.py
```

It creates a separate ROS domain, launches only fake hardware and mock perception,
auto-confirms the mock runner, checks correction and final placed-object state,
and shuts down its processes. Logs are retained in a printed `/tmp` directory.
