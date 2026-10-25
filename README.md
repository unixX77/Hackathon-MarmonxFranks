# FR3 structured LEGO sorting integration

The requested task is now a **300-brick, ten-layer interlocked source lattice**,
rebuilt as one red lattice with non-red bricks dropped into a trash box.
See [the red lattice task](docs/RED_LATTICE.md) and use
`src/lego_sorter/config/lattice_red.yaml` for its mock simulation. Stud extraction,
stud seating and real trash-box handling remain unimplemented and hardware startup
rejects that task. The 60-brick example described below is the original demo.

Implemented workspace: `/home/tanay/Documents/Hackathon/project-planner`. It contains two ROS packages (`lego_sort_interfaces`, `lego_sorter`), three persistent nodes, a full coordinator, transactional destination allocation, a persistent SAM3/D435 subprocess adapter, and an executor-safe adaptation of Ameed’s Cartesian motion implementation. Default mode is **mock**. Launching the default configuration does not connect to a robot or camera; a mission requires an explicit `start` request.

The verified demo empties **60 bricks: two layers × six rows × five columns**, builds three alternating-orientation colour stacks, and places 24 unwanted/unknown bricks in discard cells. It is not a one-brick demo. Simulated results always identify themselves as simulated. Hardware errors never fall back to mock success.

## Build and test

Use the system ROS Python, not the SAM3 environment. On this machine user-site NumPy 2 conflicts with the apt OpenCV binary; `PYTHONNOUSERSITE=1` (or `/usr/bin/python3 -s`) is required. All paths below are absolute or relative to the explicitly selected integration root.

```bash
cd /home/tanay/Documents/Hackathon/project-planner
bash scripts/build.sh
bash scripts/test.sh
```

If the Franka sources have been copied into this workspace's `src`, build from a terminal that has not sourced `hackathon_ros2_ws`; the build script rejects that old overlay to avoid duplicate packages. It disables optional dependency development tests (`BUILD_TESTS=OFF` for libfranka and `BUILD_TESTING=OFF` for ament packages), so Google Mock and dependency lint tools are not required for this runtime build. The planner's own checks still run through `scripts/test.sh`. After building, source `/opt/ros/humble/setup.bash` and this workspace's `install/setup.bash`.

The build needs ROS 2 Humble, ament/colcon, `rosidl_default_generators`, `geometry_msgs`, `builtin_interfaces`; runtime needs `rclpy`, `tf2_ros`, `sensor_msgs`, `std_msgs`, `moveit_msgs`, `shape_msgs`, `action_msgs`, NumPy, SciPy, PyYAML, and OpenCV. System dependencies are installed outside the project only by your environment administrator; these scripts do not install or modify dependencies.

The test script runs the unit/integration suite, then launches **all three actual ROS nodes** on localhost in ROS domain 174, requests a mission, and asserts 60 committed placements, 24 discard placements and an empty hand. It stops only the processes it creates. Its configuration, event logs, node logs, persistent state and result are under `runtime/ros-smoke-<timestamp>/`. Use another test domain if 174 is occupied.

Build, install and colcon logs are explicitly routed to this directory. No reference workspace is built or sourced by these scripts. Reference repositories were clean when inspected and remain untouched. This directory did not exist before implementation and was not an existing Git repository; no reference branch was switched or merged.

## Inspect, preview and run without hardware

```bash
cd /home/tanay/Documents/Hackathon/project-planner
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src/lego_sorter${PYTHONPATH:+:$PYTHONPATH}"
export ROS_HOME="$PWD/runtime/ros_home" ROS_LOG_DIR="$PWD/runtime/ros_logs"
/usr/bin/python3 -s -m lego_sorter.cli inspect
/usr/bin/python3 -s -m lego_sorter.cli preview
# Use a NEW directory for a separate mock physical world. Never erase hardware state to resume.
/usr/bin/python3 -s -m lego_sorter.cli mock --runtime "$PWD/runtime/my-new-mock"
```

Open `runtime/demo/preview/layout.html` for a layered, hoverable layout preview; `layout.json` gives exact slot coordinates. The CLI inspection and preview commands do not send motion commands. A separate optional hardware `inspection_mode: true` adds a stop at Ameed’s configured inspection height before each descent. Inspect `/manipulation_status` for `inspection_waiting: true`, then send `task_control` command `approve_inspection` to continue. Timeout/cancellation stops without closing/opening the hand; no automatic approval or positional nudges are inferred. Reusing a runtime directory preserves the world and transactions, including uncertain outcomes.

To run the ROS loop interactively:

```bash
cd /home/tanay/Documents/Hackathon/project-planner
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export ROS_HOME="$PWD/runtime/ros_home" ROS_LOG_DIR="$PWD/runtime/ros_logs"
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch lego_sorter sorter.launch.py config:="$PWD/src/lego_sorter/config/sorter.yaml"
```

In another terminal with the same exports and overlays:

```bash
ros2 service call /task_control lego_sort_interfaces/srv/TaskControl "{command: start}"
ros2 topic echo /sorter_status
ros2 service call /task_control lego_sort_interfaces/srv/TaskControl "{command: pause}"
ros2 service call /task_control lego_sort_interfaces/srv/TaskControl "{command: resume}"
ros2 service call /task_control lego_sort_interfaces/srv/TaskControl "{command: stop}"
```

`pause` finishes the current pick/place/verification transaction and pauses before the next brick. `stop` requests cancellation immediately, latches manipulation stopped, preserves the physical hand state and marks unresolved destinations uncertain. It is a software stop, not a replacement for the robot’s physical emergency stop. Neither command automatically opens the hand. Resume requires a reconciled empty hand, stationary fresh feedback and no uncertain/reserved destinations, then moves to the observation pose and takes a new image. It never replays a partially completed grasp/place.

## Replay

`mode: replay` reads normalized observation records listed in `replay_observations`. Manipulation and verification remain explicitly simulated, using the configured source geometry as the mock physical world. Replay brick centres must therefore match that world; it is not an unvalidated hardware motion mode. Each record has `mode: replay`, `bricks`, `empty_verified`, `evidence`, and `diagnostics`; IDs/timestamps are reissued per observation. Exhausting records is an error, never proof of an empty source. Real captures also save `normalized_observation.json` for inspection/replay.

Generate a complete runnable synthetic replay (61 snapshots, including verified empty) with:

```bash
cd /home/tanay/Documents/Hackathon/project-planner
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src/lego_sorter${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/python3 -s scripts/generate_replay.py
# Launch using the printed absolute config path, then explicitly call task_control start.
```

## Geometry and policy assumptions — confirm these physically

The shipped geometry is an **illustrative, explicitly uncommissioned configuration**. `geometry_confirmed: false` and `hardware_validated: false` block hardware startup. The preview is the review artifact, not a claim about your measured arrangement.

* A source layer is six rows of five bricks. Columns run along each brick’s long axis; rows run across its short axis. Row edge-to-edge gap is `row_gap_studs × stud_pitch` = 2 × 8 mm. `brick_gap` defaults to zero along columns.
* `origin` XY is the centre of the entire layer footprint; origin Z is its bottom support surface. Layer `k` is at `origin.z + (k + 0.5) × brick_height`. Adjacent layers rotate **the entire row lattice**, not just each brick, by 90° about the centre. If the real stack uses different layer origins, use zero-indexed `layer_overrides` with `origin` and `yaw_deg`. Layout yaw currently supports orthogonal orientations only and rejects others.
* One dimension model drives detector prior, object transforms, grasp height, slot generation and scene objects. Default `[0.032, 0.016, 0.0096]` m follows Ameed. The detector originally used `[0.030, 0.015, 0.010]`, the separate modular exporter uses `[0.0318, 0.0158, 0.0096]`, and your alternative task YAML uses `[0.0315, 0.0155, 0.0095]`. Measure your bricks; these are not silently treated as equivalent.
* Example destinations are individual cross-oriented towers with a minimum 50% support-area criterion. This is a geometric feasibility model, **not validation of physical stack stability or stud seating**. Configure larger row/column layouts and stricter `minimum_support_fraction` as needed; cells that lack sufficient PLACED support are skipped. Alternating a whole gapped grid can leave unsupported cells; capacity must be checked in preview and testing.
* `finger_length`, `finger_thickness`, jaw opening, workspace, source/destination origins, support planes and `fixtures` must be measured. Add the actual discard tray floor/walls, base pad and other obstacles as `fixtures` boxes. Hardware requires at least an explicit `discard_tray` fixture. Set discard origin Z to the tray’s measured support surface. The default mock treats the discard region as a tray-sized table-surface grid.

Policy modes: `single` routes only `policy.colour` to `selected`; `separate` routes listed `policy.colours` to matching destination names; `alternating` builds `diverse` using the exact repeating `policy.sequence`. In alternating mode an out-of-turn colour goes to discard rather than being stored for later; the sequence advances only on a verified commit. Unknown/low-confidence colours always go to discard. Detector `quality` is retained as diagnostic data; colour policy, not the original palette’s good/bad label, determines destination. Capacity exhaustion stops safely instead of dropping a brick or overwriting a slot.

Finger clearance is evaluated on the highest observed layer. Bounding boxes conservatively approximate measured non-axis-aligned candidates; configured layers use exact orthogonal boxes. Destination validation checks occupied cells, source keep-out geometry, workspace, jaw envelope, capacity, and exact overlap area with the previous PLACED layer. No reservation is made on uncertain support.

## Hardware environment and commissioning

No physical robot motion was started during development or tests. The actual D435, SAM3 inference, calibration, gripper feedback, MoveIt scene/contact behaviour and physical placement **remain to be validated**.

Overlay order is: `/opt/ros/humble/setup.bash` → **one** installed compatible Franka/MoveIt bringup overlay → `project-planner/install/setup.bash`. Do not source both reference manipulation workspaces: they contain duplicate `fr3_pick_place` packages. The integration uses unique names and copies no Franka dependency trees. `franka_msgs` was absent from the base environment here; hardware needs its `Grasp` and `Move` actions and the actual driver/controller stack. Bring up the robot/MoveIt separately under your established procedure; this launch deliberately does not start a driver or initiate motion.

Make a new config **inside this directory**, give hardware its own persistent runtime directory, set `mode: hardware`, and provide:

* Measured geometry, observation TCP pose (XYZ + down-tool yaw), workspace, fixtures, transfer height, force/speed, clearances, dimension model and a reviewed source/destination preview.
* `calibration_file`, `calibration_parent`, `camera_frame`, `calibration_matrix_key`. Calibration files are read, never rewritten. `status: passed_quality_gates` is required unless `allow_rejected_calibration` is explicitly enabled. Ameed’s reference YAML opted into a rejected calibration; this integration does **not** inherit that opt-in.
* `sam_python` pointing to a **separate** environment containing torch, transformers with `Sam3Model`/`Sam3Processor`, Pillow, NumPy, OpenCV, PyYAML and pyrealsense2. The reference default `/home/tanay/miniconda3/envs/sam3/bin/python` is not present here. `model_path` must be an explicitly provisioned local SAM3 model; the worker uses `local_files_only=True`, with no silent model substitution/download. New model/cache/environment files should be provisioned inside this directory to retain the requested containment.
* D435 serial and optional `camera_settings` containing RealSense **advanced-settings JSON**. Close RealSense Viewer and any other capture process. The worker owns the D435 continuously; do not also launch a RealSense ROS camera driver for the same device. Camera stream is aligned 640×480 BGR8/Z16 at 30 Hz. It requires a globally synchronized RealSense timestamp and refuses stale/unmapped clocks.
* `verification: operator`, then explicit `geometry_confirmed: true` and `hardware_validated: true` only after reviewing the commissioning items. These are configuration acknowledgements, not evidence that testing has occurred.

The robot must start in a posture from which collision-checked down-tool Cartesian movement to the observation pose is feasible. There is no automatic recovery, homing, controller restart, or invented force routine. Robot ownership is shared across both actions, observation movement and capture leases; a process lock prevents a second manipulation process using the same runtime. Exclusive robot ownership also requires not running unrelated motion clients; a local ROS process cannot prevent an external controller from commanding the arm. Joint feedback is monitored for freshness throughout actions and for motion throughout capture/inference.

## Frames and corrections

The selected detector is **only** `perception/sam3_detection/top_layer_bricks.py`, used by `run_pipeline.sh`. The modular pipeline’s `camera-frame-brick-boxes-v1` schema is rejected. The adapter consumes structured records, never printed coordinates.

`centre_cam_m` is the top-face centre; `box_centre_cam_m` is the volume centre. Rotation columns are long side, short side, up, in colour optical coordinates (+X right, +Y down, +Z forward). A capture stores ROS/system capture timestamp, base→calibration-parent transform and parent→camera transform. It looks up TF **at capture time before inference**; latest TF is never substituted. Composition is exactly `T_base_parent @ T_parent_camera @ T_camera_object`.

Ameed’s calibration is TCP-relative (`fr3_hand_tcp`, `T_ee_camera`); your alternative contains flange-relative (`fr3_link8`, `T_link8_camera_color_optical_frame`) as well as derived matrices. Set parent and matrix key consistently. The loader checks declared parent/child and rigid rotation; do not rename frames to bypass it. A converted file must carry the correct parent/child/status metadata.

The detector’s printed down-tool quaternion is not a base-frame tool command. Here the transformed object long axis defines base yaw, with a tilt rejection gate. Ameed’s tool-down convention `Rz(yaw) Rx(pi)` and two symmetric wrist-yaw candidates are retained. `camera_height_bias` is subtracted **once** in base Z; the exposed reference value is 3.7 mm. Optional layer snapping uses the configured source origin. `grasp_tcp_height` converts centre to TCP **once**; `tcp_correction_hand` is applied **once** by the motion adapter. Transporting a measured TCP point removes that correction before reusing the corrected-move helper. `place_margin` is a nonnegative release clearance; it never presses into the support surface. The actual attachment transform is measured from TCP at grasp for collision modelling.

## APIs, concurrency and state

* `ObservePile` is a bounded snapshot **service**: it provides one atomic observation, with no user-facing progress/cancellation requirement. Model loading is persistent and outside callbacks; inference and camera operations run in a separate Python subprocess. ROS callbacks await thread-pool futures; a multithreaded executor and reentrant client groups keep TF/action responses flowing. No nested `spin_until_future_complete`/`spin_once` from the copied baseline is used in hardware operation.
* `PickBrick` and `PlaceBrick` are the two manipulation actions. Feedback gives stage and hand state; results distinguish success, failure before grasp, grasp uncertainty, known holding, release uncertainty, operator stop and validation/busy failures. A successful pick leaves HOLDING; a successful place leaves EMPTY. Admission rejects wrong hand state, stale observation epoch, concurrent ownership and unsupported stud engagement.
* `MotionControl` exposes `observe_pose`, capture begin/end leases, status, stop, resume and explicit empty-hand reconciliation. Observation motion uses the same Mover and ownership lock. Any manipulation/camera movement invalidates observations. Source IDs are valid only within their observation UUID.
* `TaskControl` manages start/pause/resume/stop and operator evidence. `VerifyPlacement` checks a transaction-specific receipt in hardware; mock uses explicit simulated world occupancy. A gripper-open acknowledgement alone does not commit hardware placement.
* Empty detections are inconclusive. Hardware empty verification requires the entire configured source footprint to project inside the image, ≥98% valid depth, ≥500 source pixels, and 99.9% of measured heights within `empty_plane_tolerance` of the bare support plane. Occlusion, missing depth and raised material fail this test. This geometric verifier itself needs D435/calibration validation.

`slots.json` uses EMPTY / RESERVED / PLACED / UNCERTAIN. Writes use a unique temporary file, file fsync, atomic rename and directory fsync; a writer lock excludes concurrent coordinators. RESERVED becomes UNCERTAIN on restart. Layout/policy fingerprints prevent accidental reuse under changed geometry. `hand.json` is written before irreversible hand commands; interrupted hardware runs start UNRECONCILED. `scene.json` retains removed source cells and released destination obstacles. `events.jsonl`, `manipulation.jsonl`, per-capture images/metadata and ROS logs remain inside the runtime directory.

Hardware placement verification is requested in `events.jsonl` as `verification_required`, with slot and transaction IDs. While the coordinator waits:

```bash
ros2 service call /task_control lego_sort_interfaces/srv/TaskControl \
  "{command: 'verify red/0/0/0 TRANSACTION_ID', evidence: 'Visually confirmed stable brick in reserved slot, hand clear'}"
```

A timeout marks the slot UNCERTAIN. After an uncertain stop, physically inspect/recover the robot without relying on this coordinator to release anything. Then explicitly attest the hand is empty:

```bash
ros2 service call /task_control lego_sort_interfaces/srv/TaskControl \
  "{command: reconcile_empty, evidence: 'Operator inspected hand after manual recovery; hand is empty'}"
ros2 service call /task_control lego_sort_interfaces/srv/TaskControl \
  "{command: 'resolve_slot red/0/0/0 PLACED', evidence: 'Operator confirmed this slot contains the released brick'}"
# Use EMPTY instead only when physically confirmed empty; then resume takes a new observation.
```

There is deliberately no automatic “resume while holding” path. A held brick can be placed via the explicit PlaceBrick action only while its context and reserved destination remain valid; after an uncertain/crashed run, use operator recovery and reconcile the world. If manual recovery changes source/destination geometry, reconcile `scene.json` and the configuration while nodes are stopped; do not reset state files to suppress errors.

## Adaptation and remaining validation

`docs/provenance.json` records exact reference commits, source hashes, file paths and the narrowly extracted helpers. `vendor/ameed_motion.py` preserves Ameed’s baseline; `hardware.py` overrides its spinning waits, gripper/trajectory execution, scene handling, arrival checks and ownership use. `ameed_plans.py` extracts only six pure planning/height helpers. Three detector modules are copied without their datasets/models or the alternative modular pipeline. Your alternative pick/place implementation informed fresh-feedback gates, explicit uncertainty and transactional placement design; it was not substituted for Ameed’s implementation. No entire branch/repository was merged.

Before hardware use, validate hand-eye quality and parent conventions, measured dimensions/height corrections, timestamp synchronization, finger envelopes/contact allowances, table/tray geometry, action cancellation/terminal acknowledgements, collision scene attachments, controller timing, maximum useful stack height and stability, source-empty depth thresholds, and operator verification workflow. Ameed’s velocity retiming remains the baseline (its `acceleration_scale` setting is not an independent acceleration-limit enforcement mechanism). The optional singularity/Jacobian audit from your alternative implementation was inspected but is not ported; no singularity-avoidance guarantee is claimed. MoveIt may refuse a path and require operator intervention.

Stud engagement is **unsupported**: this system implements supported release placement only. `engage_studs: true` and goals requesting engagement are refused rather than inventing a seating/press controller.
