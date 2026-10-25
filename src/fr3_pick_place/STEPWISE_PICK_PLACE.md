# Pick and place from taught TCP poses

Run from the workspace root. The staged cycle is plan-only by default and uses
`config/task.yaml`. Its pick/place values are TCP targets, with the orientation
copied from `taught_tcp_poses.json`. The optional `--poses` argument reads that
JSON directly, including a derived place target with no recorded joint angles.
The poses must use `fr3_link0` / `fr3_hand_tcp` and the same orientation.

Configured setup:

- Block: 31.5 mm X × 15.5 mm Y × 9.5 mm Z, resting on the tabletop. Its centre
  is assumed directly above the taught TCP in X/Y. Verify this box in RViz;
  knowing the grasp worked does not independently measure the object's pose.
- Table: 2 m square centered beneath the base, top at Z = -20 mm.
- Initial total jaw gap: 80 mm. Grasp: 15 mm, 10 mm/s, 10 N, inner/outer
  epsilon 2 mm, matching the operator-tested action.
- Clearance: 100 mm upward in base Z; velocity/acceleration scaling: 0.02.
- Release clearance: `place_release_clearance_m: 0.002`; open the fingers with
  the block nominally 2 mm above the table, letting it settle onto the surface.
- Place: proposed target 50 mm along +Y from pick, not physically taught.

The collision box centre is separate from the TCP target. At pick its Z is
-15.25 mm, half the block height above the tabletop. The configured world offset
from the taught TCP to this centre is approximately -8.899 mm Z. Attachment
geometry is transformed into TCP coordinates. In hardware mode a successful
grasp captures this transform again at the measured TCP, retaining the assumed
world block pose rather than moving the box to a tracking-offset TCP target.
Preview captures it at the simulated descent endpoint instead of reading the
physical arm at home. Transfer preserves the captured grasp orientation, and
the placement TCP is computed from that attachment and desired world block
centre plus release clearance. It may differ slightly from the raw taught place
TCP to account for the captured grip offset.
The world block remains axis aligned at pick/place; this assumes the same tool
orientation at both locations. A changed object orientation needs a new model.

The cycle supplies table/block collision geometry for its planning requests;
there is no hand-eye scene setup, calibration board or camera-model requirement.
Existing camera/other attachments remain checked. The unmodeled physical camera
and bracket require operator supervision, as requested.

## Commands

Keep only one MoveIt hardware stack running. If it is already running, reuse it.
Otherwise:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=192.168.3.10 use_fake_hardware:=false load_gripper:=true
```

In another terminal, build/source and preview (no arm or gripper commands):

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select fr3_pick_place --symlink-install
source install/setup.bash
ros2 run fr3_pick_place pick_place_stepwise.py \
  --config src/fr3_pick_place/config/task.yaml --poses taught_tcp_poses.json
```

Inspect the complete robot, block and table geometry and every path in RViz.
The hand must already be homed. Start empty-handed at a clear approach pose;
the measured initial arm pose becomes the final home target. The cycle opens
the fingers at the current arm position in its first approved stage.

After reviewing the preview, enable physical arm AND gripper commands:

```bash
ros2 run fr3_pick_place pick_place_stepwise.py \
  --config src/fr3_pick_place/config/task.yaml --poses taught_tcp_poses.json --execute
```

Nine approvals: open/approach, lower, grasp, lift, transfer, lower, release,
retreat, home. Each arm segment is planned from fresh settled joints in hardware
mode. Enter executes the displayed segment; `q` exits with no automatic release
or return. Transfer/home are ordinary joint-planned paths and may curve;
lower/lift/retreat use the sampled vertical-path checks. Planning-only mode
advances virtual joint states through the sequence, without moving the robot.

You can instead copy XYZ/quaternion values into `task.yaml` and omit `--poses`.
After changing grasp height, also update `block_center_offset_world`; using
`--poses` recomputes it from the configured tabletop and upright block height.
Do not copy TCP coordinates into a block-centre field.

## Manual camera XYZ input

For each detection, edit **`src/fr3_pick_place/config/manual_camera_target.yaml`**:

```yaml
xyz_m: [0.0369, 0.0315, 0.5401]
```

These are metres in `camera_color_optical_frame`, on the block's top surface,
directly above its center in X/Y. The configured base-axis offset
`point_to_block_center_world_m: [0.0, 0.0, -0.00475]` converts that point to the
center of the upright 9.5 mm high block. Update this offset for a different
height or point definition. A point elsewhere on the top face needs an X/Y
offset too. Perception's quaternion is ignored: TCP orientation is fixed at
`[1, 0, 0, 0]`; the world-axis center-to-TCP offset comes from `task.yaml`.
Camera mode ignores `--poses` entirely, including its saved joint angles.
IK is seeded from current joints and tries up to five collision-checked
solutions, choosing the one with the smallest maximum joint change from the
current posture. There is no 0.75 rad bound in camera mode. This comparison
selects an endpoint; it does not guarantee a singularity-free or short path.
Joint limits, collision checks and per-stage approval remain enabled.
The block collision box remains
upright and axis aligned, so rotate the block to match the taught grasp for
this XYZ-only workflow.

`config/fr3_d435_handeye.yaml` is a copy of the supplied Downloads file. It
explicitly contains **nominal CAD-derived estimates, not measured calibration**.
Check that its mounting assumptions match your assembly and verify the result
before executing. `config/handeye.yaml` is a separate calibration-tool settings
file, not the extrinsic transform used here.

The script computes, once before stage 1:

```text
p_base_surface = T_base_link8(starting joints) * T_link8_color_optical * p_camera
p_base_center  = p_base_surface + point_to_block_center_world_m
p_base_tcp     = p_base_center - block_center_offset_world
```

MoveIt FK supplies `T_base_link8` from fresh settled joints. Keep the arm at the
image-capture posture until startup has printed the converted targets. Moving
the arm after capture but before conversion invalidates the coordinates;
take a new detection. This manual workflow has no image timestamp and cannot
recover the old camera posture. The target is frozen in base coordinates for
the rest of the run; it is not recomputed as the camera moves.

Build/source as above, then first inspect coordinates without planning or motion:

```bash
ros2 run fr3_pick_place pick_place_stepwise.py \
  --config src/fr3_pick_place/config/task.yaml \
  --camera-target src/fr3_pick_place/config/manual_camera_target.yaml --resolve-only
```

Preview only the first three stages (Enter advances each preview):

```bash
ros2 run fr3_pick_place pick_place_stepwise.py \
  --config src/fr3_pick_place/config/task.yaml \
  --camera-target src/fr3_pick_place/config/manual_camera_target.yaml --stop-after 3
```

Add `--execute` to that preview command for physical approach, descent and grasp.
It then stops **holding the block**, with no lift or automatic release. Start
each cycle empty-handed; stage 1 opens the hand. Place remains at the original
configured base-frame position in `task.yaml`; pick XYZ and TCP orientation are
replaced. No taught JSON is read or modified in camera mode.
Converted geometry is validated against the configured table; an inconsistent
height aborts rather than silently snapping the measurement to the table.
The error reports block bottom Z, table top Z and penetration in millimetres;
the observation flange pose is printed before conversion. For a block resting
on this table, its detected top surface should resolve to -0.0105 m in base Z
(-0.020 m tabletop plus 0.0095 m block height). If it does not, first reacquire
the detection at the current stationary arm pose, then verify the CAD mount
transform and actual tabletop height. Do not lower the collision table solely
to make an inconsistent camera point pass.

**No TF publisher or camera driver is required for this manual adapter.** It
uses the optical-frame extrinsics directly. For a complete ROS camera TF tree,
the existing `handeye_calibrate.py solve` workflow generates a
`static_transform_publisher` command after quality gates (see `HANDEYE.md`).
That is a calibration workflow, not an automatically running publisher. With
RealSense publishing its internal camera frames, publish only the flange/hand
to `camera_link` mounting edge; do not give an optical frame a second parent.

## Motion logs

Every invocation writes a unique `data/motion_logs/stepwise_<timestamp>.jsonl`
file relative to the working directory and prints its absolute path. Change the
directory with `--log-dir PATH`. Events are flushed immediately, including the
endpoint measurement before a tolerance failure. These generated files are
ignored by Git. No execution or automatic recovery is enabled by logging.

The log contains the effective config, initial joints, camera observation and
flange pose, resolved target, camera IK candidates/selection, requested pose
goals, commanded joint trajectories, MoveIt execution result codes, and settled
endpoint measurements. Endpoint events include planned and measured TCP XYZ,
quaternions, joint angles, per-joint errors, XYZ error, orientation error,
tolerances and acceptance status. TCP coordinates use MoveIt forward kinematics
and are not independent physical metrology. The joint-state timestamp belongs
to the measured snapshot; this is not continuous tracking telemetry.

Run-end events distinguish completed, stopped and aborted runs. A completed
preview is not hardware validation. Motion failures can have a result event
without an endpoint event if execution or measurement failed. Requested pose
goals and planned FK endpoints are logged separately so planning tolerance is
not confused with execution tracking error. Logs contain no singularity metric
yet. In preview mode no actual execution endpoint is recorded.

## Tracking note — 2026-10-24

Observed: a -5 mm jog produced -6.414 mm Z displacement, 0.701 mm lateral drift,
0.429° orientation change and 0.004095 rad maximum final joint error. The effort
controller had joint goal/path position tolerances disabled and zero integral
gains. This does not establish the cause of the offset.

For this teaching trial, the cycle accepts up to 2 mm final Z tracking error,
2 mm lateral error and 0.5° orientation error relative to the planned endpoint.
Z or lateral errors above 1 mm warn. A later approach measured 1.091 mm lateral
error, -0.405 mm Z error and 0.423° orientation change; the lateral endpoint
allowance was increased from 1 to 2 mm for teaching. This does not correct tracking.
Joint error above 0.01 rad also stops execution.
Path and execution-start checks remain strict. No controller gains were changed
and there is no automatic corrective motion. Inspect payload configuration and
controller tracking before relying on precise pick/place replay. A successful
action or accepted endpoint does not certify physical clearance or grip security.

Offline tests validate configuration, request construction and execution gates;
the complete physical cycle has not been validated by these tests.

The block height was corrected from 95 mm to 9.5 mm after a descent returned
only about 55% of its Cartesian path. The taller model was a likely obstruction;
the corrected geometry still requires a new planning preview. Incomplete paths
now include a read-only next-pose IK/validity diagnostic with collision contacts
when available; the diagnostic candidate is never executed.

A preview subsequently reached lift planning but failed in scene cleanup. The
lift now removes a world block only if that ID exists, and restores only the
objects it changed. Transfer planning also omits a redundant world removal after
attaching the block. Scene cleanup failures preserve the original error message.

A physical trial completed approach, descent, grasp and lift, then stopped after
transfer (stage 5): 2.058 mm lateral error, -0.941 mm Z error, 0.603 degrees
orientation error. Desk also reported approaching a singularity. The cause and
warning posture have not been identified; endpoint tolerances were not increased
again. Release and retreat were not commanded. Do not restart the sequence with
a held object: stage 1 opens the hand. Collision/path checks currently do not
include a dedicated Jacobian-based singularity check. Investigate the transfer
joint configuration and Desk warning before continuing a loaded cycle.

For the taught-pose workflow (without `--camera-target`), approach/transfer
pose-goal IK is seeded to avoid unrestricted redundant-arm
posture selection. Approach uses the taught pick joints (from `--poses` or
`approach_seed_joints` in YAML); transfer uses the current measured joints.
Collision-checked IK supplies additional joint goal constraints. A candidate
changing any reference joint by more than 0.75 rad is rejected. This bound is
a posture-change check, not a dedicated singularity threshold or a constraint
on every joint along the subsequent path.
IK requests now include the same +/-0.75 rad joint constraints and retry up to
five times before failing; every returned solution is checked independently.
For a camera target far from the taught pick, failure can require a newly taught
reference posture suitable for the new approach target. Loading `--poses`
overrides YAML `approach_seed_joints` with the JSON pick joints when present.
Changing camera offsets to bypass this posture failure would change the grasp
location rather than solve the IK reference mismatch.

A no-execution audit found an earlier candidate approach moving joint 1 by
about 105 degrees and joint 3 by about 92 degrees. Its geometric-Jacobian
condition number reached about 47.1. With seeded goals, an isolated approach
and transfer audit measured maximum condition numbers about 12.6 and 12.7;
transfer's largest joint change was about 0.12 rad. These use the same nominal
URDF Jacobian with metres/radians and do not reproduce Desk's warning criterion
or the previous executed trajectory. No hardware motion was sent during this
audit. The full virtual audit also encountered block/table contact at lift;
the isolated transfer audit started at planned above-pick joints. Thus this
audit does not validate a complete loaded cycle. Endpoint tolerances remain
unchanged and collision checks remain enabled; a recurring Desk warning still requires
inspection and must not be bypassed by raising endpoint tolerances.

The preview grasp-transform issue and numerical support contact were subsequently
corrected. Sampled validity now accepts only table/block contact of absolute
depth <= 1e-9 m while carrying; this handles floating-point zero-depth support
touch, not meaningful penetration or robot-link contact. Other invalid states
still stop planning, and contact diagnostics include depth. Placement uses the
captured grasp transform and the configured 2 mm release clearance. An automated
planning-only check against the running MoveIt scene then planned all nine stages
successfully. It sent no arm execution or gripper goals. This supersedes the
earlier incomplete virtual audit; physical tracking and the Desk singularity
warning are still not validated by a planning-only result.
