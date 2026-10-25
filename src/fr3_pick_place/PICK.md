# Single-brick picker

`pick.py` is separate from `pick_place_stepwise.py`. It picks one upright,
axis-aligned brick using the same manual camera input and fixed TCP quaternion
`[1, 0, 0, 0]`. It ends after a successful gripper action, holding the brick
without lifting or placing. A successful action is not independent visual proof
that a brick is held. The existing nine-stage script is retained.

## Commands

Run from the workspace root with one MoveIt stack already running:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select fr3_pick_place --symlink-install
source install/setup.bash
ros2 run fr3_pick_place pick.py \
  --config src/fr3_pick_place/config/task.yaml \
  --camera-target src/fr3_pick_place/config/manual_camera_target.yaml \
  --policy src/fr3_pick_place/config/pick_policy.yaml
```

This previews the request. Enter accepts each displayed planning-only segment;
no hardware arm/gripper goals are sent. Virtual joints advance through the
preview; there are no measured execution endpoint errors in preview.

For physical execution, add `--execute`. Every approach, correction, descent,
recovery retreat and gripper action still has an explicit terminal approval.
Start empty-handed. `q` at a prompt stops the request with no automatic release
or motion. Ctrl-C during execution requests cancellation and checks for a
terminal result where ROS is still available. External process termination,
computer failure or unavailable communications cannot guarantee cancellation.

Update camera XYZ at the stationary image-capture posture before each request.
The script resolves camera input once into the base frame and freezes it for
all retries/corrections of that request. It never reinterprets the same camera
XYZ at a new arm posture during recovery. The manual file has no capture
timestamp and cannot validate observation freshness: reacquisition at the
current posture is an operator responsibility. See `STEPWISE_PICK_PLACE.md`
for camera conventions, top-surface conversion and CAD calibration limitations.

## Behavior

1. Generate up to `ik_attempts` deterministic, varied joint seeds inside robot
   limits; solve collision-checked IK at the pre-grasp pose. Reject duplicate
   endpoints within 0.05 rad maximum per-joint distance. Seeds are computational
   starting guesses; they are never executed.
2. Evaluate up to `max_candidates` distinct endpoints, nearest first. Plan an
   approach to each endpoint with MoveIt, audit its timed spline, and check a
   full Cartesian descent from its virtual endpoint before selecting it.
3. Reject sampled paths with collisions, excessive accumulated joint travel,
   insufficient joint-limit margin or insufficient nominal Jacobian margin.
   Rank remaining combinations by total joint travel, extra wrist-joint travel
   weighting and inverse minimum singular value. This version does not optimize
   collision distance; collision validity is a rejection test.
4. Open the hand, approach, settle and check actual pose against the requested
   pose and the planned endpoint. Allow up to `corrections` new checked approach
   plans only within the correction envelope. Stop corrections if the combined
   requested-position/orientation error fails to improve. Correction path
   planning also performs descent lookahead. Joint endpoint error >0.01 rad
   prohibits correction. Target tolerances are not relaxed.
5. Rebuild descent from fresh measured joints (virtual joints in preview).
   Verify X/Y and fixed downward orientation first. Retain existing 2 mm
   Cartesian waypoints, complete-path requirement, timed-spline checks and
   vertical corridor checks. A feasibility path is not reused for execution.
6. During execution, monitor joint feedback freshness, Franka state/mode,
   contact/collision flags, measured-to-native-desired joint error, joint limits,
   nominal singularity margin, and (for vertical segments) TCP corridor/height.
   This is a Python observational watchdog, not a real-time safety controller.
   No full collision service is called continuously during motion; the static
   scene is checked before execution and unexpected contact flags inhibit
   automatic recovery. Stop latency must be considered when validating margins.
7. If a normal descent monitor trips, request cancellation, wait for terminal
   cancellation/success and fresh settled feedback, and check healthy robot
   state. Then try a newly planned, checked vertical retreat at measured X/Y to
   pre-grasp height. It is not a reversed trajectory or a move home. A hardware
   error, contact flag, stale feedback, unconfirmed stop or uncertain execution
   blocks automatic recovery. A failed retreat also blocks the picker.
8. After a successful retreat, reject the failed approach endpoint and permit
   at most `descent_retries` new approach/descent attempts. Even when retries
   are exhausted, a verified retreat is attempted for recoverable descent
   failures before reporting failure. No recovery movement is attempted for
   mere descent-planning failure while stationary above the brick.
9. Verify the final grasp pose before closing. Recheck after the gripper approval
   prompt. An uncertain gripper result blocks automatic restart/opening.

## Limits and validation

`task.yaml` retains block/table geometry, opening/grasp widths, speed scaling,
TCP-to-block offset and endpoint tolerances. `pick_policy.yaml` contains retry,
path-quality, correction-envelope and monitor parameters. `--poses` is not used.

The Jacobian is computed from MoveIt's loaded URDF chain including fixed tool
offsets. Its translational rows are divided by `jacobian_length_m` before SVD.
`min_sigma` is consequently a normalized nominal-model threshold, not Franka
Desk's singularity criterion. The parsed chain FK is compared with MoveIt at
each audited path start. Sampled kinematic checks are not continuous geometric
or dynamic guarantees; thresholds require validation on this robot. They may
reject the current observation posture or all approach candidates. A reported
rejection must be investigated rather than bypassed by disabling checks.

Motion controller settings and payload parameters are not changed. This version
does not add velocity servoing, live visual correction, neighboring-brick models,
placement, grasp-force verification or automatic robot fault reset. Narrow jaw
openings require measured calibration/tracking uncertainty to fit the available
clearance; configured numerical tolerances alone do not establish that fit.

Hardware execution/recovery requires fresh
`/franka_robot_state_broadcaster/robot_state` messages. Override its name with
`--robot-state-topic` if your stack uses a namespace. Missing state inhibits
real-hardware motion. For explicit mock-only execution use `--fake-execute`
as documented in `FAKE_EXECUTION.md`; the mock backend verifies the hardware
identity and uses controller-state feedback instead of native Franka state.

## Persistent development service

Add `--serve` to keep the node running between requests. It starts idle; no pick
begins until explicitly requested. Keep its terminal available for approvals.
The Trigger interface reads the target/config files afresh per request. It is
a development interface, not yet a typed perception/action integration.

```bash
ros2 service call /pick/start std_srvs/srv/Trigger '{}'
ros2 topic echo /pick/status
ros2 topic echo /pick/result
```

`/pick/start` immediately returns acceptance or BUSY/BLOCKED. A worker performs
the request while the ROS executor continues processing feedback. Statuses
include FINDING_APPROACH, CHECKING_CANDIDATE, APPROACHING,
CORRECTING_APPROACH, PLANNING_DESCENT, DESCENDING, STOPPING, RECOVERING,
GRASPING, HOLDING_OBJECT, PICK_FAILED and INTERVENTION_REQUIRED. Request failures
are published and do not terminate the serving process. A one-shot invocation
instead returns a nonzero exit code for failure.

A held/uncertain brick or an intervention-required result inhibits subsequent
requests. After **physically removing the brick and inspecting the robot**, an
operator can explicitly attest the hand is empty and enable a new task:

```bash
ros2 service call /pick/confirm_empty std_srvs/srv/Trigger '{}'
```

This is an operator assertion, not an empty-hand detector. It checks fresh robot
state but does not open the hand or reset Franka errors. Do not call it to bypass
an unresolved failure. Return to a suitable observation pose and acquire new
camera coordinates before requesting another pick.
The response acknowledges the request; wait for EMPTY_CONFIRMED status before
submitting a new task. Context cleanup runs in a worker so ROS feedback/services
continue to be processed.

## Logs and tests

JSONL logs are printed at startup under `data/motion_logs` (`--log-dir` overrides
the directory). They include seed/candidate rejection reasons, path metrics,
status transitions, requested and planned/actual pose errors, monitored samples,
execution results and request outcomes. They are readable offline after failures.

Offline tests cover policy retries/recovery ordering, refusal to grasp after
failed verification, controller-fault handling, cancellation confirmation,
finite-difference Jacobian checks, and independent preview/monitor behavior.
They do not constitute hardware validation. Review a full preview first, then
validate approach/descent/recovery under supervision with conservative settings.
