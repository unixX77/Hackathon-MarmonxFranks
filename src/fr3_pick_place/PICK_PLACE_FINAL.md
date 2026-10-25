# Combined single-brick picking and placing

`pick_place_final.py` imports the shared CLI/node lifecycle from `pick.py` and
the placement extension from `place.py`. `PlacePicker` inherits `Picker`;
`PickPlaceWorkflow` invokes the imported `PickWorkflow` before placement.
There is one picking implementation. Restart processes after editing imported
Python modules. Changes to function interfaces may require caller updates.
The standalone `pick.py` remains a pick-only tool.

This version handles one upright, axis-aligned brick and an empty destination
on the configured tabletop. Requested TCP orientation remains `[1, 0, 0, 0]`.
It completes lift, transfer, placement release and empty-hand retreat. It does
not return home automatically, stack bricks, or implement color selection.

## Configure the destination

Edit `config/place_target.yaml`:

```yaml
frame_id: fr3_link0
block_center_xyz_m: [0.3176477189209497, 0.05343183308846983, -0.01525]
release_clearance_m: 0.002
approach_clearance_m: 0.1
```

These are desired final **brick-center** coordinates, in metres in robot base.
They are not TCP coordinates or camera coordinates. The default Z is half the
9.5 mm block height above the -20 mm tabletop. The destination bottom must
match the configured tabletop within 1 mm; elevated supports are not modeled
by this version. Configure table/block dimensions in `task.yaml` first.

Release is nominally 2 mm above that center. The held-object transform captured
after grasp determines the required TCP XYZ. Physical release geometry is
estimated from actual TCP FK and the captured grasp, including orientation.
It is not a new visual measurement. Small grip/orientation errors can tilt the
block model; the planner does not silently flatten it into the table.

## Preview

Use the existing fake-hardware MoveIt launch, then in another terminal:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select fr3_pick_place --symlink-install
source install/setup.bash
ros2 run fr3_pick_place pick_place_final.py \
  --config src/fr3_pick_place/config/task.yaml \
  --camera-target src/fr3_pick_place/config/manual_camera_target.yaml \
  --policy src/fr3_pick_place/config/pick_place_policy.yaml \
  --place-target src/fr3_pick_place/config/place_target.yaml
```

Preview plans and advances virtual joints only. Each segment is published for
RViz inspection and requires Enter. Gripper opening/closing is virtual; no
execution goals are sent, and no real holding/release claim is made.
The destination above is a proposed empty tabletop location, not a physically
validated place. Inspect it in RViz.

## Hardware and service mode

Adding `--execute` enables arm/gripper commands on a real stack, still requiring
approval at every movement and gripper operation. Start empty-handed at the
camera observation posture. For animated mock execution use `--fake-execute`
instead, after restarting the updated fake stack; see `FAKE_EXECUTION.md`.
Do not use `--execute` as a fake-mode switch.

Add `--serve` to start idle and remain alive between requests:

```bash
ros2 service call /pick_place_final/start std_srvs/srv/Trigger '{}'
ros2 topic echo /pick_place_final/status
ros2 topic echo /pick_place_final/result
```

Each request reads the task, camera input and place-target files afresh. Policy
is loaded at process startup. Keep the terminal available for approvals.
Camera XYZ is resolved only once at the observation posture, then kept in base
coordinates through pick, recovery and place. Reacquire observations before a
new request; old camera XYZ must not be reinterpreted at the final retreat pose.
The manual target file has no observation timestamp.

The service returns acceptance immediately; task outcome is published later.
Successful physical completion reports PLACED; preview reports PREVIEW_COMPLETE.
Failures report the holding state and do not terminate service mode. Held or
uncertain objects and intervention-required states inhibit new requests.
Only after removing the brick/inspecting the robot can an operator explicitly
attest empty/healthy using `/pick_place_final/confirm_empty`. This service does
not detect an empty hand, open it or clear robot faults.
It acknowledges the request immediately; wait for EMPTY_CONFIRMED status before
starting another request. For an operator-removed held/uncertain brick, the
worker also removes this tool's stale attachment from the planning scene.

## Placement behavior and recovery

1. Run the inherited pick workflow. No placement starts after a failed pick.
2. Capture the relative block pose at the measured grasp TCP (virtual endpoint
   during preview). On hardware, persist it as an attached collision object.
   A successful gripper action is still not independent proof of a secure grip.
3. The combined picker checks post-grasp lift feasibility during candidate
   selection, then replans and verifies the actual lift with the brick attached.
   A failed lift keeps
   the grasp and requires assessment; there is no automatic home motion.
4. Find distinct placement IK endpoints using the inherited seed generation.
   Evaluate transfer paths, held-object descent and empty-hand retreat. Rank
   accepted combinations using the inherited sampled path checks. Future-release
   lookahead temporarily changes the hardware planning scene while stationary,
   then restores the held-object model; only one task/planning owner is supported.
5. Transfer above the destination and check arrival. Bounded corrections are
   allowed only at that clear height, with improvement required.
6. Rebuild the held-object descent from actual measured joints. A recoverable
   descent failure can trigger a newly checked upward retreat while keeping the
   gripper closed. Retry limits use `descent_retries` from the shared policy.
   Failed candidates are excluded from reselection. Fault/contact/stale feedback
   or failed recovery prohibits further automatic motion.
7. Check release pose, then check again after the approval prompt before opening.
   Hardware release requires action success and fresh finger-width feedback.
   An uncertain result keeps the state conservatively blocked/holding; there is
   no automatic retreat or repeated opening command.
8. After confirmed opening, detach the model and add a world object at the
   estimated release pose. The physical brick may settle under gravity; future
   tasks require perception to refresh this estimate. If the scene update fails,
   the physical holding state is already empty and further motion is blocked.
9. Plan and execute an empty-hand vertical retreat. Failure after release reports
   intervention required and never restarts grasp/release automatically.

Existing camera attachments and scene obstacles are retained. The physical
camera/bracket must be represented in the robot scene for collision checking.
The same demo block ID is reused: this is a single-brick development tool, not
yet an accumulating scene of many sorted bricks.

Logs remain under `data/motion_logs`. They include picked-object attachment,
placement request, candidate checks, state transitions and estimated release.
Native robot monitoring and numerical Jacobian limitations are documented in
`PICK.md`. Hardware payload/controller settings are not changed by this code.

The combined policy uses `relative_jump_factor: 10` rather than the standalone
picker's default 2. A planning-only diagnostic found factor 2 truncating full
paths whose largest joint step was below 0.011 rad. The relative heuristic is
still enabled; the independent 0.1 rad absolute jump check, dense timed-spline
validation, collision checks, joint-limit margins and nominal singularity checks
remain in effect. No incomplete trajectory is executed as a fallback. This is
a planning-heuristic adjustment, not a relaxed endpoint tracking tolerance.

Offline tests and planning previews do not validate physical gripping, release,
tracking or recovery. Validate individual hardware stages under supervision.
