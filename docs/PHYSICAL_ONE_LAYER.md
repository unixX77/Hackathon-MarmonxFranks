# Physical one-layer trial, with Enter before each stage

Use `scripts/run_one_layer_hardware.sh` for the **real robot**. The similarly
named `run_one_layer_mock.sh` remains simulation only. There is no simulation
fallback in the physical runner. No physical execution was performed by the
assistant when developing this runner. Software tests use fake hardware and
services; a complete physical grasp/place cycle is not yet validated.

Scope: at most N operator-verified placements (1–30), one loose source layer,
red to the taught red-layer centre and other/unknown colours to the taught
discard **table grid**. No connected-stud extraction, seating or box drops.
The source can approximately follow the lattice pattern without coinciding
with expected cells. Picking uses measured detection poses inside a rectangle.

## 1. ROS setup and existing hardware stack

In project terminals:

```bash
cd /home/tanay/Documents/Hackathon/project-planner
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

Keep exactly one real driver/MoveIt stack running. If none is running, after
resolving the recorded communication fault:

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=192.168.3.10 use_fake_hardware:=false use_rviz:=true
```

The cyan preview viewer cannot move the robot. In real MoveIt RViz, a dragged
marker is only a proposed goal: plan/review/execute under supervision and wait
for the physical arm to stop before capturing a taught point. These recording
commands send no motion or recovery commands. Do not push an arm with an active
position/effort controller.

## 2. Teach the source rectangle with two opposite corners

Hover the actual TCP above one corner of a generous rectangular **source area**
(not a brick-grid cell), and record:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name source_corner_a
```

Move to hover over the diagonally opposite corner, stop and record:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name source_corner_b
```

Corner order does not matter. The builder computes min/max base X/Y, centre,
width and depth. Corner hover Z is ignored. The rectangle must be at least
40 mm wide/deep, fit the configured workspace and be fully visible from the
observation pose. Leave margin around all source bricks. Keep the entire red
and discard destination grids outside this rectangle so placed bricks cannot
be detected as source again. This separation is an identity check, not an
occupancy/clearance allocation check.

Candidates outside/straddling the source rectangle are rejected before
top-layer selection. It is no longer necessary to teach an exact source-grid
centre. The older `--name source` record is not used by this runner.

## 3. Teach or retain the remaining poses

Existing `place` and `discard` records are retained. Re-record only if changed,
hovering above the centre of each complete destination region:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name place
/usr/bin/python3 -s scripts/record_destination.py --name discard
```

Those commands belong at different physical positions, not back-to-back while
the arm stays still. Both destinations are 6x5 **single-layer** grids, with
separate cells for non-red bricks, using the existing brick dimensions and
two-stud row gap. Their centre is not the first placement cell. The allocator
does not test destination occupancy, source-brick clearance or finger clearance
there, per operator request. The operator must keep each actual placement clear.
Workspace checks and MoveIt collision checking remain enabled.

Grasp geometry defaults to `hardware.yaml:grasp_tcp_height` (12 mm above the
support), inherited from `project-ameed/src/fr3_pose_pick/config/pose_pick.yaml`.
The reference uses the same `fr3_hand_tcp` frame, 32 x 16 x 9.6 mm brick and
-20 mm table Z. A new grasp recording is **not required**. For a brick resting
on the measured table, nominal grasp TCP Z is -8 mm and release TCP Z is -6 mm.
For a detected brick, the inherited planner adds 7.2 mm to the detected brick
centre Z; it does not replace camera Z with the nominal table height.

An existing `grasp` teaching overrides the configured height, and the preflight
prints which source was used. Only record an override if deliberately replacing
the inherited geometry with an established, table-clear grasp pose:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name grasp
```

Optional overrides retain the TCP-table clearance and tool-down checks. Camera
extrinsics and camera height bias are separate from the TCP grasp offset; the
historical Ameed 3.7 mm camera bias is not imported into this trial. Reusing the
reference geometry does not establish that this installation's camera alignment
or complete physical cycle has been validated.

After choosing a camera pose with the entire source rectangle and surrounding
table visible, stop and record:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name observation
```

The full measured observation quaternion is retained and used by the physical
runner, not reduced to tool-down yaw. Teaching captures are stored, with history,
in `runtime/destination-teaching/destinations.json`. Active hardware configuration
is not edited. Release height for this same-table, same-size-brick trial is the
effective grasp TCP height above the measured table plus the configured 2 mm release
margin. The configuration-check command prints these values for review.

## 4. Build/review the scoped trial configuration without motion

```bash
bash scripts/run_one_layer_hardware.sh --bricks 3 --pick-tool-x-mm -20
```

This command opens no ROS/camera/robot connection. It requires the five records
for the two source corners, place, discard and observation and saves the effective `config.yaml` under a fresh
`runtime/physical-one-layer-*` directory. It prints the source rectangle, effective
grasp height, its configuration/teaching source and release margin. Missing records fail before any hardware access.

`--pick-tool-x-mm -20` explicitly applies the operator's requested -20 mm offset
to pick poses in the tool-X heading selected nearest the current hand yaw. It
is not a global calibration edit and does not shift taught destination or
observation poses. Default is zero; magnitude is limited to 25 mm. Inspect physical
centering at each approach; q stops if the compensation is wrong. It is not a
verified correction across all camera/robot poses.

## 5. Run the physical trial

Close other camera viewers and stop any other task scripts commanding the arm.
Leave the existing real driver/MoveIt stack running. Then:

```bash
bash scripts/run_one_layer_hardware.sh --bricks 3 --pick-tool-x-mm -20 --execute
```

Replace 3 with N (1–30). Hardware mode is always staged; there is no `a`/automatic
continuation. Confirm the physical hand is empty and destination regions clear
at startup. Press Enter only for the displayed stage. Prompts cover observation
motion, capture, opening, approach, descent, grasp, lift, transfer, lower, release,
retreat and physical placement verification. At LOWER, verify real centering and
jaw alignment; at LIFT, verify a secure grasp. Verification Enter confirms the
brick was actually placed and the hand is empty. q or Ctrl+C stops without an
automatic release/return; held and uncertain state is preserved in that run's
files. A new invocation is a new trial, not blind resumption of a failed grasp.

The arm remains at its final hover. The runner stops after N verified placements
or independently verified source empty, whichever comes first. Reaching N is
reported as `TRIAL_COMPLETE`; it does not assert that the source is empty.

## Implemented changes and limits

- No 4 mm matching to an expected source-grid cell.
- No generated source-lattice collision boxes. Only this project's legacy
  `source/…` boxes are removed; unrelated scene objects remain.
- No target-source finger collision exemptions or grid-ID bookkeeping in this
  mode. A successfully grasped brick is attached from its measured/corrected pose.
- Source boundary uses capture-time TF/depth before top-layer filtering;
  no-detection alone is not treated as an empty source.
- Destination allocation occupancy/clearance check disabled only for this
  single-layer mode; separate reservation cells and receipts remain.
- Table, robot self-collision, held/placed bricks, existing fixtures, source-side
  finger selection checks, dimensional/tilt/workspace checks and arrival checks
  remain. Camera/mount geometry remains omitted as previously requested.
- Active Franka control, no current errors, fresh feedback and advancing physical
  robot time are checked before/during actions. No automatic fault recovery.
- Joint-speed scale is at most 2%; FK retiming caps sampled TCP segment averages
  at 10 mm/s. This is not a continuous hardware speed limiter.
- Full-task `hardware_validated` and `geometry_confirmed` flags remain false.
  Only the explicit terminal runner can admit this bounded, taught, one-layer,
  operator-supervised profile. Ordinary hardware launch still refuses it. This
  does not establish production/ten-layer commissioning or calibration accuracy.

Software validation: 87 tests pass, including region filtering, no fixed-source
matching/boxes, disabled destination allocation checks, stage refusal, scoped
admission, FK retiming and full observation quaternion. A complete runner test
with test doubles stops at three verified placements while leaving two source
bricks untouched. No physical grasp/place was executed during these tests.

The explicit pick correction is applied once to the observation before checking
source-neighbour finger clearance; it is not applied again during pick execution.
A workspace-wide lock prevents two copies of this physical runner from running
simultaneously. Keep other arm-commanding tools idle during a trial.
