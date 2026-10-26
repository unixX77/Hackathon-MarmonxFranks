# One-brick physical hover — 2026-10-26

The operator executed `scripts/observe_robot_frame.py --execute-hover` and
confirmed slow, successful motion. This was hover only, without grasp, release
or automatic return. The robot remained at hover.

Evidence: `runtime/robot-frame-20261026-092011-439616/hover_execution.json`.
Status COMPLETE, target TCP approximately [0.399037, -0.007684, 0.089559] m
in fr3_link0; reported TCP arrival error 0.0021633 m. Saved trajectory duration
approximately 62.9 seconds. Arrival error measures proximity to the commanded
target, not accuracy relative to the physical brick.

Operator observations after the hover:

- Brick is approximately 20 mm ahead of the fingers along the current tool +X
  axis. This is a measured-direction estimate, not a verified calibration or
  fixed correction across other poses.
- Jaw orientation is incorrect by a smaller angle than 90 degrees. Angle
  magnitude and sign have not been measured.
- Lowest fingertip-to-brick-top gap is approximately 97 mm.

No descent or grasp should be inferred as authorized by a successful hover.
Alignment remains unresolved. `hardware_validated` and `geometry_confirmed`
remain false. No calibration matrix, yaw offset or TCP correction was changed.

The model's stock finger joints move along hand Y, and hand_tcp has identity
rotation relative to the hand. The planner aligns tool X with the detected
brick long axis, making closure along its short axis. A smaller angular error
does not establish a missing 90-degree rotation. A single approximate lateral
offset also does not distinguish perception, hand-eye, frame or physical TCP
errors. Determine angular magnitude/direction and recheck alignment before
planning any correction or descent.

## Updated operator assessment and relative preview

After providing three photos (WhatsApp images at approximately 09:31), the
operator reassessed the yaw error as close to zero. The angled photos do not
support precise angular metrology. Keep yaw unchanged for the next alignment
preview; do not apply an assumed yaw correction.

`scripts/plan_tool_shift.py --tool-x-mm 20` plans a horizontal shift along the
current tool X axis projected onto base XY, preserving current TCP Z and the
full quaternion. It reads fresh stationary feedback and TF without recapturing
the nearby brick. The displacement is limited to 25 mm, and every sampled FK
pose must preserve Z within 1 mm and orientation within 0.5 degrees. Existing
hover collision checks, retiming, TCP segment-average checks and motion guards
are reused. This script has no execution option; the shared helper refuses
execution of an explicit relative target. It changes no calibration or
persistent TCP correction. The full 46-test suite passes, including signed-frame
geometry and shift-bound validation. A live relative plan and physical alignment
correction have not yet been verified.

## Operator-corrected sign and explicit execution option

The operator corrected the desired displacement to **-20 mm along current tool
X** and explicitly requested its execution command. The earlier +20 mm result
was planning only and did not move the robot.

`scripts/plan_tool_shift.py --tool-x-mm -20 --execute` now freshly plans that
relative shift and executes it using `SafeMover`'s monitored/cancellable action
adapter. Execution requires an explicit signed distance, a stationary robot,
unchanged planning guard and matching trajectory start. The existing horizontal
FK checks and conservative timing remain. Arrival position and orientation are
checked afterward. No gripper command, descent or return is included. Running
the command again would request another relative displacement.

Five mock-action tests pass for changed-start refusal, moved-guard refusal,
explicit-distance admission, adapter execution and arrival-error reporting.
The signed geometry test covers negative tool X. No physical relative shift
was executed by the assistant; physical completion is still pending the
operator's report. The earlier planning-only limitation above describes the
initial version and is superseded by this explicit option.

## Failed relative execution and communication reflex

At 09:56 the operator ran the requested -20 mm relative shift. The six-point
trajectory was 3.56475 seconds long, targeting approximately
[0.378799, -0.014155, 0.088875] m. MoveIt and the joint trajectory controller
reported SUCCEEDED, but the independent arrival check failed by 20.0 mm.

A read-only inspection afterward found the measured joints exactly equal to
the trajectory's first point. Fresh TF and independent FK agreed on
[0.397503, -0.007072, 0.088875] m, the original pose. This establishes that the
requested endpoint was not reached, rather than a stale TF-only discrepancy.
See `runtime/tool-shift-20261026-095558-172275/read_only_arrival_diagnostic.json`.

Live-driver logs in `/home/tanay/.ros/log/` show:

- `ros2_control_node_32404_1793002052057.log`: at 09:44:47, libfranka motion
  aborted by reflex with `communication_constraints_violation`.
- The same controller accepted the relative goal at 09:56:03 and reported
  success at 09:56:07, despite the subsequently unchanged measured pose.
- Driver shutdown was recorded at 10:00:50.

The original execution admission checked fresh joint publications but did not
verify physical robot-clock progression or Franka health. That check was
incomplete for this failure. Added `RobotHealth` preflight/monitoring to both
hover diagnostics: require active control (mode MOVE), no current errors, fresh
health publication and an advancing physical `FrankaRobotState.time`. A fault
during a pending action triggers the existing cancellation path. No error
recovery, driver restart or new physical shift was issued by the assistant.

`scripts/check_robot_health.py` performs this check without sending commands.
Relative execution failures now retain measured arrival data, TF age and health
snapshot when available. The 3 mm arrival tolerance is unchanged. The 57-test
suite passes; two additional targeted fake-action tests verify that health
failure prevents a goal and that an in-flight health failure cancels a goal.
These guards are not a diagnosis or repair of Ethernet/realtime communication.

Do not blindly repeat the relative shift. Inspect/recover the Franka fault via
the operator interface, restart the real driver stack as appropriate, then
inspect health and current physical pose before another movement.
