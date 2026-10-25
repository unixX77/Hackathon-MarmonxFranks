# Hover preview investigation — 2026-10-25

No real arm or gripper commands were sent during this investigation.

## Wall-clock display player (21:38)

The operator confirmed manual scrubbing from waypoint 0 to 23 changes the pose
in the isolated Trajectory display, but automatic playback still appears almost
instantaneous. This narrows the unresolved fault to automatic playback rather
than path import or robot geometry. It does not identify the precise cause in
the live RViz player.

`scripts/play_hover_preview.py` provides an independent visualization route:
load `src/lego_sorter/config/hover_state_preview.rviz` in the existing RViz,
then run the script with the saved hover display and `--interactive`. Enter
starts one playback, with elapsed time and progress printed every five seconds;
q exits. It publishes only `moveit_msgs/DisplayRobotState` to
`/lego_sorter/hover_preview_state`. It creates no action or service clients and
does not publish controller, joint-state or TF commands. It preserves the saved
timestamps and samples cubic Hermite interpolation when velocities are supplied
(otherwise linear interpolation) for a smooth ghost animation. This display
sampler does not certify controller interpolation or physical speed limits.

Verification: eight preview tests pass, including interpolation endpoints,
elapsed-time sampling, cubic velocity use, input immutability and nonfinite-time
rejection. A separate live ROS reader received 74 changing poses over 2.4833
seconds for the original path's first 2.46149 seconds. The final frame was
acknowledged. Evidence:
`runtime/preview-import-probe/wall_player_transport_test.json` and
`runtime/hover-state-playback-20261025-213849-045000/wall_clock_playback.json`.
The full 45.6678-second visual playback in this new viewer still requires the
operator's observation. Main hardware trajectory execution is unchanged.

The operator reports a tiny, fast loop despite 1x timing, wall-time playback and
a slider maximum of 23. The visible playback fault remains unresolved. DDS
acknowledgement confirms transport, not rendering or playback duration.

## Recording inspection (21:18)

Reviewed the 75.714-second recording and extracted frames in
`runtime/video-preview-audit/`. The visible Trajectory properties show 1x,
Use Sim Time unchecked, Robot Alpha 0.5 and status OK. Loop Animation is
unchecked in this recording. The scene contains overlapping orange/white
robot drawings and a planning interaction ring. The trajectory slider is not
visible in the recording. There is no clearly visible full hover animation.

Extended the C++ probe to call the installed `PlanningLinkUpdater` directly:
its returned link transforms match freshly computed FK for all 24 points.
The hypothesis of stale transform caches was tested and ruled out; see
`runtime/preview-import-probe/render_cache_report.txt`. No renderer patch is
justified by this evidence.

Added `src/lego_sorter/config/hover_preview.rviz` for an isolated display test.
Open this configuration in the existing RViz instance (retaining its robot
description parameters), then replay the saved display message. It has one
opaque purple trajectory, looping at 1x with wall time, and no live robot,
planning interaction markers, scene overlays or motion controls. It does not
replace the main MoveIt config or change robot execution. Visible playback
with this isolated configuration remains to be checked by the operator.

## Independent checks

- Compared `hackathon_ros2_ws/src/fr3_pick_place/scripts/pick_place_stepwise.py`
  and `vision_pick_place.py`: they publish `DisplayTrajectory` on
  `/display_planned_path`. Their RViz configuration uses fixed 0.05-second
  waypoint playback. This would nominally show our 24 points in 1.2 seconds,
  but the current operator reports selecting 1x, so that is not a proven cause.
- Compared `project-ameed/src/fr3_pose_pick` and `fr3_visual_pick`: their smooth
  motion implementation executes trajectories, including against fake hardware
  in simulation. That execution route must not be used on the live system to
  troubleshoot display playback.
- The live RViz command loads this workspace's installed RViz configuration.
  Loaded MoveIt libraries are system version 2.5.10, not another workspace's
  plugins. Two RViz subscriptions exist on the hover topic; this does not prove
  that two displays are enabled.
- Saved `robot-frame-20261025-210440-197507/hover_display.json` survives CDR
  serialization unchanged and is acknowledged by RViz's reliable readers.
- A standalone offline C++ probe imports the saved CDR with the installed
  MoveIt RobotTrajectory implementation, using the project URDF/SRDF. It imports
  24 points over 45.667777144 seconds. FK TCP endpoints are approximately
  [0.305060, -0.059300, 0.360709] and [0.326759, -0.045657, 0.089472] metres.
  This is approximately 271 mm of downward TCP travel.

## Corrected issue

The Cartesian service response's `start_state` contained the final arm joint
values, while its trajectory correctly started at the measured pose. Preview
construction and saved-preview replay now align start-state arm values to the
first trajectory point, preserving fingers and leaving all trajectory points,
velocities and timestamps unchanged. A regression test verifies that behavior.
The offline MoveIt import still produced the full path with the old metadata,
so this correction is not established as the cause of the fast, tiny loop.

The original source/installed RViz mismatch was corrected earlier. The source
and installed configs enable one standalone Trajectory display on
`/lego_sorter/hover_preview` at 1x with wall time. The live GUI may differ after
operator edits. Capture the expanded live display properties, slider and
viewport before making more speculative configuration changes.

## Evidence

- `runtime/preview-import-probe/import_report.txt`: installed MoveIt import/FK
- `runtime/preview-import-probe/corrected_replay_report.json`: corrected metadata
  replay, equal received payload and DDS acknowledgements
- `runtime/preview-import-probe/probe.cpp`, `CMakeLists.txt`, `display.cdr`,
  `fr3.urdf`, `fr3.srdf`: offline reproduction (no action clients or ROS node)
- `scripts/audit_hover_preview.py`: display-only replay; `--interactive` keeps
  the publisher available, Enter republishes, q exits normally

Five preview regression tests pass. The complete suite previously passed 40
tests; the added start-state regression raises the suite count to 41 and was
run with the other preview tests. No hardware execution validation is implied.
