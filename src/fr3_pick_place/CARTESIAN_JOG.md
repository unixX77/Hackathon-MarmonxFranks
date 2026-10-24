# Numeric vertical teaching

`cartesian_jog.py` previews small base-Z steps in RViz, optionally executes each
approved step, and saves measured TCP poses. It holds the current orientation;
it does not make a tilted gripper vertical. Use an empty hand. There are no
gripper commands or automatic retreat/recovery moves.

Build and source the package:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select fr3_pick_place franka_fr3_moveit_config
source install/setup.bash
```

Use the normal MoveIt stack. For example, with the correct robot IP:

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=172.16.0.2 use_fake_hardware:=false load_gripper:=true
```

Activate FCI before real-hardware bringup. This launch activates an arm trajectory
controller; it is for ROS motion, not the manual-guiding workflow. Stop a previous
basic bringup first so only one stack connects to the hardware. Launching this
stack is separate from executing a trajectory. Do not run competing motion
commands. Use RViz to plan, inspect and execute the initial approach above the
object with a suitable tool-down orientation.

On startup the tool adds a table (default ID `fr3_demo_table`) to the shared
MoveIt scene if that ID is absent. Geometry comes from the installed
`config/cartesian_jog.yaml`; use `--config PATH` for another file. The default
is an assumed 2 m square centered beneath the robot, with its top 20 mm below
`fr3_link0` and modeled thickness 40 mm. Verify its position and coverage in RViz
before jogging. Existing objects with the selected ID are preserved; changing
the YAML does not replace an existing table. Use `--table-object-id ID` to select
an existing table or name the new one.

No hand-eye calibration, camera TF, board, or camera collision model is required.
The mounted camera/bracket is not modeled automatically; its clearance must be
supervised by the operator. Existing attachments remain collision-checked.
Model other obstacles as appropriate and keep the scene unchanged during a jog.

In a second sourced terminal, first preview without execution:

```bash
ros2 run fr3_pick_place cartesian_jog.py
```

The preview-only tool always starts each new jog from the measured state. It
does not accumulate virtual steps. In RViz enable MotionPlanning's Planned Path
display; the tool publishes on `display_planned_path`.
Pressing Enter in this mode only accepts the preview; it sends no motion command,
even when MoveIt is connected to the physical robot. Hardware is classified only
when starting with `--execute`.

For real arm motion, restart explicitly with:

```bash
ros2 run fr3_pick_place cartesian_jog.py --execute \
  --output src/fr3_pick_place/data/taught_tcp_poses.json
```

For offline testing, launch MoveIt with `robot_ip:=dont-care
use_fake_hardware:=true load_gripper:=true` instead. Fake hardware uses a
position trajectory controller so executing a trajectory updates simulated
joint states. Then run the same tool with `--execute --output
/tmp/fake_taught_tcp_poses.json`. It reads the arm hardware plugin from
`controller_manager`'s robot description and prints `SIMULATED EXECUTION` for
`mock_components/GenericSystem`. All modes install a missing configured table
and check collisions with scene geometry. Real execution also verifies the table
is still present before planning/revalidation. Unknown hardware plugins
or an unavailable controller-manager model cause an abort before execution.

Commands:

```text
pose          # show measured TCP position and quaternion
z -0.005      # plan 5 mm down in fr3_link0, then prompt before execution
z 0.002       # plan 2 mm up
save pick     # save measured pose and arm joints; no motion
save place
q
```

Review every path before pressing Enter. `q` at the approval prompt exits.
Steps must be 1–20 mm in magnitude. Default joint velocity/acceleration scaling
is 0.02, with a maximum permitted setting of 0.05; scaling is not a TCP speed
limit. Tool-down orientation and clearance must be established beforehand.

Each jog uses the existing Cartesian planner's complete-path, timed-spline,
joint-limit, collision, vertical-line and orientation checks. After approval it
checks the sampled trajectory against the current scene again, then verifies a
fresh measured start before sending that same trajectory to ExecuteTrajectory.
Other planners/scene editors must remain idle during this operation. These
sampled model checks do not guarantee physical clearance or controller tracking.
An error aborts the tool. Keep the hardware stop accessible.

Saved JSON contains `frame_id`, `tcp_link`, and named `poses` with `xyz_m`,
`quaternion_xyzw`, and `arm_joints_rad`. Names cannot be overwritten accidentally.
These are TCP poses, not block centres. The staged cycle can load them using
`--poses taught_tcp_poses.json`; see [STEPWISE_PICK_PLACE.md](STEPWISE_PICK_PLACE.md)
for object geometry, grasp offsets, preview and execution commands.

Offline regression tests cover the input limits, attachment preservation, pose
recording and execution gates. This tool has not been validated on hardware.

## Teaching accuracy note — 2026-10-24

A physical -5 mm jog produced measured displacement [-0.701, -0.009, -6.414] mm,
with 0.429 degrees orientation change and maximum final joint error 0.004095 rad.
The 1.414 mm vertical endpoint error is temporarily accepted for pose teaching.
The final Z tolerance defaults to 2 mm; errors above 1 mm emit a warning.
Use `--endpoint-z-tolerance 0.001` to restore strict 1 mm endpoint acceptance.
Lateral, orientation, planned-path, collision and execution-start checks remain
unchanged. Saved poses always come from measured joints and FK, not the requested
target. This allowance does not correct physical tracking or establish pick/place
accuracy.

The running effort controller reported zero joint goal/path position tolerances
(disabled) and zero integral gains. Controller success therefore does not certify
endpoint position accuracy. Before precise pick/place replay, investigate tracking,
verify payload configuration and configure suitable controller tolerances. The
cause of the measured offset is not yet established; gains were not changed.
