# Animated fake-hardware execution in RViz

There are three distinct modes:

- No execution flag: translucent planned-path previews; opaque robot stays put.
- `--fake-execute`: executes trajectories on verified GenericSystem mock hardware,
  animates finger joints, and updates simulated brick attachment/release.
- `--execute`: real-hardware execution requiring native Franka feedback/actions.

The two execution flags are mutually exclusive. Fake mode refuses arm descriptions
or active hardware components that are not `mock_components/GenericSystem`.
It never publishes fabricated Franka health messages or calls real gripper actions.
The default, real-hardware monitoring path is retained.

## Build and restart the fake stack

Stop your previous fake launch with Ctrl-C so it reloads the updated gripper
publisher. Build in the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select franka_gripper fr3_pick_place --symlink-install
source install/setup.bash
```

Terminal 1:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=dont-care use_fake_hardware:=true load_gripper:=true
```

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run fr3_pick_place pick_place_final.py \
  --config src/fr3_pick_place/config/task.yaml \
  --camera-target src/fr3_pick_place/config/manual_camera_target.yaml \
  --policy src/fr3_pick_place/config/pick_place_policy.yaml \
  --place-target src/fr3_pick_place/config/place_target.yaml \
  --fake-execute
```

Enter approves each movement, including virtual gripper opening/closing. The
opaque robot follows controller feedback; the translucent robot still represents
the planned path. For clarity, leave Planned Path > Loop Animation disabled or
hide the planned-path robot while watching execution. Keep the scene robot and
scene geometry visible. The brick is a collision box, not a textured LEGO model.

The table and single simulated brick are initialized at the supplied coordinates.
During transport the brick is attached to the TCP; after opening, it becomes a
world object at the estimated release pose. There is no gravity settling, so the
box can remain at the configured release clearance above the tabletop.

For pick-only execution use the same flag with `pick.py`. It stops after grasping;
full brick transport visualization is supplied by the combined workflow.
`--serve` also works in fake mode; use `/pick_place_final/start` or `/pick/start`
depending on the executable. Result messages identify `backend: fake`.

## What this validates

This mode tests controller/action wiring, measured-state sequencing, gripper
state transitions, motion checks, logs and placement scene updates. The gripper
publisher accepts `[width_m, speed_m_s]` on its fake-only command topic and
animates two equal finger positions. The backend waits for aggregate joint-state
feedback confirming the requested width. Restarting an old fake stack is needed
to expose this command subscription.

Arm identity is checked at activation and before trajectory/gripper commands.
Monitoring uses fresh controller-state error feedback and joint-state FK, not
native Franka collision/contact flags. Mock hardware usually tracks ideally;
this mode does not reproduce payload dynamics, contact forces, slippage, camera
error or Desk's protective-stop behavior. It is not physical grasp validation.

The input file remains manual camera-frame data. For repeatable tests start the
fake stack at its default observation posture. If the simulated arm has moved,
do not reuse the same camera XYZ as though it came from the new camera pose;
restart the fake stack or supply a detection corresponding to that pose.
