# FR3 measured-position pick and place

For opaque-arm/gripper animation on mock hardware, use the explicit
`--fake-execute` mode described in [FAKE_EXECUTION.md](FAKE_EXECUTION.md).

For the combined imported pick/lift/place/retreat workflow, see
[PICK_PLACE_FINAL.md](PICK_PLACE_FINAL.md). Its destination is configured as a
brick-center pose in `config/place_target.yaml`.

For single-brick camera-driven picking with candidate lookahead, bounded
corrections, checked descent recovery and persistent status services, use
[`pick.py` as documented in PICK.md](PICK.md). Preview first; it stops after
grasping and does not lift/place.

For the current taught-pose Cartesian cycle, use
[STEPWISE_PICK_PLACE.md](STEPWISE_PICK_PLACE.md). It documents the configured
31.5 × 15.5 × 9.5 mm block, taught orientation, grasp settings and nine approvals.
The basic `pick_place.py` workflow below does not provide per-stage approvals.

For the wrist-mounted RealSense D435 calibration workflow, see [HANDEYE.md](HANDEYE.md).
For the separate one-brick vision/refinement workflow and offline launch demo,
see [VISION_PICK_PLACE.md](VISION_PICK_PLACE.md).

ROS 2 Humble, this workspace's FR3 MoveIt configuration, and a Franka Hand.
The default mode plans six sequential arm motions without sending any arm or
gripper execution commands. Each planning-only stage starts from the previous
plan's endpoint and models the changing finger width and carried block.

## Configure

Review `config/task.yaml`. All lengths are metres, in `fr3_link0` (robot base).
`grasp_width` is set to your tested **0.015 m** jaw opening. `block_size` is
separate: provide approximate measured dimensions for collision checking.
Without `block_center_offset_world`, `pick` and `place` are the block's centre.
The current configuration instead uses taught **TCP targets** with an explicit
offset to the block centre. Measure the table box centre and
dimensions, including its thickness. The block's X/Y axes must align with the
base X/Y axes. The current `tcp_orientation_xyzw` setting preserves the taught
orientation; when omitted the tool defaults to `(x=1, y=0, z=0, w=0)`.
Verify the grasp offset and finger depth against your actual block and table.
Additional fixtures and obstacles must already
be represented in the MoveIt planning scene.

The program adds the table and block to each request's planning-scene diff.
Only the fingers may contact the block. The block is attached to the TCP in
the carrying phases and placed at the destination for retreat planning.
It does not permanently add these objects to the shared planning scene.

## Build and preview

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select fr3_pick_place --symlink-install
source install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=unused use_fake_hardware:=true
```

In a second terminal with the same ROS environment:

```bash
ros2 run fr3_pick_place pick_place.py --config src/fr3_pick_place/config/task.yaml
```

MoveIt's planned trajectories are published for RViz. Review each phase.
These are collision-checked pose-goal plans, **not guaranteed straight-line
Cartesian descents**. Inspect the complete paths and clearance. Planning with
fake hardware does not validate real calibration, grip, timing, or dynamics.

## Hardware

Stop the fake-hardware launch. Use the controller's supported FCI setup, with
the workspace's required real-time host configuration. Keep the stop device
accessible and the workspace clear for the first slow trial. The hand should
already be homed; this program does not run homing automatically.

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=YOUR_ROBOT_IP use_fake_hardware:=false
```

Run planning-only against the live state first. To command a physical cycle:

```bash
ros2 run fr3_pick_place pick_place.py --config src/fr3_pick_place/config/task.yaml --execute
```

Execution replans each stage from the live state, checks arm and grasp results,
and stops the sequence on failure. It does not automatically open the hand on
failure, since a block may be held. Inspect the hardware before retrying; there
is no automatic recovery. Action cancellation on timeout/Ctrl-C is best effort
and is not a replacement for the hardware stop. Fake gripper mode supplies
joint states only, so `--execute` requires the real hand action servers.

Default namespaces and link names match this workspace. ROS remapping can
select another node namespace; custom robot/link prefixes require code changes.
