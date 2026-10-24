# FR3 measured-position pick and place

For the wrist-mounted RealSense D435 calibration workflow, see [HANDEYE.md](HANDEYE.md).

ROS 2 Humble, this workspace's FR3 MoveIt configuration, and a Franka Hand.
The default mode plans six sequential arm motions without sending any arm or
gripper execution commands. Each planning-only stage starts from the previous
plan's endpoint and models the changing finger width and carried block.

## Configure

Fill in `config/task.yaml`. All lengths are metres, in `fr3_link0` (robot base).
`grasp_width` is set to your tested **0.015 m** jaw opening. `block_size` is
separate: provide approximate measured dimensions for collision checking.
`pick` and `place` are the block's **centre**. Measure the table box centre and
dimensions, including its thickness. The block's X/Y axes must align with the
base X/Y axes; the fingers close across its Y dimension. The tool orientation
is fixed to quaternion `(x=1, y=0, z=0, w=0)` with tool Z pointing down.
The grasp places `fr3_hand_tcp` at the block centre: verify finger depth against
your actual block and table. Additional fixtures and obstacles must already
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

## Autonomous Lego pick and place (camera watching continuously)

`lego_auto.py` searches for the brick from several scan poses, then steps down through hover
heights, re-aiming at the brick from the camera stream at each stop, grasps, carries it to
`--place-xy`, releases and checks where it landed. Settings are in `config/lego.yaml`
(`scan_poses`, `servo_heights`, ...). It asks you to type `GO` before any motion.

Arm motion is straight Cartesian lines only (`smooth_motion.py`): collision checked, smooth speed
profile, and the wrist always turns the short way (the jaws are symmetric, so a brick angle and
that angle + 180 degrees are the same grasp). A move that cannot be done as a clean line is refused.
In RViz the red box is the true simulated brick, the green box is where the camera thinks it is.

**Simulation (no robot, no camera).** `sim_world.py` provides a fake wrist camera, a fake hand and
a movable brick; the brick shows in RViz as the `SimBrick` marker, the camera's estimate as the green `BrickEstimate` box, the camera picture in the
`Camera` panel, and the fingers open and close. The brick sits at a random place and angle that only
`sim_world` knows, with decoys (a red wrong-size object, a blue brick) that must be ignored. The simulated camera is deliberately a few mm off its calibration
(`sim_cal_error`), so the result is not perfect. Use a private ROS domain so it cannot reach a real
robot or other people's sessions:

```bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77
ros2 launch fr3_pick_place lego_bringup.launch.py use_fake_hardware:=true sim_world:=true
# second terminal, same two exports:
ros2 run fr3_pick_place lego_auto.py --config src/fr3_pick_place/config/lego.yaml --place-xy 0.45 -0.20 --yes
```

`--yes` skips the GO prompt only when `sim_world` is running. The launch also swaps the fake arm
to a position controller (`config/sim_arm_controller.yaml`), because the stock effort controller
does not move the fake hardware.

**Step mode.** On a real robot `lego_auto.py` asks before every arm move and hand action: it plans
the move without executing it, prints the line length, hand yaw, biggest joint turn and duration,
and waits. Enter = do it, `a` = run the rest automatically, `q` = stop (the arm stays where it is,
the hand is not touched). `--no-step` turns it off; `--step` turns it on in simulation.

**Real robot.** Camera running, then
`ros2 launch fr3_pick_place lego_bringup.launch.py spawn_brick:=false` and the same
`lego_auto.py` command without `--yes`.
