# fr3_visual_pick: camera-driven pick and place

Detects the brick in the wrist RealSense image, converts it with the hand-eye calibration and
picks and places it. Hand-eye calibration tools and their notes (`HANDEYE.md`) live here too.
The pose-driven pipeline is the sibling package `fr3_pose_pick`.

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
ros2 launch fr3_visual_pick lego_bringup.launch.py use_fake_hardware:=true sim_world:=true
# second terminal, same two exports:
ros2 run fr3_visual_pick lego_auto.py --config src/fr3_visual_pick/config/lego.yaml --place-xy 0.45 -0.20 --yes
```

`--yes` skips the GO prompt only when `sim_world` is running. The launch also swaps the fake arm
to a position controller (`config/sim_arm_controller.yaml`), because the stock effort controller
does not move the fake hardware.

**Step mode.** On a real robot `lego_auto.py` asks before every arm move and hand action: it plans
the move without executing it, prints the line length, hand yaw, biggest joint turn and duration,
and waits. Enter = do it, `a` = run the rest automatically, `q` = stop (the arm stays where it is,
the hand is not touched). `--no-step` turns it off; `--step` turns it on in simulation.

**Real robot.** Camera running, then
`ros2 launch fr3_visual_pick lego_bringup.launch.py spawn_brick:=false` and the same
`lego_auto.py` command without `--yes`.
