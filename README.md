# Hackathon MarmonxFranks (branch `ameed`)

FR3 + Franka Hand + wrist RealSense D435 Lego pick and place (ROS 2 Humble). This folder is the
main workspace: build, simulate and run everything from the repo root.

| Path | Contents |
|---|---|
| `src/fr3_pick_place/` | Pose-driven pick (`pose_pick.py`), click-to-pose helper (`pixel_to_base.py`), straight-line motion, hand-eye tools, earlier camera-based Lego pipeline |
| `src/franka_ros2`, `src/libfranka`, `src/franka_description` | Franka dependencies (nested repos) |
| `franka-handeye/` | Standalone hand-eye calibration app (nested repo) |
| `data/calibration/` | Hand-eye / ChArUco samples and results. `charuco_result.json` is the one used for picking |
| `data/lego_debug/` | Detection debug images |
| `tanay/` | Separate worktree of the `tanay` branch (not part of this branch, not built here) |
| `build.sh` | Builds the workspace (keeps colcon out of `tanay/`, which has its own `fr3_pick_place`) |

## Build

```bash
./build.sh                         # first time about 3 min, later seconds
source install/setup.bash
```
Plain `colcon build` fails with "duplicate package names" because of `tanay/`: use `./build.sh`.

## Run (from the repo root, every terminal)

```bash
cd ~/hackathon_repo
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77      # same two lines in every terminal
```

Simulation (terminal 1; read the `HIDDEN TRUTH` brick position it prints):
```bash
ros2 launch fr3_pick_place lego_bringup.launch.py use_fake_hardware:=true sim_world:=true spawn_brick:=false
```
Pick (terminal 2, a pose a few mm off the hidden brick):
```bash
ros2 run fr3_pick_place pose_pick.py --config src/fr3_pick_place/config/pose_pick.yaml \
     --pose 0.455 0.096 --yaw 27 --yes --step --inspect
```
Real robot: terminal 1 is `ros2 launch fr3_pick_place lego_bringup.launch.py robot_ip:=192.168.3.10 spawn_brick:=false`,
terminal 2 drops `--yes`. Details, the correction commands and the safety limits are in
`src/fr3_pick_place/README.md`.

Tests: `python3 -I -m pytest -q src/fr3_pick_place/test`
