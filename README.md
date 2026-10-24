# Hackathon MarmonxFranks (branch `ameed`)

FR3 + Franka Hand + wrist RealSense D435 Lego pick and place (ROS 2 Humble).

| Path | Contents |
|---|---|
| `src/fr3_pick_place/` | Pick-and-place, autonomous Lego pipeline, hand-eye tools |
| `src/franka_ros2`, `src/libfranka`, `src/franka_description` | Franka dependencies (nested repos) |
| `franka-handeye/` | Standalone hand-eye calibration app (nested repo) |
| `data/calibration/` | Hand-eye / ChArUco samples and results. Use `charuco_result.json` for picking |
| `data/lego_debug/` | Detection debug images |

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```
