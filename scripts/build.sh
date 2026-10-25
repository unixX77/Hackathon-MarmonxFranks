#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tanay/Documents/Hackathon/project-planner
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export ROS_HOME="$ROOT/runtime/ros_home" ROS_LOG_DIR="$ROOT/runtime/ros_logs"
# Do not build the copied Franka packages over the original workspace.
if [[ "${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:${COLCON_PREFIX_PATH:-}" == *"/hackathon_ros2_ws/"* ]]; then
  echo 'The old hackathon_ros2_ws is sourced. Build from a clean terminal without that overlay.' >&2
  exit 1
fi
# ROS setup scripts may inspect unset variables.
set +u
source /opt/ros/humble/setup.bash
set -u
# BUILD_TESTS controls libfranka; BUILD_TESTING controls ament packages.
# Their optional development tests require extra dependencies such as Google Mock.
# The planner's own checks are run separately by scripts/test.sh.
colcon --log-base "$ROOT/log" build --base-paths "$ROOT/src" --build-base "$ROOT/build" --install-base "$ROOT/install" --cmake-args -DBUILD_TESTS=OFF -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
