#!/usr/bin/env bash
# Build the workspace from the repo root. --base-paths keeps colcon out of tanay/ (a separate
# worktree of the tanay branch that has its own copy of fr3_pick_place).
set -e
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash
colcon build --base-paths src franka-handeye --symlink-install \
  --cmake-args -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release "$@"
