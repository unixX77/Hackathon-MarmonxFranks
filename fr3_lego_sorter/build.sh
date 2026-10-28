#!/usr/bin/env bash
# Build the ROS 2 package. --symlink-install: the installed scripts are links to src/, so a change to a script or the config
# needs no rebuild, and the scripts find vision/ and the config through them.
set -e
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash
FRANKA_WS=${FRANKA_WS:-$HOME/Documents/Hackathon/hackathon_ros2_ws}
[ -f "$FRANKA_WS/install/setup.bash" ] && source "$FRANKA_WS/install/setup.bash"
colcon build --base-paths src --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release "$@"
