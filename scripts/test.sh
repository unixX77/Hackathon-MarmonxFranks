#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tanay/Documents/Hackathon/project-planner
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export ROS_HOME="$ROOT/runtime/ros_home" ROS_LOG_DIR="$ROOT/runtime/ros_logs"
export PYTHONPATH="$ROOT/src/lego_sorter${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/python3 -s -m unittest discover -s tests -v 2>&1 | tee "$ROOT/runtime/unit-tests.log"
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/install/setup.bash"
set -u
/usr/bin/python3 -s "$ROOT/scripts/ros_smoke.py" | tee "$ROOT/runtime/ros-smoke-latest.log"
