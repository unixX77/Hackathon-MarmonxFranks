#!/usr/bin/env bash
# REAL hardware runner. No simulation fallback. Every action requires Enter.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/install/setup.bash"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src/lego_sorter${PYTHONPATH:+:$PYTHONPATH}"
export ROS_HOME="$ROOT/runtime/ros_home" ROS_LOG_DIR="$ROOT/runtime/ros_logs"
exec /usr/bin/python3 -s "$ROOT/scripts/run_one_layer_hardware.py" "$@"
