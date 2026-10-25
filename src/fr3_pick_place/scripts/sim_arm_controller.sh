#!/bin/bash
# SIMULATION ONLY (use_fake_hardware:=true): replace the effort-based arm controller by a
# position-based one so the fake arm follows MoveIt trajectories. Refuses to touch a real robot.
#   sim_arm_controller.sh CONFIG.yaml          normal use (the launch file runs this)
#   sim_arm_controller.sh CONFIG.yaml --now    repair: do not wait for the stock controller first
# Safe to run again at any time: it works out what is loaded and finishes the job. A lost
# service reply used to leave the arm with NO controller (the unload worked, the reply timed out,
# the retry was told "no such controller" and the script quit); this repairs that state.
CONFIG="$1"

# ros2 control colours its output (each line starts with an escape code), so strip that first.
loaded() { ros2 control list_controllers 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -q "^fr3_arm_controller"; }

if [ "$2" != "--now" ]; then
  for _ in $(seq 1 60); do
    ros2 control list_controllers 2>/dev/null | grep -q "fr3_arm_controller.*active" && break
    sleep 1
  done
fi
# Safety: only swap when the hardware really is the mock plugin; never on a real robot.
if ! ros2 control list_hardware_components 2>/dev/null | grep -q "mock_components/GenericSystem"; then
  echo "Hardware is not the mock plugin: NOT touching the arm controller."; exit 1
fi
if loaded; then
  ros2 control switch_controllers --deactivate fr3_arm_controller || true
  for attempt in 1 2 3; do
    ros2 control unload_controller fr3_arm_controller || true   # the reply can be lost: check the state instead
    loaded || break
    echo "fr3_arm_controller still loaded after unload attempt $attempt; retrying"; sleep 2
  done
  if loaded; then echo "Could not unload fr3_arm_controller; giving up."; exit 1; fi
fi
if ! ros2 run controller_manager spawner fr3_arm_controller -c /controller_manager -p "$CONFIG"; then
  echo "Could not load the position controller for the simulated arm."; exit 1
fi
echo "Simulated arm now follows position commands."
