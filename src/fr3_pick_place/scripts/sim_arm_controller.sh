#!/bin/bash
# SIMULATION ONLY (use_fake_hardware:=true): replace the effort-based arm controller by a
# position-based one so the fake arm follows MoveIt trajectories. Refuses to touch a real robot.
set -e
CONFIG="$1"
for _ in $(seq 1 60); do
  ros2 control list_controllers 2>/dev/null | grep -q "fr3_arm_controller.*active" && break
  sleep 1
done
# Safety: only swap when the hardware really is the mock plugin; never on a real robot.
if ! ros2 control list_hardware_components 2>/dev/null | grep -q "mock_components/GenericSystem"; then
  echo "Hardware is not the mock plugin: NOT touching the arm controller."; exit 1
fi
ros2 control switch_controllers --deactivate fr3_arm_controller
ros2 control unload_controller fr3_arm_controller
ros2 run controller_manager spawner fr3_arm_controller -c /controller_manager -p "$CONFIG"
echo "Simulated arm now follows position commands."
