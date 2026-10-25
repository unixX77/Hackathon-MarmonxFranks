# Validation report — 2026-10-25

## Passed

- ROS 2 Humble colcon build of `lego_sort_interfaces` and `lego_sorter`. Build/install/log directories are inside project-planner. Only diagnostics were deliberate byte-compilation-disabled warnings.
- 28 unit/integration tests: transforms, schema rejection, calibration frames, dimensional consistency, alternating layer geometry, support/clearance/capacity, all colour policies, atomic reservation recovery, process ownership, stale observations, capture ownership, pick/holding/place transitions, grasp/release uncertainty, failed lift, cancellation, pause/stop/resume, restart, placement verification and independent empty-depth verification.
- A complete 60-brick offline mock mission from a two-layer six-by-five source, including 24 controlled discard placements and alternating destination layers.
- Three actual installed ROS nodes: complete 60-brick mission, with pause, stop-from-paused and resume exercised through ROS services. All 60 destination transactions committed; hand ended empty.
- A generated 61-observation replay executed through the actual three ROS nodes: 60 placements plus independent simulated empty-world observation, with pause/stop/resume exercised.
- Installed launch file resolved via `ros2 launch lego_sorter sorter.launch.py --show-args`.
- Package inventory contains exactly two unique integration package names. All integration Python source parses.
- Final reference Git status remained clean: project-main and project-ameed detached HEAD; hackathon_ros2_ws on tanay.

## Evidence

- `runtime/unit-tests.log`
- `runtime/ros-smoke-latest.log` (contains exact per-node runtime directory)
- `runtime/replay-smoke-latest.log` (contains exact replay runtime directory)
- `runtime/replay-config-path.txt` (generated replay config and snapshot directory)
- `runtime/launch-check.log`
- `runtime/demo/preview/layout.html` and `layout.json`
- `log/latest_build/` and `log/package-inventory/`

Tests identified and resolved the system OpenCV/user-site NumPy mismatch, a stale-status assertion in the smoke harness, and ownership/restart edge cases. Use system Python with PYTHONNOUSERSITE=1; the separate SAM3 worker deliberately removes ROS PYTHONPATH/PYTHONHOME while retaining its explicit Python environment.

## Not validated on physical hardware

No physical motion or camera acquisition was started. Base ROS provides MoveIt messages but no franka_msgs; the reference SAM3 interpreter path is absent. Consequently actual Franka trajectory/gripper execution, camera global-clock behaviour, SAM3 loading/inference, hand-eye calibration, scene/contact allowances, physical arrival/attachment accuracy, optional hardware inspection approval, and real placement/empty-depth verification require commissioning. Hardware configuration refuses to start without explicit geometry/calibration/model/environment provisioning and commissioning flags. Tests never report missing hardware behaviour as success.

The source/destination layout remains an explicit configurable assumption, not measured geometry. The sample support-area rule does not establish real tall-stack stability. Hardware verification is transaction-specific operator confirmation. Stud engagement remains explicitly unsupported without a validated seating controller. See README for exact commands, assumptions and recovery procedures.
