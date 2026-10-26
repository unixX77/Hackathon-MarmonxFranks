#!/usr/bin/env python3
"""Small horizontal tool-X shift at hover. Real motion requires explicit --execute."""
import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/lego_sorter'))
from lego_sorter.preview import horizontal_tool_shift
from observe_robot_frame import plan_hover


def execute_shift(node, buffer, feedback, c, trajectory, target, output):
    """Execute the freshly checked shift using the shared monitored action adapter."""
    import numpy as np
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.time import Time
    from moveit_msgs.action import ExecuteTrajectory
    from lego_sorter.hardware import SafeMover, RobotHealth

    class ShiftExecutor:
        action = SafeMover.action
        execute = SafeMover.execute

    runner = ShiftExecutor()
    runner.feedback = feedback
    runner.health = RobotHealth(node, c)
    runner.cancel = threading.Event()
    runner.executor = ActionClient(node, ExecuteTrajectory, 'execute_trajectory')
    started = False
    target_xyz = np.array([target.position.x, target.position.y, target.position.z])
    report = dict(target_tcp_m=target_xyz.tolist())
    names = [f'fr3_joint{i}' for i in range(1, 8)]
    try:
        if not runner.executor.wait_for_server(timeout_sec=5):
            raise RuntimeError('MoveIt execute_trajectory action unavailable')
        runner.health.wait_ready()
        joint = trajectory.joint_trajectory
        first = np.array([joint.points[0].positions[joint.joint_names.index(n)] for n in names])
        feedback.stationary()
        if np.max(np.abs(feedback.fresh() - first)) > .01:
            raise RuntimeError('Robot no longer matches the shift start; refusing execution')
        with feedback.lock:
            if feedback.guard_moved:
                raise RuntimeError('Robot moved during shift planning; refusing execution')
        last = joint.points[-1].time_from_start
        duration = last.sec + last.nanosec / 1e9
        print(f'EXECUTING RELATIVE HOVER SHIFT: {duration:.1f} seconds; no gripper commands or descent.', flush=True)
        (output / 'shift_execution.json').write_text(json.dumps(dict(
            status='STARTED', **report)))
        started = True
        runner.execute('RELATIVE_HOVER_SHIFT', trajectory)
        deadline = time.monotonic() + 3
        while True:
            try:
                feedback.stationary()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        arrival_tf = buffer.lookup_transform(c['base_frame'], c['ee_frame'], Time(),
                                             timeout=Duration(seconds=3))
        tf = arrival_tf.transform
        p, q = tf.translation, tf.rotation
        tf_age = abs(node.get_clock().now().nanoseconds - Time.from_msg(arrival_tf.header.stamp).nanoseconds) / 1e9
        arrival = np.array([p.x, p.y, p.z])
        error = float(np.linalg.norm(arrival - target_xyz))
        a = np.array([q.x, q.y, q.z, q.w])
        b = np.array([target.orientation.x, target.orientation.y,
                      target.orientation.z, target.orientation.w])
        angle = math.degrees(2*math.acos(float(np.clip(abs(a @ b) / (np.linalg.norm(a)*np.linalg.norm(b)), 0., 1.))))
        report.update(arrival_tcp_m=arrival.tolist(), arrival_error_m=error,
                      arrival_angle_deg=angle, tf_age_s=tf_age,
                      measured_joints=feedback.fresh().tolist(), robot_health=runner.health.check())
        if tf_age > c['feedback_max_age_s']:
            raise RuntimeError('Arrival TCP transform is stale')
        if error > c['arrival_abort'] or angle > c['arrival_angle_deg']:
            raise RuntimeError(f'Shift arrival check failed: {1000*error:.1f} mm, {angle:.2f} degrees')
        (output / 'shift_execution.json').write_text(json.dumps(dict(status='COMPLETE', **report)))
        print('Shift complete. Robot holds at hover; no grasp, descent or automatic return.', flush=True)
    except BaseException as exc:
        runner.cancel.set()
        (output / 'shift_execution.json').write_text(json.dumps(dict(
            status='INTERRUPTED_OR_FAILED' if started else 'REFUSED_BEFORE_EXECUTION', detail=str(exc),
            health_at_failure=runner.health.snapshot(), **report)))
        raise
    finally:
        runner.executor.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool-x-mm', type=float, help='Signed tool-X displacement; default +20 mm for planning, required explicitly for execution; maximum magnitude 25 mm')
    parser.add_argument('--execute', action='store_true', help='REAL MOTION: freshly plan and execute this relative shift; no gripper commands')
    parser.add_argument('--config', type=Path, default=ROOT / 'src/lego_sorter/config/hardware.yaml')
    args = parser.parse_args()
    if args.execute and args.tool_x_mm is None:
        parser.error('--execute requires an explicit signed --tool-x-mm')
    if args.tool_x_mm is None:
        args.tool_x_mm = 20.
    if not math.isfinite(args.tool_x_mm) or not 0 < abs(args.tool_x_mm) <= 25:
        parser.error('--tool-x-mm must be finite, nonzero, and within +/-25 mm')

    import rclpy
    import yaml
    from geometry_msgs.msg import Pose
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from lego_sorter.hardware import Feedback

    c = yaml.safe_load(args.config.read_text())
    output = ROOT / 'runtime' / ('tool-shift-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)
    print(f'Results: {output}', flush=True)
    print('REAL RELATIVE SHIFT EXECUTION REQUESTED. No camera capture or gripper commands.' if args.execute
          else 'PLAN ONLY: no camera capture, arm execution or gripper commands.', flush=True)
    rclpy.init()
    node = rclpy.create_node('lego_tool_shift_diagnostic')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    feedback = Feedback(node, c)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                feedback.stationary()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('No stationary fresh feedback; check drivers and keep the robot still')
                time.sleep(.05)
        with feedback.lock:
            feedback.guard = feedback.samples[-1][1].copy()
            feedback.guard_moved = False
        tf = buffer.lookup_transform(c['base_frame'], c['ee_frame'], Time(),
                                     timeout=Duration(seconds=3))
        age = abs(node.get_clock().now().nanoseconds - Time.from_msg(tf.header.stamp).nanoseconds) / 1e9
        if age > c['feedback_max_age_s']:
            raise RuntimeError('TCP transform is stale')
        p, q = tf.transform.translation, tf.transform.rotation
        start = [p.x, p.y, p.z]
        quaternion = [q.x, q.y, q.z, q.w]
        target_xyz = horizontal_tool_shift(start, quaternion, args.tool_x_mm / 1000)
        target = Pose()
        target.position.x, target.position.y, target.position.z = target_xyz
        target.orientation = copy.deepcopy(q)
        context = dict(operation='relative_tool_x_alignment', execute_requested=args.execute,
                       requested_shift_tool_x_mm=args.tool_x_mm,
                       start_tcp_m=start, target_tcp_m=target_xyz,
                       orientation_xyzw=quaternion, tcp_tf_age_s=age,
                       base_z_held_constant=True, orientation_held_constant=True,
                       measurement='Operator estimate; not a persistent calibration correction')
        (output / 'shift_request.json').write_text(json.dumps(context, indent=2, allow_nan=False))
        print(f'Current TCP: {start}\nProposed TCP: {target_xyz}\nOrientation unchanged; Z unchanged.', flush=True)
        trajectory = plan_hover(node, buffer, feedback, c, None, output, .05, execute=False,
                                target_pose=target, diagnostic_context=context)
        if args.execute:
            execute_shift(node, buffer, feedback, c, trajectory, target, output)
        else:
            print('Relative shift planned only. Preview with:\n'
                  f'/usr/bin/python3 -s scripts/play_hover_preview.py {output / "hover_display.json"} --interactive', flush=True)
    finally:
        with feedback.lock:
            feedback.guard = None
        executor.shutdown(timeout_sec=3)
        thread.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
