#!/usr/bin/env python3
"""Camera/TF check and hover planning; motion requires explicit --execute-hover."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/lego_sorter'))


def plan_hover(node, buffer, feedback, c, brick, output, clearance, execute=False):
    """Capture-based hover only; default is planning/display without execution."""
    import numpy as np
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import DisplayTrajectory, PlanningSceneComponents, PlanningScene
    from moveit_msgs.srv import GetCartesianPath, GetPlanningScene, ApplyPlanningScene, GetPositionFK
    from rclpy.duration import Duration
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from rclpy.time import Time
    from rosidl_runtime_py.convert import message_to_ordereddict
    from lego_sorter.hardware import wait_future, table_collision
    from lego_sorter.vendor.ameed_motion import nearest_yaw, tool_down_quaternion, swing_violation, retime

    feedback.stationary()
    xyz = list(brick['top_base_raw_m'])
    xyz[2] += clearance  # TCP height above the measured top, not a grasp pose.
    if not (c['workspace_x'][0] <= xyz[0] <= c['workspace_x'][1]
            and c['workspace_y'][0] <= xyz[1] <= c['workspace_y'][1]
            and c['table_top_z'] + clearance <= xyz[2] <= c['max_hover_z']):
        raise ValueError('Hover target outside configured workspace/height limits')
    tf = buffer.lookup_transform(c['base_frame'], c['ee_frame'], Time(),
                                 timeout=Duration(seconds=3)).transform
    q = tf.rotation
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    yaw = nearest_yaw(math.radians(brick['yaw_base_deg']), math.atan2(rot[1, 0], rot[0, 0]))
    target = Pose()
    target.position.x, target.position.y, target.position.z = map(float, xyz)
    target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = tool_down_quaternion(yaw)
    names = [f'fr3_joint{i}' for i in range(1, 8)]
    request = GetCartesianPath.Request()
    request.header.frame_id = c['base_frame']
    request.group_name, request.link_name = 'fr3_arm', c['ee_frame']
    request.start_state.is_diff = True
    request.start_state.joint_state.name = names
    request.start_state.joint_state.position = feedback.fresh().tolist()
    request.waypoints = [target]
    request.max_step = .005
    request.revolute_jump_threshold = .3
    request.avoid_collisions = True
    scene_client = node.create_client(GetPlanningScene, 'get_planning_scene')
    path_client = node.create_client(GetCartesianPath, 'compute_cartesian_path')
    apply_client = node.create_client(ApplyPlanningScene, 'apply_planning_scene')
    if not (scene_client.wait_for_service(timeout_sec=5) and path_client.wait_for_service(timeout_sec=5)
            and apply_client.wait_for_service(timeout_sec=5)):
        raise RuntimeError('MoveIt planning services unavailable')
    table = table_collision(c)
    diff = PlanningScene(is_diff=True)
    diff.robot_state.is_diff = True
    diff.world.collision_objects = [table]
    if not wait_future(apply_client.call_async(ApplyPlanningScene.Request(scene=diff)), 10).success:
        raise RuntimeError('MoveIt rejected the table collision object')
    scene_req = GetPlanningScene.Request()
    scene_req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
    scene = wait_future(scene_client.call_async(scene_req), 10).scene
    objects = [obj.id for obj in scene.world.collision_objects]
    if 'lego_table' not in objects:
        raise RuntimeError('Table collision object missing from MoveIt scene after applying it')
    print(f'Current MoveIt world objects: {objects}', flush=True)
    print(f'Table added: centre={c["table_center"]}, size={c["table_size"]} m; top Z={c["table_top_z"]} m.', flush=True)
    print('Camera/mount and other fixtures are not added. No collision exemptions were introduced.', flush=True)
    result = wait_future(path_client.call_async(request), 20)
    if result.error_code.val != 1 or result.fraction < .999:
        raise RuntimeError(f'Hover path refused: fraction={result.fraction:.3f}, MoveIt code={result.error_code.val}')
    trajectory = result.solution.joint_trajectory
    if len(trajectory.points) < 2:
        raise RuntimeError('Hover planner returned fewer than two trajectory points')
    indices = [trajectory.joint_names.index(n) for n in names]
    positions = np.array([[p.positions[i] for i in indices] for p in trajectory.points])
    if np.max(np.abs(positions[0] - request.start_state.joint_state.position)) > .03:
        raise RuntimeError('Plan does not start at the measured joint state')
    if np.max(np.abs(np.diff(positions, axis=0))) > .3:
        raise RuntimeError('Hover plan has a joint discontinuity')
    violation = swing_violation(np.ptp(positions, axis=0), c['max_joint_swing'], c['max_wrist_swing'])
    if violation:
        raise RuntimeError(f'Hover plan exceeds configured joint swing: {violation}')
    # GetCartesianPath's default timing is too fast for this inspection preview.
    # Use the main motion adapter's retiming helper, capped at 2% joint-speed scale.
    speed_scale = min(float(c['velocity_scale']), .02)
    times, velocities = retime(positions, speed_scale)
    # Evaluate TCP positions, rather than mistaking joint scaling for metres/sec.
    fk_client = node.create_client(GetPositionFK, 'compute_fk')
    if not fk_client.wait_for_service(timeout_sec=5):
        raise RuntimeError('compute_fk unavailable; cannot check planned TCP segment speeds')
    tcp_positions = []
    for point in trajectory.points:
        fk = GetPositionFK.Request()
        fk.header.frame_id = c['base_frame']
        fk.fk_link_names = [c['ee_frame']]
        fk.robot_state.is_diff = True
        fk.robot_state.joint_state.name = list(trajectory.joint_names)
        fk.robot_state.joint_state.position = list(point.positions)
        response = wait_future(fk_client.call_async(fk), 5)
        if response.error_code.val != 1 or len(response.pose_stamped) != 1:
            raise RuntimeError('TCP forward-kinematics check failed')
        p = response.pose_stamped[0].pose.position
        tcp_positions.append([p.x, p.y, p.z])
    # Conservative preview/demo setting: segment averages <= 10 mm/s.
    # This is a sampled planning check, not a continuous hardware speed limiter.
    speed_limit = float(c.get('hover_tcp_segment_speed_m_s', .01))
    if not math.isfinite(speed_limit) or not 0 < speed_limit <= .02:
        raise ValueError('Hover TCP segment speed must be positive and at most .02 m/s')
    lengths = np.linalg.norm(np.diff(np.asarray(tcp_positions), axis=0), axis=1)
    peak = float(np.max(lengths / np.diff(times)))
    stretch = max(1., peak / speed_limit)
    times *= stretch
    velocities /= stretch
    for point, stamp, velocity in zip(trajectory.points, times, velocities):
        point.time_from_start.sec = int(stamp)
        point.time_from_start.nanosec = int((stamp % 1) * 1e9)
        point.velocities = [float(velocity[names.index(joint)]) for joint in trajectory.joint_names]
        point.accelerations = []
    feedback.stationary()
    with feedback.lock:
        if feedback.guard_moved:
            raise RuntimeError('Robot moved since capture; discard the hover plan')
    summary = dict(purpose='Hover-only plan; execution requires explicit --execute-hover.',
                   target_tcp_m=xyz, target_yaw_deg=math.degrees(yaw),
                   clearance_above_brick_top_m=clearance, fraction=result.fraction,
                   preview_joint_speed_scale=speed_scale, preview_duration_s=float(times[-1]),
                   maximum_tcp_segment_average_m_s=peak/stretch,
                   tcp_segment_speed_limit_m_s=speed_limit,
                   world_object_ids=objects, camera_collision_model_added=False,
                   table_collision=message_to_ordereddict(table),
                   trajectory=message_to_ordereddict(result.solution))
    (output / 'hover_plan.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    display_topic = '/lego_sorter/hover_preview'
    publisher = node.create_publisher(DisplayTrajectory, display_topic, qos)
    display = DisplayTrajectory()
    display.model_id = 'fr3'
    display.trajectory_start = result.start_state
    display.trajectory = [result.solution]
    # Save the complete message, including the start state, so the exact preview
    # can be audited/replayed without camera capture or any motion interface.
    from lego_sorter.preview import audit_display, align_display_start
    from rosidl_runtime_py.set_message import set_message_fields
    display_data, start_adjustment = align_display_start(message_to_ordereddict(display))
    # The Cartesian service's response.start_state may contain the endpoint.
    # Only display metadata is corrected; the executed RobotTrajectory is untouched.
    from moveit_msgs.msg import RobotState
    display.trajectory_start = RobotState()
    import copy
    set_message_fields(display.trajectory_start, copy.deepcopy(display_data['trajectory_start']))
    display_audit = audit_display(display_data)
    (output / 'hover_display.json').write_text(json.dumps(display_data, indent=2, allow_nan=False))
    (output / 'hover_display_audit.json').write_text(json.dumps(display_audit, indent=2))
    # DDS discovery can take longer than .5 seconds. Match readers before
    # sending, then check transport acknowledgement rather than assuming delivery.
    deadline = time.monotonic() + 5
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        time.sleep(.05)  # The background executor handles ROS callbacks.
    readers = publisher.get_subscription_count()
    publisher.publish(display)
    acknowledged = publisher.wait_for_all_acked(Duration(seconds=3))
    print(f'PLAN ONLY: TCP hover target (m)={np.round(xyz, 5).tolist()}, yaw={math.degrees(yaw):.1f} deg')
    print(f'Path fraction: {result.fraction:.3f}; plan saved: {output / "hover_plan.json"}')
    print(f'Preview timing: {times[-1]:.1f} seconds, joint-speed scale {100 * speed_scale:.1f}%')
    print(f'Maximum planned TCP segment-average speed: {1000 * peak/stretch:.1f} mm/s '
          '(sampled check, not a hardware speed limiter)')
    print(f'Published {display_topic} for RViz review. No trajectory was executed.')
    print(f'Preview transport: {readers} matched subscriber(s); acknowledged={acknowledged}. '
          'This does not verify visible playback.')
    print(f'Preview start-state alignment: maximum correction {start_adjustment:.6f} rad.')
    if execute:
        from rclpy.action import ActionClient
        from moveit_msgs.action import ExecuteTrajectory
        from lego_sorter.hardware import SafeMover

        class HoverExecutor:
            # Use the main adapter's monitored execution and cancellation behavior.
            action = SafeMover.action
            execute = SafeMover.execute

        runner = HoverExecutor()
        runner.feedback = feedback
        runner.cancel = threading.Event()
        runner.executor = ActionClient(node, ExecuteTrajectory, 'execute_trajectory')
        if not runner.executor.wait_for_server(timeout_sec=5):
            raise RuntimeError('MoveIt execute_trajectory action unavailable')
        feedback.stationary()
        if np.max(np.abs(feedback.fresh() - positions[0])) > .01:
            raise RuntimeError('Robot no longer matches the trajectory start; recapture and replan')
        with feedback.lock:
            if feedback.guard_moved:
                raise RuntimeError('Robot moved during planning; refusing hover execution')
        print(f'EXECUTING ONE HOVER ONLY: estimated {times[-1]:.1f} seconds. No grasp or descent beyond hover.', flush=True)
        (output / 'hover_execution.json').write_text(json.dumps(dict(status='STARTED',target_tcp_m=xyz)))
        try:
            runner.execute('ONE_BRICK_HOVER', result.solution)
            # Guard tracks expected arm motion now; it is not a failure during execution.
            deadline = time.monotonic() + 3
            while True:
                try:
                    feedback.stationary()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.05)
            arrival = buffer.lookup_transform(c['base_frame'], c['ee_frame'], Time(),
                timeout=Duration(seconds=3)).transform.translation
            error = float(np.linalg.norm(np.array([arrival.x, arrival.y, arrival.z]) - xyz))
            if error > c['arrival_abort']:
                raise RuntimeError(f'Hover arrival position error {1000 * error:.1f} mm exceeds configured tolerance')
            (output / 'hover_execution.json').write_text(json.dumps(dict(status='COMPLETE',target_tcp_m=xyz,arrival_error_m=error)))
            print('Hover complete. Robot stays at hover; no grasp, release or automatic return was commanded.')
        except BaseException as exc:
            runner.cancel.set()
            (output / 'hover_execution.json').write_text(json.dumps(dict(status='INTERRUPTED_OR_FAILED',detail=str(exc))))
            raise
        finally:
            runner.executor.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'src/lego_sorter/config/hardware.yaml')
    parser.add_argument('--plan-hover', action='store_true', help='Plan/display a hover above exactly one high-confidence red brick; never execute')
    parser.add_argument('--execute-hover', action='store_true', help='EXPLICIT REAL MOTION: capture, plan and execute one hover; no gripper commands')
    parser.add_argument('--hover-clearance', type=float, default=.10, help='TCP clearance above detected brick top in metres (default .10)')
    opts = parser.parse_args()
    opts.plan_hover = opts.plan_hover or opts.execute_hover
    if not math.isfinite(opts.hover_clearance) or not .05 <= opts.hover_clearance <= .30:
        parser.error('--hover-clearance must be between .05 and .30 metres')
    import numpy as np
    import yaml
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from lego_sorter.core import calibration, matrix
    from lego_sorter.hardware import Feedback
    from lego_sorter.perception import SamProcess

    c = yaml.safe_load(opts.config.read_text())
    pc = calibration(c)  # Preserves frame/rigidity checks and explicit rejection override.
    output = ROOT / 'runtime' / ('robot-frame-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)
    c['runtime'] = str(output)
    config_path = output / 'config.yaml'
    config_path.write_text(yaml.safe_dump(c))
    print(f'Results: {output}', flush=True)
    if opts.execute_hover:
        print('REAL HOVER EXECUTION REQUESTED. Keep the arm stationary until planning finishes; no gripper commands.', flush=True)
    else:
        print('Keep the arm stationary. No arm or gripper commands will be sent.', flush=True)
    print(f'Calibration override: {c.get("allow_rejected_calibration", False)}', flush=True)

    rclpy.init()
    node = rclpy.create_node('lego_camera_frame_check')
    buffer = Buffer(cache_time=Duration(seconds=120.0))
    listener = TransformListener(buffer, node)
    feedback = Feedback(node, c)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    worker = SamProcess(c, str(config_path))
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                feedback.stationary()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('No stationary fresh FR3 feedback. Check bringup and keep the robot still.')
                time.sleep(.05)
        guard = feedback.fresh()
        with feedback.lock:
            feedback.guard = guard
            feedback.guard_moved = False
        worker.start()
        feedback.stationary()
        oid = uuid.uuid4().hex
        worker.send({'id': oid})
        captured = worker.read(15)
        if captured.get('event') != 'captured' or captured.get('id') != oid:
            raise RuntimeError('Unexpected camera capture response')
        quality = captured.get('capture_quality', {})
        if quality:
            print(f'Usable depth across image: {100 * quality["usable_depth_fraction"]:.1f}%; '
                  f'zero: {100 * quality["zero_depth_fraction"]:.1f}%; '
                  f'max raw value: {100 * quality["max_raw_depth_fraction"]:.1f}%', flush=True)
        transform = buffer.lookup_transform(c['base_frame'], c['calibration_parent'],
            Time(nanoseconds=captured['stamp_ns']), timeout=Duration(seconds=5.0)).transform
        p, q = transform.translation, transform.rotation
        bp = matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        worker.send(dict(event='transform', id=oid, base_parent=bp.tolist(), parent_camera=pc.tolist()))
        result = worker.read(c['snapshot_timeout_s'])
        if result.get('event') != 'result' or result.get('id') != oid:
            raise RuntimeError('Unexpected detector response')
        feedback.stationary()
        with feedback.lock:
            if feedback.guard_moved:
                raise RuntimeError('Robot moved during the capture/inference; discard this observation')
        bc = bp @ pc
        bricks = []
        for b in result['metadata']['bricks']:
            if not b.get('id') or b['status'] not in ('brick', 'split'):
                continue
            centre = (bc @ np.r_[b['box_centre_cam_m'], 1])[:3]
            top = (bc @ np.r_[b['centre_cam_m'], 1])[:3]
            rotation = bc[:3, :3] @ np.asarray(b['box_rotation_cam'])
            bricks.append(dict(id=b['id'], status=b['status'], colour=b.get('colour'),
                confidence=b.get('colour_confidence'), centre_camera_m=b['box_centre_cam_m'],
                centre_base_raw_m=centre.tolist(), top_base_raw_m=top.tolist(),
                yaw_base_deg=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))))
        report = dict(purpose='Read-only coordinate inspection; not a motion target or accuracy certification',
            capture_stamp_ns=captured['stamp_ns'], base_frame=c['base_frame'],
            calibration_parent=c['calibration_parent'], camera_frame=c['camera_frame'],
            calibration_file=c['calibration_file'], allow_rejected_calibration=c.get('allow_rejected_calibration', False),
            base_from_parent=bp.tolist(), parent_from_camera=pc.tolist(),
            configured_camera_height_bias_m=c['camera_height_bias'],
            corrections_applied='None: raw calibrated coordinates; no height bias, snapping or TCP correction',
            observation_path=result['path'], capture_quality=quality, bricks=bricks)
        (output / 'robot_frame_observation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'TCP/calibration-parent position in base (m): {bp[:3, 3].round(5).tolist()}')
        print(f'Accepted detector candidates: {len(bricks)}')
        for b in bricks:
            print(f'Brick {b["id"]}: {b["colour"]}, confidence={b["confidence"]}; '
                  f'raw base centre (m)={np.round(b["centre_base_raw_m"], 5).tolist()}; '
                  f'base yaw={b["yaw_base_deg"]:.1f} deg')
        print(f'Report: {output / "robot_frame_observation.json"}')
        preview = Path(result['path']) / 'preview.png'
        if preview.exists():
            print(f'Preview: {preview}')
        else:
            print(f'No fitted preview generated. Captured RGB: {Path(result["path"]) / "rgb.png"}')
        if not bricks:
            print('No accepted poses: the camera-to-base brick check is still pending.')
        if opts.plan_hover:
            if len(bricks) != 1 or bricks[0]['colour'] != 'red' or bricks[0]['confidence'] != 'high':
                raise RuntimeError('Hover diagnostic requires exactly one accepted high-confidence red brick')
            plan_hover(node, buffer, feedback, c, bricks[0], output, opts.hover_clearance, opts.execute_hover)
        if not opts.execute_hover:
            print('No motion sent. These coordinates still need independent physical comparison.')
    finally:
        worker.close()
        with feedback.lock:
            feedback.guard = None
        executor.shutdown(timeout_sec=3)
        spin_thread.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
