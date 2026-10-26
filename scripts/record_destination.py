#!/usr/bin/env python3
"""Record actual stationary TCP poses for destinations or observation. Never move."""
import argparse
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/lego_sorter'))
from lego_sorter.core import atomic, inside


def capture_record(name, position, quaternion, joints, tf_stamp_ns, health, config):
    if name not in ('place', 'discard', 'observation', 'source', 'grasp', 'source_corner_a', 'source_corner_b'):
        raise ValueError('Name must be place, discard, observation, source or grasp')
    if (len(position) != 3 or len(quaternion) != 4 or len(joints) != 7
            or not all(math.isfinite(v) for v in [*position, *quaternion, *joints])):
        raise ValueError('Need finite TCP XYZ, quaternion and seven measured joints')
    if not (config['workspace_x'][0] <= position[0] <= config['workspace_x'][1]
            and config['workspace_y'][0] <= position[1] <= config['workspace_y'][1]):
        raise ValueError('Taught centre is outside configured XY workspace')
    if abs(sum(v*v for v in quaternion)-1.) > .001:
        raise ValueError('TCP quaternion must have unit norm')
    meaning = dict(place='centre_of_6x5_red_layer', discard='centre_of_separate_table_discard_area',
                   observation='camera_observation_pose', source='centre_of_source_grid',
                   grasp='operator_measured_grasp_tcp_pose',source_corner_a='source_boundary_corner',
                   source_corner_b='source_boundary_corner')[name]
    record = dict(recorded_at=datetime.now().astimezone().isoformat(), destination=name,
                meaning=meaning,
                base_frame=config['base_frame'], tcp_frame=config['ee_frame'],
                xy_m=list(position[:2]), captured_tcp_xyz_m=list(position),
                captured_tcp_quaternion_xyzw=list(quaternion),
                measured_arm_joints_rad=list(joints), tcp_tf_stamp_ns=tf_stamp_ns,
                known_table_support_z_m=float(config['table_top_z']), release_tcp_z_m=None,
                robot_health=health, status='RECORDED_ONLY_NOT_COMMISSIONED',
                note='TCP Z and orientation are capture evidence, not taught release height or lattice yaw. '
                     'Full footprints, finger clearance and release geometry still require verification.')
    if name == 'observation':
        x, y, z, w = quaternion
        yaw = math.atan2(2*(x*y+w*z), 1-2*(y*y+z*z))
        record['observation_pose_xyz_yaw'] = [*position, yaw]
        record['tool_down_tilt_deg'] = math.degrees(math.acos(max(-1.,min(1.,-(1-2*(x*x+y*y))))))
        record['note'] = ('Full measured pose retained. The current mock observation model uses XYZ and '
                          'yaw with a tool-down orientation; this does not commission a hardware observation move.')
    if name == 'grasp':
        record['grasp_tcp_above_table_m'] = position[2] - config['table_top_z']
        record['note'] = ('Actual TCP pose recorded as a candidate grasp measurement. '
                          'Recording does not establish alignment, finger clearance or a successful grasp. '
                          'This is not automatically applied to hardware configuration.')
    return record


def save_record(output, record):
    output = inside(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'destinations.json'
    with (output / 'capture.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else dict(
            purpose='Read-only destination teaching; no task config or commissioning flags changed',
            destinations={}, history=[])
        data['history'].append(record)
        data['destinations'][record['destination']] = record
        atomic(path, data)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, choices=['place', 'discard', 'observation', 'source', 'grasp','source_corner_a','source_corner_b'])
    parser.add_argument('--output', type=Path, default=ROOT/'runtime/destination-teaching')
    args = parser.parse_args()
    output = inside(args.output)
    import rclpy
    import yaml
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from lego_sorter.hardware import Feedback, RobotHealth

    c = yaml.safe_load((ROOT/'src/lego_sorter/config/hardware.yaml').read_text())
    rclpy.init()
    node = rclpy.create_node('lego_destination_recorder_readonly')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    feedback, health = Feedback(node, c), RobotHealth(node, c)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        print(f'Recording {args.name}: keep the arm stationary. No motion commands.', flush=True)
        deadline = time.monotonic() + 10
        while True:
            try:
                feedback.stationary()
                health.check_feedback()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        before = feedback.fresh()
        transform = buffer.lookup_transform(c['base_frame'], c['ee_frame'], Time(),
                                             timeout=Duration(seconds=3))
        stamp = Time.from_msg(transform.header.stamp).nanoseconds
        if abs(node.get_clock().now().nanoseconds - stamp)/1e9 > c['feedback_max_age_s']:
            raise RuntimeError('TCP transform is stale; destination was not recorded')
        feedback.stationary()
        after = feedback.fresh()
        if max(abs(after-before)) > c['stationary_joint_delta']:
            raise RuntimeError('Arm moved during capture; destination was not recorded')
        p, q = transform.transform.translation, transform.transform.rotation
        record = capture_record(args.name, [p.x,p.y,p.z], [q.x,q.y,q.z,q.w],
                                after.tolist(), stamp, health.check_feedback(), c)
        path = save_record(output, record)
        print(json.dumps(record, indent=2))
        print(f'Saved: {path}\nNo motion sent. Active sorter configuration is unchanged.')
    finally:
        executor.shutdown(timeout_sec=3)
        thread.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
