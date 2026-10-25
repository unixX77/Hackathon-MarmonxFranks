#!/usr/bin/env python3
"""Audit a saved hover; optionally replay DisplayTrajectory only. No motion APIs."""
import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import select
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/lego_sorter'))
from lego_sorter.preview import audit_display, align_display_start


def saved_display(path):
    data = json.loads(path.read_text())
    if 'trajectory_start' in data:
        return data, 'exact saved DisplayTrajectory'
    joint = data['trajectory']['joint_trajectory']
    # Historical plans did not save trajectory_start. Populate it from the first
    # waypoint for visualization; this does not make an old plan executable.
    return dict(model_id='fr3', trajectory_start=dict(is_diff=False, joint_state=dict(
        name=joint['joint_names'], position=joint['points'][0]['positions'])),
        trajectory=[data['trajectory']]), 'start reconstructed from first waypoint'


def replay(display, seconds, output, interactive=False):
    import rclpy
    from moveit_msgs.msg import DisplayTrajectory
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from rclpy.serialization import serialize_message, deserialize_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rosidl_runtime_py.set_message import set_message_fields

    message = DisplayTrajectory()
    # Humble's setter replaces nested input dictionaries with ROS messages.
    set_message_fields(message, copy.deepcopy(display))
    wire = serialize_message(message)
    decoded = message_to_ordereddict(deserialize_message(wire, DisplayTrajectory))
    if decoded['trajectory'] != display['trajectory']:
        raise RuntimeError('ROS serialization changed the saved trajectory')
    # Normalize default ROS fields before comparing the received message.
    expected = audit_display(decoded)
    (output / 'sent_display.json').write_text(json.dumps(decoded, indent=2, allow_nan=False))
    rclpy.init()
    node = rclpy.create_node('hover_preview_display_only_audit')
    topic = '/lego_sorter/hover_preview'
    received = []

    def receive(msg):
        value = message_to_ordereddict(msg)
        report = audit_display(value)
        report['matches_sent'] = report['display_sha256'] == expected['display_sha256']
        received.append(report)
        (output / 'received_display.json').write_text(json.dumps(value, indent=2, allow_nan=False))

    try:
        # Volatile reader matches RViz's subscription. Publisher stays alive for
        # discovery; a half-second sleep alone cannot demonstrate delivery.
        sub = node.create_subscription(DisplayTrajectory, topic, receive, 10)
        pub = node.create_publisher(DisplayTrajectory, topic, QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        peers = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            peers = [e for e in node.get_subscriptions_info_by_topic(topic)
                     if e.node_name != node.get_name()]
            if peers and pub.get_subscription_count() >= 2:
                break
        endpoints = {t: dict(
            publishers=[dict(node=e.node_namespace + '/' + e.node_name, gid=list(e.endpoint_gid))
                        for e in node.get_publishers_info_by_topic(t)],
            subscribers=[dict(node=e.node_namespace + '/' + e.node_name, gid=list(e.endpoint_gid))
                         for e in node.get_subscriptions_info_by_topic(t)])
                     for t in (topic, '/display_planned_path')}
        pub.publish(message)
        from rclpy.duration import Duration
        acknowledged = pub.wait_for_all_acked(Duration(seconds=seconds))
        deadline = time.monotonic() + seconds
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        report = dict(topic=topic, wire_bytes=len(wire), cdr_roundtrip=expected,
                      reliable_readers_acknowledged=acknowledged,
                      endpoints=endpoints, received=received,
                      external_subscriber_matched=bool(peers))
        (output / 'transport_audit.json').write_text(json.dumps(report, indent=2))
        if not received or not all(x['matches_sent'] for x in received) or not acknowledged:
            raise RuntimeError(f'Preview transport verification failed; see {output}')
        if interactive:
            print('DISPLAY ONLY: preview publisher stays available. No robot commands.\n'
                  'Press Enter to replay after adjusting RViz; type q then Enter to quit.', flush=True)
            print(f'Subscriber nodes: {[e.node_name for e in peers]}', flush=True)
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=.1)
                if select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line or line.strip().lower() == 'q':
                        break
                    pub.publish(message)
                    print('Preview republished. Visualization only.', flush=True)
        return report
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path, help='hover_plan.json or hover_display.json')
    parser.add_argument('--publish-preview', action='store_true', help='Publish visualization only; no planning or execution')
    parser.add_argument('--interactive', action='store_true', help='Keep display publisher alive; Enter replays, q quits (implies --publish-preview)')
    parser.add_argument('--timeout', type=float, default=5.)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 30:
        parser.error('--timeout must be positive and at most 30 seconds')
    display, origin = saved_display(args.plan)
    display, adjustment = align_display_start(display)
    report = dict(source=str(args.plan.resolve()), start_state_source=origin,
                  start_state_alignment_max_rad=adjustment,
                  audit=audit_display(display))
    if args.publish_preview or args.interactive:
        output = ROOT / 'runtime' / ('hover-preview-audit-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        output.mkdir(parents=True)
        report['transport'] = replay(display, args.timeout, output, args.interactive)
        report['evidence'] = str(output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
