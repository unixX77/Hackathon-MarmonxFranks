#!/usr/bin/env python3
"""Wall-clock RViz pose playback. Publishes DisplayRobotState only; no motion APIs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import select
import sys
import time

from audit_hover_preview import ROOT, saved_display
from lego_sorter.preview import PreviewSampler


def play(display, output, interactive=False, timeout=10.):
    import rclpy
    from moveit_msgs.msg import DisplayRobotState
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from rosidl_runtime_py.set_message import set_message_fields

    sampler = PreviewSampler(display)
    rclpy.init()
    node = rclpy.create_node('hover_preview_wall_clock_player')
    topic = '/lego_sorter/hover_preview_state'
    publisher = node.create_publisher(DisplayRobotState, topic, QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    report = dict(purpose='DisplayRobotState visualization only; no robot commands',
                  topic=topic, saved_duration_s=sampler.duration, runs=[])
    run = None
    try:
        deadline = time.monotonic() + timeout
        while publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError('No preview-state subscriber. Load hover_state_preview.rviz in the existing RViz first.')
            rclpy.spin_once(node, timeout_sec=.1)
        print(f'DISPLAY ONLY: {sampler.duration:.3f} seconds, paced by the Python wall clock.\n'
              f'Topic: {topic}. No arm, gripper, controller or TF commands.', flush=True)
        message = DisplayRobotState()
        message.hide = False
        # Highlight the ghost so it is distinct from any physical-state robot.
        set_message_fields(message, dict(highlight_links=[dict(id=name, color=dict(
            r=.2, g=.8, b=1., a=1.)) for name in
            [*[f'fr3_link{i}' for i in range(9)], 'fr3_hand', 'fr3_hand_tcp',
             'fr3_leftfinger', 'fr3_rightfinger']]))

        def publish(elapsed):
            from moveit_msgs.msg import RobotState
            message.state = RobotState()
            set_message_fields(message.state, sampler.state(elapsed))
            message.state.joint_state.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(message)

        publish(0.)
        start = None if interactive else time.monotonic()
        run = None
        next_log = 0.
        if interactive:
            print('Press Enter to start a full playback. Type q then Enter to quit. No automatic looping.', flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1/30)
            if interactive and select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line or line.strip().lower() == 'q':
                    break
                if run is not None:
                    run['interrupted_by_replay'] = True
                start, run, next_log = time.monotonic(), None, 0.
            if start is None:
                continue
            if run is None:
                run = dict(frame_count=0, completed=False, samples=[])
                report['runs'].append(run)
            elapsed = time.monotonic() - start
            sample_time = min(elapsed, sampler.duration)
            publish(sample_time)
            run['frame_count'] += 1
            if elapsed >= next_log or elapsed >= sampler.duration:
                joint = message.state.joint_state
                run['samples'].append(dict(wall_elapsed_s=elapsed, trajectory_time_s=sample_time,
                                           joint_names=list(joint.name), positions=list(joint.position)))
                print(f'Preview {sample_time:5.1f} / {sampler.duration:.1f} s '
                      f'(wall elapsed {elapsed:.1f} s)', flush=True)
                next_log += 5.
            if elapsed >= sampler.duration:
                from rclpy.duration import Duration
                run['final_frame_acknowledged'] = publisher.wait_for_all_acked(Duration(seconds=2))
                run.update(completed=True, wall_duration_s=elapsed)
                (output / 'wall_clock_playback.json').write_text(json.dumps(report, indent=2))
                print('Preview complete. No robot motion was commanded.', flush=True)
                start, run = None, None
                if not interactive:
                    break
        return report
    finally:
        if run is not None:
            run['stopped_before_completion'] = True
        (output / 'wall_clock_playback.json').write_text(json.dumps(report, indent=2))
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path, help='Saved hover_display.json or hover_plan.json')
    parser.add_argument('--interactive', action='store_true', help='Enter starts/restarts playback; q exits')
    args = parser.parse_args()
    display, _ = saved_display(args.plan)
    output = ROOT / 'runtime' / ('hover-state-playback-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)
    print(f'Evidence: {output}', flush=True)
    try:
        play(display, output, args.interactive)
    except KeyboardInterrupt:
        print('Display playback stopped. No robot commands were sent.')


if __name__ == '__main__':
    main()
