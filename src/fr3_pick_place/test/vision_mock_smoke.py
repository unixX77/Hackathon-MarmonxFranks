#!/usr/bin/env python3
"""Run an isolated headless mock cycle; no real hardware or camera is launched."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import secrets

# A separate DDS domain prevents interaction with an existing robot stack.
os.environ['ROS_DOMAIN_ID'] = str(100+secrets.randbelow(100))
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['PYTHONNOUSERSITE'] = '1'

import rclpy
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


def main():
    stack = runner = None
    node = None
    directory = Path(tempfile.mkdtemp(prefix='fr3-vision-smoke-'))
    os.environ['ROS_LOG_DIR'] = str(directory / 'ros')
    print('Logs:', directory, flush=True)
    with (directory/'stack.log').open('w') as stack_log, (directory/'runner.log').open('w') as runner_log:
        try:
            stack = subprocess.Popen(['ros2', 'launch', 'fr3_pick_place', 'vision.launch.py', 'use_rviz:=false'],
                                     stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True)
            rclpy.init()
            node = rclpy.create_node('vision_mock_smoke')
            observer = node.create_client(Trigger, 'vision/observe')
            scene = node.create_client(GetPlanningScene, 'get_planning_scene')
            if not observer.wait_for_service(timeout_sec=45) or not scene.wait_for_service(timeout_sec=45):
                raise RuntimeError('Mock stack unavailable; inspect stack.log.')
            runner = subprocess.Popen(['ros2', 'run', 'fr3_pick_place', 'vision_pick_place.py', '--config',
                str(ROOT/'config/vision_demo.yaml'), '--execute', '--auto-confirm'],
                stdout=runner_log, stderr=subprocess.STDOUT, start_new_session=True)
            code = runner.wait(timeout=180)
            runner_log.flush()
            output = (directory/'runner.log').read_text()
            print(output, flush=True)
            if code:
                raise RuntimeError(f'Mock runner exited {code}.')
            assert 'Bounded XY/yaw correction' in output, 'Correction was not exercised'
            assert 'cycle complete' in output
            req = GetPlanningScene.Request()
            req.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                                         | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
            future = scene.call_async(req)
            rclpy.spin_until_future_complete(node, future, timeout_sec=10)
            assert future.done()
            result = future.result().scene
            assert not any(o.object.id == 'fr3_vision_brick' for o in result.robot_state.attached_collision_objects)
            block = next(o for o in result.world.collision_objects if o.id == 'fr3_vision_brick')
            p = block.primitive_poses[0].position
            # MoveIt may normalize shape poses into a separate object.pose.
            origin = block.pose
            centre = Rotation.from_quat([origin.orientation.x, origin.orientation.y,
                origin.orientation.z, origin.orientation.w]).apply([p.x, p.y, p.z])
            centre += [origin.position.x, origin.position.y, origin.position.z]
            print('Placed object centre in base:', centre, flush=True)
            assert abs(centre[0]-.38) < .003 and abs(centre[1]-.10) < .003 and abs(centre[2]-.362) < .003
            print('PASS: approach, correction, stable observations, descent, attach, transfer, detach and retreat.', flush=True)
        finally:
            stop(runner)
            stop(stack)
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
