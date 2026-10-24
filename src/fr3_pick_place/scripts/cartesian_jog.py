#!/usr/bin/env python3
"""Operator-gated base-Z steps and measured TCP recording; plan-only by default."""
import argparse
import copy
import json
import math
from contextlib import contextmanager
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import yaml
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity

from pick_place import Task, box
from pick_place_cartesian import ARM, BASE, TCP, CartesianPlanner, coordinates, validate_vertical
from pick_place_stepwise import StepwiseTask, OperatorStop, check_start, confirm


def is_mock_arm(urdf):
    """Classify the actual controller-manager model, never a user bypass flag."""
    systems = [system for system in ET.fromstring(urdf).findall('ros2_control')
               if set(ARM).intersection(joint.get('name') for joint in system.findall('joint'))]
    if len(systems) != 1:
        raise RuntimeError('Cannot identify one FR3 arm hardware component in controller_manager model.')
    plugin = systems[0].findtext('hardware/plugin', '').strip()
    if plugin == 'mock_components/GenericSystem':
        return True
    if plugin == 'franka_hardware/FrankaHardwareInterface':
        return False
    raise RuntimeError(f'Unrecognized arm hardware plugin {plugin!r}; refusing execution.')


def parse_step(value):
    step = float(value)
    if not math.isfinite(step) or not 0.001 <= abs(step) <= 0.020:
        raise ValueError('Use a signed step from 0.001 to 0.020 metres (1 to 20 mm).')
    return step


def record_pose(path, name, pose, joints):
    if not name or not all(c.isalnum() or c in '_-' for c in name):
        raise ValueError('Pose name must contain only letters, numbers, underscores or hyphens.')
    xyz, quat = coordinates(pose)
    data = {'frame_id': BASE, 'tcp_link': TCP, 'poses': {}}
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('frame_id') != BASE or data.get('tcp_link') != TCP or not isinstance(data.get('poses'), dict):
            raise ValueError('Existing pose file has an incompatible format.')
    if name in data['poses']:
        raise ValueError(f'Pose {name!r} already exists; choose a new name or output file.')
    data['poses'][name] = {'xyz_m': xyz, 'quaternion_xyzw': quat,
                           'arm_joints_rad': {n: joints[n] for n in ARM},
                           'recorded_unix_s': time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


class ScenePlanner(CartesianPlanner):
    """Check jogs against the shared scene, retaining existing attachments."""

    @contextmanager
    def scoped_scene(self, carrying=False, placed=False):
        if carrying or placed:
            raise RuntimeError('Teaching jogs do not model grasp/release transitions.')
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                                     | PlanningSceneComponents.ROBOT_STATE
                                     | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        original = self.call(self.task.scene_client, req).scene
        if self.task.hardware:
            table = next((o for o in original.world.collision_objects
                          if o.id == self.task.table_object_id), None)
            if table is None or not (table.primitives or table.meshes or table.planes):
                raise RuntimeError(f'Measured table collision object {self.task.table_object_id!r} '
                                   'missing from MoveIt; add it before hardware jogging.')
        yield original.robot_state

    def state(self, original, carrying, joints):
        # Keep every existing attachment, including a camera or modeled payload.
        state = copy.deepcopy(original)
        state.is_diff = False
        values = dict(zip(state.joint_state.name, state.joint_state.position))
        values.update(joints)
        state.joint_state.name = list(values)
        state.joint_state.position = list(values.values())
        state.joint_state.velocity = []
        state.joint_state.effort = []
        state.joint_state.header.stamp.sec = state.joint_state.header.stamp.nanosec = 0
        return state

    def revalidate(self, trajectory, approved, target_z):
        # Recheck against the current scene after the operator's preview pause.
        with self.scoped_scene() as original:
            start = self.fk(approved)

            def inspect(joints):
                if any(not self.bounds[n][0] <= q <= self.bounds[n][1] for n, q in joints.items()):
                    raise RuntimeError('Joint limit exceeded.')
                candidate = self.state(original, False, joints)
                valid = self.call(self.validity, GetStateValidity.Request(
                    robot_state=candidate, group_name='fr3_arm'))
                if not valid.valid:
                    raise RuntimeError('Current scene invalidates the previewed path; replan.')
                return self.fk(candidate)

            validate_vertical(trajectory, start, target_z, inspect, self.limits)


class JogTask:
    # Reuse the existing timestamp, settling and action cancellation checks.
    wait = Task.wait
    action = Task.action
    on_joints = StepwiseTask.on_joints
    snapshot = StepwiseTask.snapshot
    settled_snapshot = StepwiseTask.settled_snapshot

    def __init__(self, node, args):
        self.node, self.execute, self.hardware = node, args.execute, False
        self.endpoint_z_tolerance_m = args.endpoint_z_tolerance
        if self.execute:
            parameters = node.create_client(GetParameters, 'controller_manager/get_parameters')
            if not parameters.wait_for_service(timeout_sec=10):
                raise RuntimeError('Cannot verify real/fake hardware: controller_manager parameters unavailable.')
            response = self.wait(parameters.call_async(GetParameters.Request(names=['robot_description'])), 10)
            if (response is None or len(response.values) != 1
                    or response.values[0].type != ParameterType.PARAMETER_STRING):
                raise RuntimeError('Cannot verify controller_manager robot_description; refusing execution.')
            self.hardware = not is_mock_arm(response.values[0].string_value)
        self.table_object_id = args.table_object_id
        self.c = {'velocity_scale': args.scale, 'acceleration_scale': args.scale}
        self.latest, self.latest_stamp, self.received = {}, None, 0.
        self.scene_client = node.create_client(GetPlanningScene, 'get_planning_scene')
        if not self.scene_client.wait_for_service(timeout_sec=10):
            raise RuntimeError('MoveIt get_planning_scene unavailable.')
        self.ensure_table(args.config)
        self.subscription = node.create_subscription(
            JointState, 'joint_states', self.on_joints, qos_profile_sensor_data)
        self.preview = node.create_publisher(DisplayTrajectory, 'display_planned_path', 10)
        self.vertical = ScenePlanner(self)
        self.executor = None
        if self.execute:
            self.executor = ActionClient(node, ExecuteTrajectory, 'execute_trajectory')
            if not self.executor.wait_for_server(timeout_sec=10):
                raise RuntimeError('MoveIt execute_trajectory unavailable.')
        self.joints = self.settled_snapshot()

    def ensure_table(self, config_path):
        """Install the configured table only when its ID is absent from MoveIt."""
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        scene = self.wait(self.scene_client.call_async(req), 10).scene
        if any(o.id == self.table_object_id for o in scene.world.collision_objects):
            return
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            raise ValueError('Jog configuration must be a YAML mapping.')
        for key in ('table_collision_center', 'table_collision_size'):
            value = config.get(key)
            if (not isinstance(value, list) or len(value) != 3
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in value)):
                raise ValueError(f'{key} must contain three finite values in metres.')
        if min(config['table_collision_size']) <= 0:
            raise ValueError('Table dimensions must be positive.')
        diff = PlanningScene(is_diff=True)
        diff.robot_state.is_diff = True
        diff.world.collision_objects = [box(self.table_object_id,
                                           config['table_collision_size'],
                                           config['table_collision_center'], BASE)]
        client = self.node.create_client(ApplyPlanningScene, 'apply_planning_scene')
        if not client.wait_for_service(timeout_sec=10):
            raise RuntimeError('MoveIt apply_planning_scene unavailable.')
        result = self.wait(client.call_async(ApplyPlanningScene.Request(scene=diff)), 10)
        if result is None or not result.success:
            raise RuntimeError('Could not add configured table to MoveIt.')
        self.node.get_logger().info(
            f'Added table {self.table_object_id!r} from {config_path}; inspect it in RViz before jogging.')

    def measured_pose(self):
        self.joints = self.settled_snapshot()
        with self.vertical.scoped_scene() as original:
            return self.vertical.fk(self.vertical.state(original, False, self.joints))

    def jog(self, step):
        pose = self.measured_pose()
        target, _ = coordinates(pose)
        target[2] += step
        trajectory, approved = self.vertical.plan(target)
        self.preview.publish(DisplayTrajectory(trajectory_start=approved, trajectory=[trajectory]))
        mode = ('physical arm' if self.hardware else 'simulated arm') if self.execute else 'preview only; no motion'
        confirm(f'Base Z {step * 1000:+.1f} mm ({mode}). Review the path in RViz.', self.execute)
        if not self.execute:
            print('Preview complete. Next command starts from measured joints; no virtual advance.')
            return
        self.vertical.revalidate(trajectory, approved, target[2])
        live = self.settled_snapshot()
        check_start(live, trajectory, tolerance=0.001)
        self.vertical.check_execution_start(live, approved)
        result = self.action(self.executor, ExecuteTrajectory.Goal(trajectory=trajectory))
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'Execution failed ({result.error_code.val}); inspect robot.')
        actual = self.measured_pose()
        xyz, q = coordinates(actual)
        initial, initial_q = coordinates(pose)
        angle = math.degrees(2 * math.acos(min(1., abs(sum(a*b for a, b in zip(q, initial_q))))))
        delta = [a-b for a, b in zip(xyz, initial)]
        lateral = math.hypot(delta[0], delta[1])
        z_error = xyz[2] - target[2]
        jt = trajectory.joint_trajectory
        joint_error = max(abs(self.joints[n] - value)
                          for n, value in zip(jt.joint_names, jt.points[-1].positions))
        report = (f'Requested dZ={step*1000:+.3f} mm; '
                  f'measured dXYZ=[{delta[0]*1000:+.3f}, {delta[1]*1000:+.3f}, {delta[2]*1000:+.3f}] mm; '
                  f'lateral error={lateral*1000:.3f} mm; Z error={z_error*1000:+.3f} mm; '
                  f'orientation change={angle:.3f} deg; max final joint error={joint_error:.6f} rad. '
                  f'Endpoint limits: lateral={self.vertical.limits.line_tolerance_m*1000:.3f} mm, '
                  f'Z={self.endpoint_z_tolerance_m*1000:.3f} mm, '
                  f'orientation={self.vertical.limits.orientation_tolerance_deg:.3f} deg.')
        self.node.get_logger().info(report)
        print('Measured TCP:', xyz, 'quaternion xyzw:', q)
        if (lateral > self.vertical.limits.line_tolerance_m
                or abs(z_error) > self.endpoint_z_tolerance_m
                or angle > self.vertical.limits.orientation_tolerance_deg):
            raise RuntimeError('Execution reported success but measured endpoint is outside tolerance. '
                               + report)
        if abs(z_error) > self.vertical.limits.line_tolerance_m:
            self.node.get_logger().warning(
                'Vertical endpoint error exceeds 1 mm but is accepted for teaching. '
                'Saved poses use measured joints/TCP; controller tracking still needs investigation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Execute confirmed steps on the connected real/fake stack')
    parser.add_argument('--scale', type=float, default=0.02, help='Joint velocity/acceleration scaling')
    parser.add_argument('--endpoint-z-tolerance', type=float, default=0.002,
                        help='Measured final Z error allowed for teaching, in metres (default: 0.002)')
    parser.add_argument('--output', type=Path, default=Path('taught_tcp_poses.json'))
    parser.add_argument('--config', type=Path,
                        default=Path(get_package_share_directory('fr3_pick_place')) / 'config/cartesian_jog.yaml',
                        help='Table geometry to add when the table object is absent from MoveIt')
    parser.add_argument('--table-object-id', default='fr3_demo_table')
    args, ros_args = parser.parse_known_args()
    if not math.isfinite(args.scale) or not 0 < args.scale <= 0.05:
        parser.error('--scale must be in (0, 0.05].')
    if not math.isfinite(args.endpoint_z_tolerance) or not 0 < args.endpoint_z_tolerance <= 0.002:
        parser.error('--endpoint-z-tolerance must be in (0, 0.002] metres.')
    if not sys.stdin.isatty():
        parser.error('An interactive terminal is required.')
    rclpy.init(args=ros_args)
    node = rclpy.create_node('fr3_cartesian_jog')
    try:
        task = JogTask(node, args)
        print(('HARDWARE EXECUTION' if task.hardware else 'SIMULATED EXECUTION: mock arm verified')
              if args.execute else 'PLANNING ONLY: no motion commands')
        if not args.execute:
            print('To execute steps, restart with: ros2 run fr3_pick_place cartesian_jog.py --execute')
        print('Empty-hand teaching. Commands: z -0.005 | z 0.005 | pose | save pick | save place | q')
        print('Distances are metres in fr3_link0. Orientation is held at its current value.')
        print('No camera collision model is required. Existing scene attachments remain collision-checked.')
        print('Do not move with RViz or another program while a jog is pending.')
        while True:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            words = input('jog> ').strip().split()
            if words == ['q']:
                break
            try:
                if words == ['pose']:
                    print('TCP xyz / quaternion xyzw:', coordinates(task.measured_pose()))
                elif len(words) == 2 and words[0] == 'save':
                    pose = task.measured_pose()
                    record_pose(args.output, words[1], pose, task.joints)
                    print(f'Saved measured TCP pose {words[1]!r} to {args.output.resolve()}')
                elif len(words) == 2 and words[0] == 'z':
                    step = parse_step(words[1])
                    task.jog(step)
                else:
                    print('Use: z SIGNED_METRES | pose | save NAME | q')
            except ValueError as exc:
                print(exc)
    except (OperatorStop, EOFError, KeyboardInterrupt):
        node.get_logger().warning('Stopped. No automatic retreat or gripper action.')
    except Exception as exc:
        node.get_logger().error(f'Aborted: {exc}. Inspect robot before restarting.')
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
