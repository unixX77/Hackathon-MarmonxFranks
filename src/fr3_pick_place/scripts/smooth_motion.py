"""Straight-line, joint-continuous arm motion and hand control for the Lego pick and place.

Why not MoveIt's usual pose-goal planning (OMPL)? It samples random paths: the arm may swing
joints through hundreds of degrees to reach a point 10 cm away. Here every move is a straight
Cartesian line (MoveIt's compute_cartesian_path service, collision checked), the hand yaw is
rotated the short way, and a smooth speed profile is applied before the trajectory is executed.
A move that cannot be done as a clean straight line is refused instead of improvised.

The pure helpers (tool_down_quaternion, nearest_yaw, retime) need no ROS and are unit tested.
"""
import math
import time

import numpy as np

# FR3 joint speed limits (rad/s). Moves run at a small fraction of these.
VMAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
RAMP_SECONDS = 1.5          # time to reach cruise speed, and to stop again
MAX_JOINT_SWING = 2.0       # refuse any single move that turns one of joints 1-6 by more than this (rad)
MAX_WRIST_SWING = 2.0       # same for joint 7, the wrist: a 90 degree hand yaw is ~1.6 rad by itself
MIN_FRACTION = 0.999        # the whole straight line must be reachable and collision free
WRIST_SOFT_LIMIT = 2.4      # rad: joint 7 may travel to +-3.05. Past this, try the other equivalent hand yaw
UNWIND_WRIST_SWING = 3.4    # rad: that other yaw is 180 degrees (3.14 rad) away, so allow that much wrist turn


def swing_violation(swing, arm_limit, wrist_limit):
    """(joint number, swing, limit) of the first joint that turns too far in one move, else None.
    Joints 1-6 use arm_limit; joint 7 (wrist rotation, harmless in free space) uses wrist_limit."""
    for index, value in enumerate(swing):
        limit = wrist_limit if index == 6 else arm_limit
        if value > limit:
            return index + 1, float(value), float(limit)
    return None


def wrist_excursion(positions):
    """Largest |joint 7| along a joint path (rad): how close the wrist gets to its +-3.05 stop."""
    return float(np.max(np.abs(np.asarray(positions, dtype=float)[:, 6])))


class StepAbort(Exception):
    """Raised when the operator stops the run at a step prompt."""


def tool_down_quaternion(yaw):
    """xyzw quaternion of Rz(yaw) * Rx(pi): tool Z points down, jaws along base Y at yaw 0."""
    return (math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0)


def hand_yaw(rotation):
    """Yaw of a tool-down hand rotation matrix (inverse of tool_down_quaternion)."""
    return math.atan2(rotation[1, 0], rotation[0, 0])


def nearest_yaw(axis_yaw, current_yaw):
    """The jaws are symmetric, so axis_yaw and axis_yaw +- pi are the same grasp. Take the one
    closest to the current hand yaw: the wrist then never turns more than 90 degrees."""
    k = round((current_yaw - axis_yaw) / math.pi)
    return axis_yaw + k * math.pi


def retime(positions, scale, ramp=RAMP_SECONDS):
    """Time stamps and velocities for a dense joint path: trapezoidal speed profile whose cruise
    speed is `scale` times the joint limits. Returns (times, velocities)."""
    q = np.asarray(positions, dtype=float)
    if len(q) < 2:
        return np.zeros(len(q)), np.zeros_like(q)
    step = np.max(np.abs(np.diff(q, axis=0)) / (VMAX * scale), axis=1)   # seconds at cruise speed
    u = np.concatenate([[0.0], np.cumsum(step)])
    total = u[-1]
    if total < 1e-9:
        return np.zeros(len(q)), np.zeros_like(q)
    if total >= ramp:
        end = total + ramp
        t = np.where(u <= ramp / 2, np.sqrt(2 * u * ramp),
                     np.where(u >= total - ramp / 2, end - np.sqrt(np.maximum(2 * (total - u) * ramp, 0)),
                              ramp + (u - ramp / 2)))
    else:
        end = 2 * math.sqrt(total * ramp)
        t = np.where(u <= total / 2, np.sqrt(2 * u * ramp),
                     end - np.sqrt(np.maximum(2 * (total - u) * ramp, 0)))
    t = np.maximum.accumulate(t)
    t = t + np.arange(len(t)) * 1e-4                                   # strictly increasing
    v = np.zeros_like(q)
    v[1:-1] = (q[2:] - q[:-2]) / (t[2:] - t[:-2])[:, None]
    return t, v


class Mover:
    """Executes straight-line moves and the hand actions through MoveIt / the Franka hand."""

    def __init__(self, node, c, tf_buffer, block_in_tcp):
        from franka_msgs.action import Grasp, Move
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
        from rclpy.action import ActionClient
        self.node, self.c, self.tf, self.block_in_tcp = node, c, tf_buffer, block_in_tcp
        self.scale = float(c['velocity_scale'])
        self.step_mode, self.step_no = False, 0
        self.path_client = node.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.scene_client = node.create_client(ApplyPlanningScene, 'apply_planning_scene')
        self.executor = ActionClient(node, ExecuteTrajectory, 'execute_trajectory')
        self.move_client = ActionClient(node, Move, 'franka_gripper/move')
        self.grasp_client = ActionClient(node, Grasp, 'franka_gripper/grasp')
        if not self.path_client.wait_for_service(timeout_sec=15):
            raise RuntimeError('compute_cartesian_path unavailable: is MoveIt running?')
        if not self.scene_client.wait_for_service(timeout_sec=15):
            raise RuntimeError('apply_planning_scene unavailable: is MoveIt running?')
        if not self.executor.wait_for_server(timeout_sec=15):
            raise RuntimeError('execute_trajectory unavailable: is MoveIt running?')
        if not all(cl.wait_for_server(timeout_sec=10) for cl in (self.move_client, self.grasp_client)):
            raise RuntimeError('Franka Hand action servers unavailable')

    # ------------------------------------------------------------------ helpers
    def wait(self, future, timeout):
        import rclpy
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError('ROS operation timed out; inspect the robot before retrying')
        return future.result()

    def hand_pose(self):
        from rclpy.time import Time
        import rclpy
        from scipy.spatial.transform import Rotation
        end = time.monotonic() + 10.0
        while not self.tf.can_transform(self.c['base_frame'], self.c['ee_frame'], Time()):
            if time.monotonic() > end:
                raise RuntimeError('No hand pose from TF: is the robot (joint states) running?')
            rclpy.spin_once(self.node, timeout_sec=0.1)          # let the TF listener receive
        tf = self.tf.lookup_transform(self.c['base_frame'], self.c['ee_frame'], Time()).transform
        q, p = tf.rotation, tf.translation
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        pose[:3, 3] = [p.x, p.y, p.z]
        return pose

    # --------------------------------------------------------------- step mode
    def ask_step(self, label, detail):
        """In step mode, describe the next action and wait for the operator before doing it."""
        if not self.step_mode:
            return
        self.step_no += 1
        print(f'\nSTEP {self.step_no}: {label}\n  {detail}', flush=True)
        while True:
            try:
                answer = input('  Enter = do it | a = run the rest automatically | q = stop here: ').strip().lower()
            except EOFError:
                raise StepAbort('step mode needs a terminal to answer in') from None
            if answer == '':
                return
            if answer == 'a':
                self.step_mode = False
                print('  Continuing automatically.', flush=True)
                return
            if answer == 'q':
                raise StepAbort('stopped at your request')

    # ------------------------------------------------------------------- motion
    def plan_line(self, label, xyz, target_yaw, wrist_limit):
        """Collision-checked straight line to xyz with the jaws at target_yaw. Returns None when the
        hand is already there, else (trajectory message, joint path q, joint swing)."""
        from geometry_msgs.msg import Pose
        from moveit_msgs.srv import GetCartesianPath
        target = Pose()
        target.position.x, target.position.y, target.position.z = map(float, xyz)
        (target.orientation.x, target.orientation.y,
         target.orientation.z, target.orientation.w) = tool_down_quaternion(target_yaw)
        request = GetCartesianPath.Request()
        request.header.frame_id = self.c['base_frame']
        request.start_state.is_diff = True
        request.group_name = 'fr3_arm'
        request.link_name = self.c['ee_frame']
        request.waypoints = [target]
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.revolute_jump_threshold = 0.3        # an IK flip between steps is a jump: refuse
        request.avoid_collisions = True
        response = self.wait(self.path_client.call_async(request), 20)
        points = response.solution.joint_trajectory.points
        if response.fraction >= MIN_FRACTION and len(points) < 2:
            return None
        if response.fraction < MIN_FRACTION or len(points) < 2:
            raise RuntimeError(f'{label}: straight line not possible (reached {response.fraction * 100:.0f} %, '
                               f'MoveIt code {response.error_code.val}); stopping instead of improvising')
        q = np.array([p.positions for p in points])
        swing = np.abs(q.max(axis=0) - q.min(axis=0))
        too_far = swing_violation(swing, MAX_JOINT_SWING, wrist_limit)
        if too_far:
            raise RuntimeError(f'{label}: would swing joint {too_far[0]} by {too_far[1]:.2f} rad '
                               f'(limit {too_far[2]:.2f}); refusing')
        return response.solution, q, swing

    def move(self, label, xyz, yaw=None):
        """Straight line to xyz with the tool pointing down. yaw is the jaw-axis angle (mod pi);
        None keeps the current hand yaw.

        The jaws are symmetric, so yaw and yaw +- pi are the same grasp. The one closest to the current
        hand yaw is tried first (least wrist turn). If that is refused, or would take the wrist (joint 7)
        past WRIST_SOFT_LIMIT, the other one is tried and the plan that keeps the wrist further from its
        stop wins: otherwise a long series of picks slowly winds the wrist up against its limit."""
        current = self.hand_pose()
        now_yaw = hand_yaw(current[:3, :3])
        first = now_yaw if yaw is None else nearest_yaw(yaw, now_yaw)
        other = first + math.pi if first <= now_yaw else first - math.pi
        plans, failure = [], None
        for target_yaw, wrist_limit in ((first, MAX_WRIST_SWING), (other, UNWIND_WRIST_SWING)):
            if plans and wrist_excursion(plans[0][2][1]) <= WRIST_SOFT_LIMIT:
                break                                   # the nearest yaw is fine: no need to look further
            if yaw is None and target_yaw != first:
                break                                   # keeping the current yaw: no alternative
            try:
                plan = self.plan_line(label, xyz, target_yaw, wrist_limit)
            except RuntimeError as problem:
                failure = failure or problem
                continue
            if plan is None:
                self.node.get_logger().info(f'{label}: already there')
                return
            plans.append((target_yaw, wrist_limit, plan))
        if not plans:
            raise failure
        target_yaw, _, (trajectory, q, swing) = min(plans, key=lambda item: wrist_excursion(item[2][1]))
        if target_yaw != first:
            self.node.get_logger().info(f'{label}: the nearest hand yaw was refused or leaves the wrist near its '
                                        f'stop: using the equivalent yaw {math.degrees(target_yaw):.0f} deg instead')
        points = trajectory.joint_trajectory.points
        times, velocities = retime(q, self.scale)
        trajectory.multi_dof_joint_trajectory.points = []
        for point, t, v in zip(points, times, velocities):
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1.0) * 1e9)
            point.velocities = [float(x) for x in v]
            point.accelerations = []
        distance = float(np.linalg.norm(np.array(xyz) - current[:3, 3]))
        summary = (f'{distance * 100:.0f} cm straight line to x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f}, '
                   f'hand yaw {math.degrees(now_yaw):.0f} -> {math.degrees(target_yaw):.0f} deg, '
                   f'biggest joint turn {swing.max():.2f} rad, wrist reaches {wrist_excursion(q):.2f} rad, '
                   f'about {times[-1]:.0f} s at {self.scale * 100:.0f} % speed')
        self.node.get_logger().info(f'{label}: {summary}')
        self.ask_step(label, 'Planned, not moving yet: ' + summary)
        self.execute(label, trajectory)

    def execute(self, label, trajectory):
        from action_msgs.msg import GoalStatus
        from moveit_msgs.action import ExecuteTrajectory
        handle = self.wait(self.executor.send_goal_async(ExecuteTrajectory.Goal(trajectory=trajectory)), 10)
        if not handle.accepted:
            raise RuntimeError(f'{label}: MoveIt rejected the trajectory')
        try:
            result = self.wait(handle.get_result_async(), 180)
        except (KeyboardInterrupt, TimeoutError):
            self.wait(handle.cancel_goal_async(), 5)
            raise
        if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code.val != 1:
            raise RuntimeError(f'{label}: execution failed (status {result.status}, '
                               f'code {result.result.error_code.val})')

    # -------------------------------------------------------------------- hand
    def hand(self, close=False):
        from action_msgs.msg import GoalStatus
        from franka_msgs.action import Grasp, Move
        width = self.c['grasp_width'] if close else self.c['open_width']
        self.ask_step('Close the hand on the brick' if close else 'Open the hand',
                      f'Hand to {width * 1000:.0f} mm' + (f' with {self.c["grasp_force"]:.0f} N' if close else ''))
        goal = Grasp.Goal() if close else Move.Goal()
        goal.width, goal.speed = float(width), float(self.c['gripper_speed'])
        if close:
            goal.force = float(self.c['grasp_force'])
            goal.epsilon.inner = goal.epsilon.outer = 0.003
        client = self.grasp_client if close else self.move_client
        handle = self.wait(client.send_goal_async(goal), 10)
        if not handle.accepted:
            raise RuntimeError('Hand action rejected')
        result = self.wait(handle.get_result_async(), 60)
        if result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.success:
            raise RuntimeError('Grasp failed: the hand did not close on the brick' if close
                               else 'Hand failed to open')

    # ------------------------------------------------------------- scene state
    def apply_scene(self, scene):
        from moveit_msgs.srv import ApplyPlanningScene
        self.wait(self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene)), 10)

    def prepare_scene(self):
        """Table in the scene, nothing left over from earlier runs."""
        from moveit_msgs.msg import CollisionObject
        from spawn_scene import BRICK_ID, TABLE_ID, apply_scene, base_pad_objects, table_object
        apply_scene(self.node, add=[table_object(self.c)] + base_pad_objects(self.c), remove=(BRICK_ID,),
                    strict=False)
        self.detach_block()
        apply_scene(self.node, remove=('fr3_demo_block',), strict=False)

    def attach_block(self):
        """Make the carried brick part of the hand for collision checking (a hair smaller than
        the real one, so it never counts as touching the table)."""
        from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
        from spawn_scene import make_box
        size = list(self.c['block_size'])
        size[2] -= 0.002
        obj = make_box('fr3_demo_block', size, self.block_in_tcp, frame=self.c['ee_frame'])
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [AttachedCollisionObject(
            link_name=self.c['ee_frame'], object=obj,
            touch_links=['fr3_leftfinger', 'fr3_rightfinger', 'fr3_hand'])]
        self.apply_scene(scene)

    def detach_block(self):
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [AttachedCollisionObject(
            link_name=self.c['ee_frame'],
            object=CollisionObject(id='fr3_demo_block', operation=CollisionObject.REMOVE))]
        self.apply_scene(scene)
