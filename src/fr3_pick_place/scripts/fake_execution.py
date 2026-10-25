"""Explicit mock-only backend. Never manufactures native Franka health messages."""
import copy
import math
import time
import xml.etree.ElementTree as ET

from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import ListHardwareComponents
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from pick_core import ARM, Intervention
from pick_place import block_object, box


def verify_mock_description(urdf):
    controls = ET.fromstring(urdf).findall('ros2_control')
    owners = []
    for control in controls:
        names = {j.get('name') for j in control.findall('joint')}
        if names.intersection(ARM):
            plugin = control.findtext('hardware/plugin', '').strip()
            if plugin != 'mock_components/GenericSystem':
                raise Intervention(f'Fake execution refused: arm URDF plugin is {plugin!r}, not mock_components/GenericSystem')
            owners.extend(names.intersection(ARM))
    if len(owners) != 7 or set(owners) != set(ARM):
        raise Intervention('Fake execution refused: URDF must assign all seven joints once to mock hardware')


def verify_mock_components(components):
    owners = []
    for component in components:
        names = {i.name.split('/')[0] for i in component.command_interfaces}
        arm = names.intersection(ARM)
        if arm:
            if component.class_type != 'mock_components/GenericSystem' or component.state.id != 3:
                raise Intervention('Fake execution refused: active arm hardware is not GenericSystem mock hardware')
            owners.extend(arm)
    if len(owners) != 7 or set(owners) != set(ARM):
        raise Intervention('Fake execution refused: cannot identify a unique active mock owner for every arm joint')


class FakeExecution:
    def __init__(self, task):
        self.task = task
        self.controller_state = None
        self.received = 0.
        verify_mock_description(task.robot_description)
        self.components = task.node.create_client(ListHardwareComponents, 'controller_manager/list_hardware_components')
        if not self.components.wait_for_service(timeout_sec=5):
            raise Intervention('Fake execution requires controller_manager/list_hardware_components')
        self.verify_identity()
        self.subscription = task.node.create_subscription(
            JointTrajectoryControllerState, 'fr3_arm_controller/controller_state',
            self.on_controller, qos_profile_sensor_data)
        self.width_command = task.node.create_publisher(Float64MultiArray, 'franka_gripper/fake_width_command', 10)
        deadline = time.monotonic()+5
        while self.width_command.get_subscription_count() == 0 and time.monotonic() < deadline:
            task.pump()
        if not self.width_command.get_subscription_count():
            raise Intervention('Restart the fake MoveIt stack with the updated fake gripper publisher; fake_width_command has no subscriber')

    def on_controller(self, state):
        self.controller_state, self.received = state, time.monotonic()

    def verify_identity(self):
        verify_mock_description(self.task.robot_description)
        response = self.task.wait(self.components.call_async(ListHardwareComponents.Request()), 5)
        verify_mock_components(response.component)

    def check_health(self):
        state = self.controller_state
        now = self.task.node.get_clock().now().nanoseconds*1e-9
        if (state is None or time.monotonic()-self.received > .5
                or not -.05 <= now-(state.header.stamp.sec+state.header.stamp.nanosec*1e-9) <= .5):
            raise Intervention('Fake controller-state feedback is stale or missing')
        if (set(state.joint_names) != set(ARM) or len(state.error.positions) != 7
                or not all(math.isfinite(v) for v in state.error.positions)):
            raise Intervention('Fake controller joint tracking feedback is incomplete/non-finite')

    def tracking_error(self):
        self.check_health()
        return max(abs(v) for v in self.controller_state.error.positions)

    def hand(self, width, speed):
        self.verify_identity()
        self.check_health()
        self.task.ensure_running()
        command = Float64MultiArray(data=[float(width), float(speed)])
        self.width_command.publish(command)
        started = time.monotonic()
        deadline = started + .08/speed + 3.
        settled = 0
        while time.monotonic() < deadline:
            self.task.ensure_running()
            self.task.pump(.03)
            self.check_health()
            values = self.task.latest
            names = ['fr3_finger_joint1', 'fr3_finger_joint2']
            if self.task.received < started or any(n not in values for n in names):
                continue
            error = abs(sum(values[n] for n in names)-width)
            settled = settled+1 if error <= .0002 else 0
            if settled >= 3:
                self.task.joints = dict(values)
                self.task.log_event('fake_gripper', width_m=width, measured_width_m=sum(values[n] for n in names))
                return
        raise Intervention('Fake gripper did not reach its commanded width')

    def install_scene(self):
        """Initialize the single simulated brick at the supplied target."""
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        original = self.task.vertical.call(self.task.scene_client, req).scene
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        if any(o.object.id == 'fr3_demo_block' for o in original.robot_state.attached_collision_objects):
            removed = AttachedCollisionObject(link_name='fr3_hand_tcp')
            removed.object.id, removed.object.operation = 'fr3_demo_block', CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects = [removed]
        scene.allowed_collision_matrix = copy.deepcopy(self.task.acm)
        scene.world.collision_objects = [box('fr3_demo_table', self.task.c['table_size'], self.task.c['table_center']),
                                         block_object(self.task.c)]
        if not self.task.vertical.call(self.task.vertical.apply, ApplyPlanningScene.Request(scene=scene)).success:
            raise Intervention('Could not initialize simulated table/brick scene')
        self.task.log_event('fake_scene_initialized', simulated=True)
