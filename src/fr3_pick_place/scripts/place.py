"""Reusable placement workflow sharing Picker planning, monitoring and logging.

Use through pick_place_final.py: a standalone process cannot infer a held brick.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import math

from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from scipy.spatial.transform import Rotation

from pick import Picker, Plan
from pick_core import DescentStopped, Intervention, PickFailure, PickWorkflow, angle_deg
from pick_place import Task, block_object, rotate_vector
from pick_place_cartesian import BASE, TCP, PlanningInfrastructureError, coordinates
from pick_place_stepwise import OperatorStop, confirm, finite_vector


@dataclass(frozen=True)
class PlaceTarget:
    center: list
    release_clearance: float = .002
    approach_clearance: float = .1

    @classmethod
    def from_config(cls, data, task):
        if not isinstance(data, dict) or data.get('frame_id') != BASE:
            raise ValueError(f'Placement requires frame_id: {BASE}')
        allowed = {'frame_id', 'block_center_xyz_m', 'release_clearance_m', 'approach_clearance_m'}
        if set(data)-allowed:
            raise ValueError(f'Unknown placement fields: {sorted(set(data)-allowed)}')
        center = finite_vector(data['block_center_xyz_m'], 3, 'block_center_xyz_m')
        release = data.get('release_clearance_m', .002)
        clearance = data.get('approach_clearance_m', .1)
        for key, value, low, high in [('release_clearance_m', release, 0., .005),
                                      ('approach_clearance_m', clearance, .01, .3)]:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f'{key} must be in [{low},{high}] metres')
        top = task['table_center'][2]+task['table_size'][2]/2
        bottom = center[2]-task['block_size'][2]/2
        if abs(bottom-top) > .001:
            raise ValueError('This placement workflow requires the desired brick bottom on the configured tabletop (within 1 mm)')
        return cls(center, release, clearance)


def object_pose(tcp_xyz, tcp_q, attachment):
    offset = rotate_vector(tcp_q, attachment['xyz_m'])
    q = (Rotation.from_quat(tcp_q)*Rotation.from_quat(attachment['quaternion_xyzw'])).as_quat()
    return dict(xyz_m=[a+b for a, b in zip(tcp_xyz, offset)], quaternion_xyzw=q.tolist())


def release_tcp(target, attachment):
    offset = rotate_vector([1., 0., 0., 0.], attachment['xyz_m'])
    center = list(target.center)
    center[2] += target.release_clearance
    return [a-b for a, b in zip(center, offset)]


class PlacementWorkflow:
    """Never release unless the requested release pose is verified."""
    def __init__(self, adapter, policy):
        self.a, self.p = adapter, policy

    def run(self):
        self.a.require_held()
        self.a.status('LIFTING', 'Lifting the held brick clear of the pickup location')
        self.a.lift_held()
        target = self.a.place_above()
        for attempt in range(self.p.descent_retries+1):
            self.a.status('FINDING_PLACE_APPROACH', 'Checking transfer, placement descent and empty-hand retreat')
            self.a.move_arm(self.a.find_place_approach(target), 'Transfer above placement')
            self.a.align_place(target)
            try:
                plan = self.a.plan_place_descent()
            except Intervention:
                raise
            except PickFailure:
                if attempt == self.p.descent_retries:
                    raise
                self.a.reject_place_approach()
                continue
            try:
                self.a.status('PLACING', 'Lowering the held brick to the release pose')
                self.a.move_arm(plan, 'Placement descent', vertical=True)
                if not self.a.verify(self.a.place_release())['accepted']:
                    raise DescentStopped('Placement endpoint is outside tolerance')
            except DescentStopped as exc:
                self.a.status('RECOVERING_PLACE', f'Recovering while retaining the grasp: {exc}')
                self.a.recover_place(target)
                if attempt == self.p.descent_retries:
                    raise PickFailure('PLACEMENT_FAILED: retreated with brick held; retry budget exhausted') from exc
                self.a.reject_place_approach()
                continue
            self.a.status('RELEASING', 'Release pose verified; opening the gripper')
            self.a.release_held()
            self.a.status('RETREATING', 'Retreating above the released brick')
            self.a.retreat_released(target[2])
            state = 'PLACED' if self.a.hardware else 'PREVIEW_COMPLETE'
            self.a.status(state, 'Placement and retreat complete' if self.a.hardware else
                          'Pick, lift, transfer, release and retreat preview complete; no hardware commands sent')
            return state
        raise PickFailure('No placement descent completed')


class PlacePicker(Picker):
    """Picker extension; all picking algorithms stay inherited from pick.py."""
    def __init__(self, *args, **kwargs):
        self.phase = 'empty'
        self.place_target = None
        self.failed_place_approaches = []
        self.selected_place_approach = None
        super().__init__(*args, **kwargs)

    def scene_mode(self):
        phase = getattr(self, 'phase', 'empty')
        return phase in ('held', 'release_uncertain'), phase == 'released'

    def confirm_empty_context(self):
        if self.phase in ('held', 'release_uncertain') and self.hardware:
            req = GetPlanningScene.Request()
            req.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                                          | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
            scene = self.vertical.call(self.scene_client, req).scene
            if any(a.object.id == 'fr3_demo_block' for a in scene.robot_state.attached_collision_objects):
                detach = PlanningScene(is_diff=True)
                detach.robot_state.is_diff = True
                removed = AttachedCollisionObject(link_name=TCP)
                removed.object.id, removed.object.operation = 'fr3_demo_block', CollisionObject.REMOVE
                detach.robot_state.attached_collision_objects = [removed]
                self.apply_object_scene(detach)
            if any(o.id == 'fr3_demo_block' for o in scene.world.collision_objects):
                remove = PlanningScene(is_diff=True)
                remove.robot_state.is_diff = True
                remove.world.collision_objects = [CollisionObject(id='fr3_demo_block', operation=CollisionObject.REMOVE)]
                self.apply_object_scene(remove)
        self.phase = 'empty'
        self.c.pop('attached_block_pose', None)
        self.c.pop('carrying_orientation_xyzw', None)

    def require_held(self):
        if (self.phase != 'held' or 'attached_block_pose' not in self.c
                or self.blocked or (self.hardware and not self.holding)):
            raise Intervention('No verified held-object context; placement inhibited')

    def check_pick_candidate(self, approach, descent):
        """Choose a pick posture that also permits lift after closing fingers."""
        self.status('CHECKING_LIFT', 'Checking post-grasp lift feasibility for this pick candidate')
        saved_joints, saved_phase = dict(self.joints), self.phase
        keys = ('attached_block_pose', 'carrying_orientation_xyzw')
        saved = {k: copy.deepcopy(self.c[k]) for k in keys if k in self.c}
        try:
            jt = descent.trajectory.joint_trajectory
            self.joints.update(zip(jt.joint_names, jt.points[-1].positions))
            for finger in (1, 2):
                self.joints[f'fr3_finger_joint{finger}'] = self.c['grasp_width']/2
            self.capture_grasp_transform(self.joints)
            self.c['carrying_orientation_xyzw'] = [1., 0., 0., 0.]
            self.phase = 'held'
            _, pose = self.state_pose(self.joints)
            xyz, _ = coordinates(pose)
            xyz[2] += self.c['clearance']
            trajectory, state = self.vertical.plan(xyz, carrying=True)
            score, _ = self.audit(Plan(trajectory, state, xyz, True, True), collision=False)
            return score
        finally:
            self.joints, self.phase = saved_joints, saved_phase
            for key in keys:
                self.c.pop(key, None)
            self.c.update(saved)

    def apply_object_scene(self, scene):
        if self.hardware:
            response = self.vertical.call(self.vertical.apply, ApplyPlanningScene.Request(scene=scene))
            if not response.success:
                raise Intervention('Held/released object scene update failed; inspect scene before moving')

    def capture_held(self):
        self.joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        self.capture_grasp_transform(self.joints)
        # Fixed requested orientation for subsequent moves. Actual captured grip
        # is retained in attached_block_pose, including its relative orientation.
        self.c['carrying_orientation_xyzw'] = [1., 0., 0., 0.]
        self.phase = 'held'
        self.failed_approaches = []  # pick rejections do not blacklist placement postures
        scene = self.held_scene()
        self.apply_object_scene(scene)
        self.log_event('held_object_captured', attachment=self.c['attached_block_pose'], hardware=self.hardware)

    def held_scene(self):
        attached = AttachedCollisionObject(link_name=TCP, touch_links=['fr3_leftfinger', 'fr3_rightfinger'])
        attached.object = block_object(self.c, carrying=True)
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        # MoveIt transfers the same-ID world object on attachment. No redundant
        # world REMOVE is sent in this scene diff.
        return scene

    def released_scene(self):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        removed = AttachedCollisionObject(link_name=TCP)
        removed.object.id, removed.object.operation = 'fr3_demo_block', CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects = [removed]
        scene.world.collision_objects = [block_object(self.c, placed=True)]
        return scene

    @contextmanager
    def preview_release_scene(self):
        """Temporarily model a future release while the real hand stays closed."""
        self.apply_object_scene(self.released_scene())
        try:
            self.phase = 'released'
            yield
        finally:
            self.phase = 'held'
            # Remove the virtual world object BEFORE attaching. A same-diff
            # redundant REMOVE after attachment can fail in MoveIt.
            remove = PlanningScene(is_diff=True)
            remove.robot_state.is_diff = True
            remove.world.collision_objects = [CollisionObject(id='fr3_demo_block', operation=CollisionObject.REMOVE)]
            self.apply_object_scene(remove)
            self.apply_object_scene(self.held_scene())

    def place_release(self):
        self.require_held()
        return release_tcp(self.place_target, self.c['attached_block_pose'])

    def place_above(self):
        xyz = self.place_release()
        xyz[2] += self.place_target.approach_clearance
        return xyz

    def vertical_plan(self, target, carrying, placed=False):
        self.ensure_running()
        self.joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        try:
            trajectory, state = self.vertical.plan(target, carrying=carrying, placed=placed)
            plan = Plan(trajectory, state, list(target), True, carrying, placed)
            self.audit(plan, collision=False)
            return plan
        except (PlanningInfrastructureError, TimeoutError) as exc:
            raise Intervention(f'Placement planning infrastructure failed: {exc}') from exc
        except Intervention:
            raise
        except RuntimeError as exc:
            raise PickFailure(f'Vertical placement path failed: {exc}') from exc

    def lift_held(self):
        self.require_held()
        self.joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        _, pose = self.state_pose(self.joints)
        target, _ = coordinates(pose)
        target[2] += self.c['clearance']
        try:
            plan = self.vertical_plan(target, carrying=True)
            self.move_arm(plan, 'Lift held brick', vertical=True)
            if not self.verify(target)['accepted']:
                raise Intervention('Lift endpoint failed verification; brick remains held')
        except DescentStopped as exc:
            raise Intervention('Lift stopped; inspect held brick before any recovery') from exc

    def reject_place_approach(self):
        if self.selected_place_approach is not None:
            self.failed_place_approaches.append(dict(self.selected_place_approach))

    def find_place_approach(self, target):
        self.require_held()
        start = self.settled_snapshot() if self.hardware else dict(self.joints)
        candidates = [s for s in self.ik_candidates(target, start) if not any(
            max(abs(s[n]-old[n]) for n in self.vertical.bounds) < .05
            for old in self.failed_place_approaches)]
        evaluated = []
        saved_placed = copy.deepcopy(self.c.get('placed_block_pose'))
        try:
            for index, solution in enumerate(candidates):
                self.ensure_running()
                self.status('CHECKING_PLACE_CANDIDATE', f'Checking placement candidate {index+1}/{len(candidates)}')
                self.joints, self.commands, self.phase = dict(start), [], 'held'
                self.forced_solution = solution
                self.execute = False
                try:
                    Task.motion(self, 'Candidate placement transfer', target, carrying=True)
                    transfer = Plan(copy.deepcopy(self.commands[-1][2]), copy.deepcopy(self.preview_start),
                                    list(target), carrying=True)
                    score, _ = self.audit(transfer)
                    trajectory, state = self.vertical.plan(self.place_release(), carrying=True)
                    descent = Plan(trajectory, state, self.place_release(), True, True)
                    descent_score, _ = self.audit(descent, collision=False)
                    jt = trajectory.joint_trajectory
                    self.joints.update(zip(jt.joint_names, jt.points[-1].positions))
                    _, endpoint = self.state_pose(self.joints)
                    xyz, q = coordinates(endpoint)
                    self.c['placed_block_pose'] = object_pose(xyz, q, self.c['attached_block_pose'])
                    with self.preview_release_scene():
                        for finger in (1, 2):
                            self.joints[f'fr3_finger_joint{finger}'] = self.c['open_width']/2
                        retreat_target = [xyz[0], xyz[1], target[2]]
                        retreat, retreat_state = self.vertical.plan(retreat_target, placed=True)
                        retreat_score, _ = self.audit(Plan(retreat, retreat_state, retreat_target, True,
                                                          placed=True), collision=False)
                    evaluated.append((score+descent_score+retreat_score, transfer))
                except (PlanningInfrastructureError, TimeoutError) as exc:
                    raise Intervention(f'Placement planning infrastructure failed: {exc}') from exc
                except Intervention:
                    raise
                except RuntimeError as exc:
                    self.log_event('place_candidate_rejected', index=index, reason=str(exc))
                    self.node.get_logger().warning(f'Placement candidate rejected: {exc}')
                finally:
                    self.phase = 'held'
        finally:
            self.forced_solution = None
            self.joints, self.commands, self.phase = start, [], 'held'
            if saved_placed is None:
                self.c.pop('placed_block_pose', None)
            else:
                self.c['placed_block_pose'] = saved_placed
        if not evaluated:
            raise PickFailure('NO_VALID_PLACE_PLAN: no transfer with feasible descent and retreat; brick remains held')
        score, plan = min(evaluated, key=lambda item: item[0])
        jt = plan.trajectory.joint_trajectory
        self.selected_place_approach = dict(zip(jt.joint_names, jt.points[-1].positions))
        self.log_event('place_approach_selected', score=score, target=target)
        return plan

    def align_place(self, target):
        previous = float('inf')
        for attempt in range(self.policy.corrections+1):
            error = self.verify(target)
            if error['accepted']:
                return
            score = error['xyz_m']/self.policy.correction_xyz_m + error['angle_deg']/self.policy.correction_angle_deg
            if (attempt == self.policy.corrections or score >= previous-1e-3
                    or error['xyz_m'] > self.policy.correction_xyz_m
                    or error['angle_deg'] > self.policy.correction_angle_deg or error['joint_rad'] > .01):
                raise PickFailure('PLACE_ALIGNMENT_FAILED: retaining brick; no release')
            previous = score
            self.status('CORRECTING_PLACE_APPROACH', f'Correcting placement approach {attempt+1}/{self.policy.corrections}')
            self.move_arm(self.find_place_approach(target), 'Placement approach correction')

    def plan_place_descent(self):
        self.require_held()
        self.joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        _, pose = self.state_pose(self.joints)
        xyz, q = coordinates(pose)
        target = self.place_release()
        if (math.hypot(xyz[0]-target[0], xyz[1]-target[1]) > self.c['endpoint_lateral_tolerance_m']
                or angle_deg(q, [1., 0., 0., 0.]) > self.c['endpoint_orientation_tolerance_deg']):
            raise PickFailure('Placement descent inhibited: TCP is not aligned')
        return self.vertical_plan(target, carrying=True)

    def recover_place(self, above):
        self.require_held()
        self.ensure_running()
        self.robot_ok()
        self.joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        _, pose = self.state_pose(self.joints)
        xyz, q = coordinates(pose)
        if (xyz[2] > above[2]+.001 or math.hypot(xyz[0]-above[0], xyz[1]-above[1]) > self.policy.monitor_xy_m
                or angle_deg(q, [1., 0., 0., 0.]) > self.policy.monitor_angle_deg):
            raise Intervention('No verified held-object retreat corridor')
        if above[2]-xyz[2] > .001:
            target = [xyz[0], xyz[1], above[2]]
            try:
                self.move_arm(self.vertical_plan(target, carrying=True), 'Held-object recovery retreat', vertical=True)
                if not self.verify(target)['accepted']:
                    raise Intervention('Held-object recovery endpoint failed verification')
            except (OperatorStop, KeyboardInterrupt):
                raise
            except RuntimeError as exc:
                raise Intervention(f'Held-object retreat failed: {exc}') from exc

    def release_held(self):
        self.require_held()
        self.ensure_running()
        target = self.place_release()
        confirm('Release brick at verified placement pose', self.hardware)
        self.ensure_running()
        if not self.verify(target)['accepted']:
            raise Intervention('Placement pose changed during approval; retaining grasp')
        self.robot_ok()
        _, pose = self.state_pose(self.joints)
        xyz, q = coordinates(pose)
        released = object_pose(xyz, q, self.c['attached_block_pose'])
        self.execute = self.hardware
        try:
            self.command_hand()
            if self.hardware:
                self.joints = self.settled_snapshot()
                names = ['fr3_finger_joint1', 'fr3_finger_joint2']
                if (any(n not in self.joints for n in names)
                        or abs(sum(self.joints[n] for n in names)-self.c['open_width']) > .002):
                    raise RuntimeError('Fresh finger feedback did not confirm the open width')
        except BaseException as exc:
            self.phase, self.holding, self.blocked = 'release_uncertain', True, True
            raise Intervention('Release outcome uncertain; no automatic retreat or restart') from exc
        finally:
            self.execute = False
        self.holding, self.phase = False, 'released'
        self.c['placed_block_pose'] = released
        self.apply_object_scene(self.released_scene())
        self.log_event('object_released', pose=released, estimated=True, hardware=self.hardware)

    def retreat_released(self, height):
        if self.phase != 'released' or self.holding:
            raise Intervention('Release is not confirmed; empty-hand retreat inhibited')
        self.joints = self.settled_snapshot() if self.hardware else dict(self.joints)
        _, pose = self.state_pose(self.joints)
        xyz, _ = coordinates(pose)
        target = [xyz[0], xyz[1], height]
        try:
            self.move_arm(self.vertical_plan(target, carrying=False, placed=True), 'Empty-hand placement retreat', vertical=True)
            if not self.verify(target)['accepted']:
                raise Intervention('Empty-hand retreat endpoint failed verification')
        except (OperatorStop, KeyboardInterrupt):
            raise
        except RuntimeError as exc:
            raise Intervention(f'Brick released but retreat failed; do not repeat release: {exc}') from exc


class PickPlaceWorkflow:
    def __init__(self, adapter, policy):
        self.a, self.p = adapter, policy

    def run(self):
        PickWorkflow(self.a, self.p).run()
        self.a.status('ATTACHING_OBJECT', 'Capturing the grasp transform for lift and placement')
        self.a.capture_held()
        return PlacementWorkflow(self.a, self.p).run()
