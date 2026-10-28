#!/usr/bin/python3 -s
"""EMPTY THE WHOLE PILE: look, pick the top layer, look again, until nothing is left. One command, no listener, no publisher.

  ros2 run fr3_pose_pick sort_pile.py --config src/fr3_pose_pick/config/pose_pick.yaml

The run settings are the sort_pile section of the config (per_round 15, max_failures 10, review codex, gui false, ...); an option on
the command line wins over the config for that run.

Every round:
  1. go_to_pose.py observe: the hand to the recorded observation pose (record it once: record_pose.py observe);
  2. the camera pipeline (vision/run_pipeline_classifier.sh, one frame taken by itself) with the hand pose read from TF before and
     after (the arm must be still), exactly as brick_publisher.py;
  3. only the top layer of the picture (palletize_listener.py's depth filter);
  4. pose_pick_stacks.py with those bricks (--mode stacks, the default: good bricks onto the stack of their colour, bad ones to the bin),
     or pose_pick_palletize.py (--mode row: the good row).
A brick the camera missed is simply seen again in the next round; when a layer is gone the next picture shows the layer below.
Bricks the camera sees on a colour stack's footprint, or outside the pile area, are never picked: once the pile is low a stack would
otherwise be the highest thing in the picture. The pile area is the rectangle between two opposite corners recorded with the hand
(record_pose.py pile_corner_1, then pile_corner_2), or pile_area [[x min, x max], [y min, y max]] in the config; none: anywhere.

It STOPS:
  - DONE when --empty-scans pictures in a row (default 2: one more look to be sure) find no brick;
  - when a round has nothing it may pick (only good colours without a stack are left: they stay on the pile, said so);
  - after --max-rounds rounds (default 40);
  - after --max-failures failed rounds in a row. A failed round (the move, the pipeline, a pick, a place, a robot reflex) is
    recovered from: the robot errors are cleared and the arm controller restarted, a brick still in the hand goes to the bin,
    and the next round scans again. When the recovery itself fails it stops at once and says why.
  Ctrl+C or the e-stop at any time.

Options it does not know go to the pipeline (--review codex, --second-opinion codex). Own options:
  --config FILE  --mode stacks|row  --empty-scans N  --max-rounds N  --countdown SECONDS (before the first pick of the run, default
  0: none; later rounds none)  --pipeline FILE  --dry-run (look and print the command of each round, never pick: the arm only goes to
  observe)  --no-gui / --gui  --per-round N (bricks picked per scan)  --max-shift M  --max-failures N (failed rounds in a row
  recovered from before it stops)  --review codex|claude|none. Every default is in the sort_pile section of the config.

THE PILE MOVES: it is not fixed to the plate, and pulling a brick off its studs (10 to 25 N) or a finger closing on it can drag the
whole pile a few mm (16:03 on 2026-10-27: 9 mm, every remaining brick the same). Every pick of a round aims at the positions of that
round's scan, so after a drag the next ones miss (the 3rd brick of a round, twice). So only --per-round bricks are taken per scan, the
pile's movement is measured at every scan (the bricks that stayed, compared with the last scan), and once it moved more than --max-shift
only ONE brick is taken per scan for the rest of the run. Pressing the pile onto the plate makes it fast again.

A window (sort_gui.py) shows the picture of the last scan with every brick outlined: yellow to do, cyan the one the arm is on, green
stacked, purple in the bin, red failed, orange no stack for its colour, grey not this round. Closing it does not stop the robot.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import brick_publisher as bp
import palletize_listener as pl
import record_pose as rp
import stack_listener as sl

HERE = Path(__file__).resolve().parent
SCRIPTS = {'stacks': ('pose_pick_stacks.py', sl.command), 'row': ('pose_pick_palletize.py', pl.command)}
EMPTY_MARKERS = ('No foreground instances found',)      # the detector found nothing at all: an empty pile, not a failure
SETTINGS = ('mode', 'empty_scans', 'max_rounds', 'countdown', 'per_round', 'max_shift', 'max_failures')   # sort_pile: in the config


def say(text):
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def run(command):
    say('Running: ' + ' '.join(map(str, command)))
    return subprocess.run([str(v) for v in command]).returncode


def run_pick(command, todo, board):
    """Run the pick script, show its output, and follow it on the board: '=== Brick N of M' is the brick it is on now,
    'DONE: brick picked and placed' that one finished. Returns the exit code."""
    say('Running: ' + ' '.join(map(str, command)))
    process = subprocess.Popen([str(v) for v in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env=dict(os.environ, PYTHONUNBUFFERED='1'))
    current, done = None, 0
    for line in process.stdout:
        print(line, end='', flush=True)
        if line.startswith('=== Brick ') and ' of ' in line:
            try:
                current = todo[int(line.split()[2]) - 1]
            except (ValueError, IndexError):
                continue
            board.mark(current['id'], 'current')
            board.text(board.lines[0], f'Now: #{current["id"]} {current.get("colour", "")} '
                       + ('-> bin' if current['label'] == 'B' else f'-> {current.get("colour", "")} stack'))
        elif 'DONE: brick picked and placed' in line and current is not None:
            board.mark(current['id'], 'bin' if current['label'] == 'B' else 'done')
            current, done = None, done + 1
    code = process.wait()
    if code and current is not None:
        board.mark(current['id'], 'failed')
    return code, current, done


STUD = 0.0017                  # m: the studs on top of a brick, included in the pipeline's height above the plate


def layers_from_heights(final, c):
    """{brick id: layer} from the pipeline's own height above the plate (height_m: the table plane and the bricks measured in the same
    picture, so the 1-2 deg slope of the hand-eye conversion is not in it, unlike the base-frame height, which put a layer-2 brick at
    2.73 layers on 2026-10-27 16:49 and the fingers closed above it). {} if the result file has no heights."""
    import json
    from sort_gui import result_files
    _, path = result_files(final) if final else (None, None)
    if not path:
        return {}
    height = float(c['block_size'][2])
    try:
        return {int(b['id']): max(1, int(round((float(b['height_m']) - STUD) / height)))
                for b in json.loads(path.read_text())['bricks'] if b.get('id') and b.get('height_m') is not None}
    except (OSError, ValueError, KeyError):
        return {}


class Recovery:
    """After a failed pick: read the gripper, clear the robot's error state with the Franka action server's error_recovery (the red
    light goes green) and restart the arm's trajectory controller (without that MoveIt's moves did not go through until the bringup
    was restarted), so that the next round can go back to the observation pose and look again."""

    def __init__(self, node, c):
        from franka_msgs.action import ErrorRecovery, Move
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        self.c, self.fingers, self.stamp = c, {}, None
        self.goal_type, self.move_type = ErrorRecovery.Goal, Move.Goal
        from controller_manager_msgs.srv import SetHardwareComponentState, SwitchController
        self.client = ActionClient(node, ErrorRecovery, '/action_server/error_recovery')
        self.hand_client = ActionClient(node, Move, 'franka_gripper/move')
        self.switch_type = SwitchController.Request
        self.switch = node.create_client(SwitchController, '/controller_manager/switch_controller')
        self.hardware_type = SetHardwareComponentState.Request
        self.hardware = node.create_client(SetHardwareComponentState, '/controller_manager/set_hardware_component_state')
        self.controller = str(c.get('arm_controller', 'fr3_arm_controller'))
        self.component = str(c.get('hardware_component', 'FrankaHardwareInterface'))
        node.create_subscription(JointState, 'franka_gripper/joint_states', self._on_fingers, 10)

    def _wait(self, future, end, what):
        while not future.done() and time.monotonic() < end:
            time.sleep(0.05)                           # the node is spun by the other thread
        if not future.done():
            raise RuntimeError(f'{what} did not answer')
        return future.result()

    def open_hand(self, timeout=15.0):
        """Open the hand to open_width (a held brick drops). Raises RuntimeError when it does not open."""
        if not self.hand_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('franka_gripper/move is not available')
        goal = self.move_type()
        goal.width, goal.speed = float(self.c['open_width']), float(self.c['gripper_speed'])
        end = time.monotonic() + timeout
        handle = self._wait(self.hand_client.send_goal_async(goal), end, 'opening the hand')
        if not handle.accepted:
            raise RuntimeError('the hand refused to open')
        self._wait(handle.get_result_async(), end, 'opening the hand')
        time.sleep(0.5)
        if self.holding() is not False:
            raise RuntimeError('the hand did not open')

    def _on_fingers(self, message):
        self.fingers.update(zip(message.name, message.position))
        self.stamp = time.monotonic()

    def finger_width(self):
        """m between the fingers, or None without a recent reading."""
        if len(self.fingers) < 2 or self.stamp is None or time.monotonic() - self.stamp > 2.0:
            return None
        return float(sum(self.fingers.values()))

    def holding(self):
        """True when the fingers are neither open nor shut: a brick (or part of one) is between them. None: no reading."""
        width = self.finger_width()
        if width is None:
            return None
        return 0.005 < width < float(self.c['open_width']) - 0.002

    def clear_errors(self, timeout=20.0):
        if not self.client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('error_recovery is not available (/action_server/error_recovery)')
        future = self.client.send_goal_async(self.goal_type())
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            time.sleep(0.05)                           # the node is spun by the other thread
        if not future.done() or not future.result().accepted:
            raise RuntimeError('error_recovery did not accept the request')
        result = future.result().get_result_async()
        while not result.done() and time.monotonic() < end:
            time.sleep(0.05)
        if not result.done():
            raise RuntimeError('error_recovery did not answer')
        time.sleep(0.5)
        self.restart_controller(end)

    def restart_controller(self, end):
        """Bring the Franka hardware component back to active, then deactivate and activate the arm's trajectory controller. After a
        reflex ros2_control drops the hardware component to 'unconfigured' (its joint interfaces 'unavailable'): error_recovery turns
        the light green, but no move goes through until the bringup is restarted, and switching the controller alone is refused
        ('Not acceptable command interfaces combination', 2026-10-27 17:52)."""
        from lifecycle_msgs.msg import State
        if not self.hardware.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/controller_manager/set_hardware_component_state is not available')
        request = self.hardware_type()
        request.name = self.component
        request.target_state = State(id=State.PRIMARY_STATE_ACTIVE, label='active')
        answer = self._wait(self.hardware.call_async(request), end, f'activating {self.component}')
        if not answer.ok:
            raise RuntimeError(f'could not activate the hardware component {self.component} (it is {answer.state.label})')
        time.sleep(0.5)
        if not self.switch.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/controller_manager/switch_controller is not available')
        for deactivate in (True, False):
            request = self.switch_type()
            if deactivate:
                request.deactivate_controllers = [self.controller]
            else:
                request.activate_controllers = [self.controller]
            request.strictness = self.switch_type.STRICT
            answer = self._wait(self.switch.call_async(request), end, 'switching the arm controller')
            if not answer.ok:
                raise RuntimeError(f'could not {"deactivate" if deactivate else "activate"} {self.controller}')
            time.sleep(0.5)


class NoBoard:
    """sort_gui.Board without a window (--no-gui)."""
    lines = ['']

    def text(self, *lines):
        self.lines = list(lines)

    def scan(self, final, statuses):
        pass

    def mark(self, ident, status):
        pass

    def close(self, wait=0.0):
        pass


def look(node, buffer, c, pipeline, pipeline_args):
    """One picture: (bricks, hand pose [x y z qx qy qz qw], result folder), bricks [] for an empty pile. Raises RuntimeError."""
    import numpy as np
    before = rp.pose_record(bp.hand_samples(node, buffer, c, count=5, interval=0.05))       # also: the arm is still (1 mm)
    code, lines, final = bp.run_pipeline(pipeline, pipeline_args)
    if code:
        if any(marker in line for line in lines for marker in EMPTY_MARKERS):
            return [], None, ''
        raise RuntimeError(f'the pipeline failed (exit code {code})')
    after = rp.pose_record(bp.hand_samples(node, buffer, c, count=5, interval=0.05))
    moved = float(np.linalg.norm(np.subtract(after['xyz'], before['xyz'])))
    turned = bp.turn_deg(before['quat_xyzw'], after['quat_xyzw'])
    if moved > bp.MAX_MOVE or turned > bp.MAX_TURN_DEG:
        raise RuntimeError(f'the arm moved during the pipeline ({moved * 1000:.1f} mm, {turned:.1f} deg)')
    bricks, problems, found = bp.parse_bricks(lines)
    if not found:
        raise RuntimeError('the pipeline output has no brick list')
    for problem in problems:
        say('WARNING: ' + problem)
    return bricks, list(before['xyz']) + list(before['quat_xyzw']), final


def to_base(bricks, ee_pose, c):
    """Each brick's camera point in the base frame (x, y), as pose_pick.py turns it (hand-eye result, camera_offset_base)."""
    import numpy as np
    from camera_frame import camera_point_to_base, load_calibration, pose_matrix
    _, t_hand_cam = load_calibration(c['calibration_file'], c, c.get('camera_frame', 'camera_color_optical_frame'))
    t_base_hand = pose_matrix(ee_pose[:3], ee_pose[3:])
    shift = np.array(list(c.get('camera_offset_base') or [0.0, 0.0]) + [0.0])
    return [tuple(float(v) for v in (camera_point_to_base(t_base_hand, t_hand_cam, b['camera_xyz']) + shift)[:2]) for b in bricks]


def pile_shift(before, after):
    """How far the pile moved between two scans: the median displacement of the bricks seen in both (each one matched to the nearest
    within 12 mm), as (dx, dy, how many matched), or None when fewer than 3 match. Bricks of the pile lie a brick width apart or more,
    so a match is the same brick."""
    import numpy as np
    if not before or not after:
        return None
    a, moves = np.array(after), []
    for point in before:
        distance = np.linalg.norm(a - np.array(point), axis=1)
        if distance.min() <= 0.012:
            moves.append(a[distance.argmin()] - np.array(point))
    if len(moves) < 3:
        return None
    dx, dy = np.median(np.array(moves), axis=0)
    return float(dx), float(dy), len(moves)


def pile_area(c, config):
    """[[x min, x max], [y min, y max]] where bricks may be picked, or None (anywhere but on a stack): pile_area in the config, or the
    rectangle between the two hand positions recorded as pile_corner_1 and pile_corner_2 (record_pose.py). Not both."""
    poses = rp.load_poses(rp.poses_path(config))
    corners = [poses.get(name) for name in ('pile_corner_1', 'pile_corner_2')]
    if any(corners) and not all(corners):
        raise ValueError('only one pile corner is recorded: record both, pile_corner_1 and pile_corner_2 (opposite corners)')
    if all(corners):
        if c.get('pile_area'):
            raise ValueError('the pile area is both in the config (pile_area) and recorded (pile_corner_1 / 2): remove one of them')
        (x1, y1), (x2, y2) = [corner['xyz'][:2] for corner in corners]
        return [[min(x1, x2), max(x1, x2)], [min(y1, y2), max(y1, y2)]]
    return c.get('pile_area')


def on_the_pile(bricks, ee_pose, c, config, mode):
    """(the bricks of the pile, [(brick, why it is not), ...]). Each camera point is turned into the base frame as pose_pick.py does
    (hand-eye result, camera_offset_base); a brick on a colour stack's footprint (the camera may see the stacks too, and once the pile
    is low a stack is the highest thing in the picture) or outside pile_area (if set) is not the pile's."""
    import pose_pick as pp
    c = pp.apply_baseplate(dict(c))
    zones = []
    if mode == 'stacks':
        import pose_pick_stacks as ps
        size_x, size_y = ps.footprint(c)
        gap = ps.layout(c)[2]
        zones = [(colour, centre, size_x / 2 + gap, size_y / 2 + gap) for colour, centre in ps.stacks_from_config(c, config).items()]
    area = pile_area(c, config)
    keep, out = [], []
    for brick, (x, y) in zip(bricks, to_base(bricks, ee_pose, c)):
        zone = next((z for z in zones if abs(x - z[1][0]) <= z[2] and abs(y - z[1][1]) <= z[3]), None)
        if zone:
            out.append((brick, f'on the {zone[0]} stack'))
        elif area and not (area[0][0] <= x <= area[0][1] and area[1][0] <= y <= area[1][1]):
            out.append((brick, f'outside pile_area at x={x:.3f} y={y:.3f}'))
        else:
            keep.append(brick)
    return keep, out


def pickable(bricks, mode, c, config):
    """The bricks the pick script will really move: in stacks mode a good colour without a stack stays on the pile."""
    if mode != 'stacks':
        return bricks
    import pose_pick_stacks as ps
    stacks = ps.stacks_from_config(c, config)
    return [b for b in bricks if b['label'] == 'B' or str(b.get('colour') or '').replace(' ', '_') in stacks]


def run_settings(config):
    """The sort_pile section of the config ({} when there is none or the file cannot be read: the check below says why)."""
    import yaml
    try:
        return (yaml.safe_load(open(config, encoding='utf-8')) or {}).get('sort_pile') or {}
    except (OSError, yaml.YAMLError):
        return {}


def with_review(pipeline_args, review):
    """The pipeline options with the config's review (codex / claude) unless the command gives one; --review none: no review."""
    for i, arg in enumerate(pipeline_args):
        if arg == '--review' and i + 1 < len(pipeline_args) and pipeline_args[i + 1] == 'none':
            return pipeline_args[:i] + pipeline_args[i + 2:]
        if arg.startswith('--review'):
            return pipeline_args
    return [*pipeline_args, '--review', str(review)] if review else pipeline_args


def main(argv=None):
    from rclpy.utilities import remove_ros_args
    argv = remove_ros_args(list(sys.argv if argv is None else ['sort_pile.py', *argv]))[1:]
    if '-h' in argv or '--help' in argv:
        print(__doc__)
        return 0
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    own.add_argument('--config', default='src/fr3_pose_pick/config/pose_pick.yaml')
    own.add_argument('--mode', choices=sorted(SCRIPTS), default='stacks')
    own.add_argument('--empty-scans', type=int, default=2)
    own.add_argument('--max-rounds', type=int, default=40)
    own.add_argument('--countdown', type=int, default=0)
    own.add_argument('--pipeline', type=Path, default=bp.PIPELINE)
    own.add_argument('--dry-run', action='store_true')
    own.add_argument('--no-gui', action='store_true')
    own.add_argument('--per-round', type=int, default=2)
    own.add_argument('--max-shift', type=float, default=0.002)
    own.add_argument('--max-failures', type=int, default=3)
    own.add_argument('--gui', action='store_true')
    import yaml
    settings = run_settings(own.parse_known_args(argv)[0].config)
    own.set_defaults(**{key: settings[key] for key in SETTINGS if settings.get(key) is not None})
    if settings.get('pipeline'):
        own.set_defaults(pipeline=Path(settings['pipeline']))
    if settings.get('gui') is not None:
        own.set_defaults(no_gui=not settings['gui'])
    args, pipeline_args = own.parse_known_args(argv)
    args.no_gui = args.no_gui and not args.gui
    config = str(Path(args.config).resolve())
    if args.mode not in SCRIPTS:
        print(f'ERROR: mode {args.mode!r} in the config: stacks or row', file=sys.stderr)
        return 2
    if '--recording' in pipeline_args:
        print('ERROR: sort_pile.py takes a new picture every round: --recording cannot be used', file=sys.stderr)
        return 2
    pipeline_args = with_review(pipeline_args, settings.get('review'))
    pipeline_args = ['--auto-capture', *pipeline_args]
    print(f'Settings: --per-round {args.per_round} --max-failures {args.max_failures} --max-shift {args.max_shift} '
          f'--empty-scans {args.empty_scans} --max-rounds {args.max_rounds} --mode {args.mode} '
          f'{"--no-gui" if args.no_gui else "with the window"}; pipeline {args.pipeline} {" ".join(pipeline_args)}', flush=True)
    try:
        c = yaml.safe_load(open(config, encoding='utf-8'))
        if 'observe' not in rp.load_poses(rp.poses_path(config)):
            raise ValueError('no observation pose: hold the arm where the pictures are taken from and run  record_pose.py observe --config '
                             + args.config)
        if not args.pipeline.is_file():
            raise ValueError(f'no pipeline at {args.pipeline}: give it with --pipeline')
        area = pile_area(c, config)
        print('Pile area: ' + (f'x {area[0][0]:.3f} .. {area[0][1]:.3f}, y {area[1][0]:.3f} .. {area[1][1]:.3f}' if area
                               else 'none (anywhere but on a colour stack)'), flush=True)
        if args.mode == 'stacks':
            import pose_pick_stacks as ps
            if not ps.stacks_from_config(c, config):
                raise ValueError('no colour stacks: record_pose.py stack_<colour> over each stack centre (or colour_stacks in the config)')
    except (ValueError, OSError) as problem:
        print(f'ERROR: {problem}', file=sys.stderr)
        return 2
    script, command = SCRIPTS[args.mode]

    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = rclpy.create_node('sort_pile')
    buffer = Buffer(cache_time=Duration(seconds=30))
    listener = TransformListener(buffer, node)         # keep a reference alive
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:            # the shutdown at the end
            pass
    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    if args.no_gui:
        board = NoBoard()
    else:
        import sort_gui
        board = sort_gui.Board()
    empty, picked, round_no = 0, 0, 0
    limit, left_behind = max(1, args.per_round), []          # bricks per round; where the bricks not picked last round were
    failures, failed_at = 0, []                              # failed rounds in a row; [x, y, times] of every brick that failed
    try:
        recovery = Recovery(node, c)
    except ImportError:
        recovery = None                                      # no franka_msgs: no recovery, a failure stops the run

    def recover(why, held=None):
        """After a failure: may the run go on? Clears the robot errors; a brick still in the hand is carried to the bin and let go
        there (the bin is the one place a brick may always go; a good one is then lost to the sorting, and that is said). Raises
        RuntimeError when the run may not go on."""
        nonlocal failures, picked
        failures += 1
        if failures > args.max_failures:
            raise RuntimeError(f'{why}; {failures} failures in a row (--max-failures {args.max_failures})')
        if recovery is None:
            raise RuntimeError(f'{why}; no recovery without franka_msgs')
        holding = recovery.holding()
        if holding is None:
            raise RuntimeError(f'{why}; the gripper cannot be read, so no automatic recovery')
        say(f'RECOVERY ({failures} of {args.max_failures}): {why}. Clearing the robot errors'
            + (f', then the brick in the hand ({recovery.finger_width() * 1000:.1f} mm) to the bin' if holding else '')
            + ', then back to the observation pose to look again.')
        board.text(f'Round {round_no}: {picked} bricks moved', f'RECOVERY {failures}/{args.max_failures}: {why}')
        try:
            recovery.clear_errors()
        except RuntimeError as problem:
            raise RuntimeError(f'{why}; error recovery failed ({problem}): recover the robot by hand') from None
        if not holding:
            return
        if run([sys.executable, '-s', HERE / 'go_to_pose.py', 'bin', '--config', config]):
            raise RuntimeError(f'{why}; with a brick in the hand the arm could not go to the bin after the recovery: the brick is '
                               'still in the hand')
        try:
            recovery.open_hand()
        except RuntimeError as problem:
            raise RuntimeError(f'{why}; at the bin: {problem}') from None
        what = f'#{held["id"]} {held.get("colour", "")}' if held else 'the brick'
        if held and held['label'] == 'G':
            say(f'  {what} was GOOD: dropped in the bin by the recovery, not on its stack.')
        else:
            say(f'  {what} dropped in the bin.')
            picked += 1 if held else 0
        if held:
            board.mark(held['id'], 'bin')

    def finish(text):
        say(text)
        board.text(f'Round {round_no}: {picked} bricks moved', text)

    try:
        for round_no in range(1, args.max_rounds + 1):
            say(f'===== ROUND {round_no}: to the observation pose =====')
            board.text(f'Round {round_no}: {picked} bricks moved so far', 'Moving to the observation pose')
            if run([sys.executable, '-s', HERE / 'go_to_pose.py', 'observe', '--config', config]):
                recover('could not go to the observation pose')          # a brick in the hand here is unknown: it goes to the bin
                if run([sys.executable, '-s', HERE / 'go_to_pose.py', 'observe', '--config', config]):
                    raise RuntimeError('could not go to the observation pose, also after the recovery')
            board.text(f'Round {round_no}: {picked} bricks moved so far', 'Scanning: camera picture and detection')
            bricks, ee_pose, final = look(node, buffer, c, args.pipeline, pipeline_args)
            show = {b['id']: 'later' for b in bricks}
            if bricks:
                bricks, out = on_the_pile(bricks, ee_pose, c, config, args.mode)
                for brick, why in out:
                    say(f'  #{brick["id"]} {brick.get("colour", "")} is not on the pile ({why}): left alone')
            if not bricks:
                board.scan(final, show)
                empty += 1
                say(f'No brick in the picture ({empty} of {args.empty_scans} empty looks).')
                if empty >= args.empty_scans:
                    finish(f'DONE: the pile is empty. {picked} bricks moved in {round_no} rounds.')
                    return 0
                continue
            empty = 0
            where = to_base(bricks, ee_pose, c)
            moved = pile_shift(left_behind, where)
            if moved is not None:
                size = (moved[0] ** 2 + moved[1] ** 2) ** 0.5
                say(f'The pile moved {size * 1000:.1f} mm (dx={moved[0] * 1000:+.1f} dy={moved[1] * 1000:+.1f} mm, {moved[2]} bricks '
                    'compared) since the last scan.')
                if size > args.max_shift and limit > 1:
                    limit = 1
                    say(f'  More than {args.max_shift * 1000:.1f} mm: the picks drag the pile. From now on ONE brick per scan (slower, '
                        'but every pick aims at where the brick really is). Press the pile onto the plate to stop it moving.')
            message = {'run': final, 'ee_pose': ee_pose, 'bricks': bricks}
            kept, below = pl.top_layer_only(message, config)
            say('Depth below the top layer (camera, mm): ' + ', '.join(
                f'#{b["id"]} {mm:.1f}{"" if b in kept else " (lower layer, later)"}' for b, mm in below))
            todo = pickable(kept, args.mode, c, config)
            for brick in kept:
                show[brick['id']] = 'todo' if brick in todo else 'skipped'
                if brick not in todo:
                    say(f'  #{brick["id"]} {brick.get("colour", "")} has no stack: left on the pile')
            board.scan(final, show)
            if not todo:
                finish('STOPPED: the top layer only has bricks of colours without a stack (they stay on the pile): '
                       + ', '.join(sorted({b.get("colour") or "?" for b in kept})) + '. Add their stacks, then start again.')
                return 1
            position = {b['id']: p for b, p in zip(bricks, where)}
            for brick in list(todo):                             # a brick that failed twice is left alone
                x, y = position[brick['id']]
                if any(times >= 2 and abs(x - fx) < 0.006 and abs(y - fy) < 0.006 for fx, fy, times in failed_at):
                    todo.remove(brick)
                    show[brick['id']] = 'skipped'
                    say(f'  #{brick["id"]} {brick.get("colour", "")} failed twice here: left on the pile')
            layers = layers_from_heights(final, c)
            extra = []
            known = [layers[b['id']] for b in todo if b['id'] in layers]
            if known:
                layer = max(known)                               # this round: the top layer by the camera's own height above the plate
                for brick in [b for b in todo if layers.get(b['id'], layer) != layer]:
                    todo.remove(brick)
                    show[brick['id']] = 'later'
                extra = ['--layers', str(layer)]
                say(f'Layer {layer} (from the camera\'s height above the plate, not the base-frame height: that slopes with the '
                    'hand-eye calibration).')
            if not todo:
                finish('STOPPED: nothing left that may be picked (failed twice, or no stack for its colour).')
                return 1
            if len(todo) > limit:
                for brick in todo[limit:]:
                    show[brick['id']] = 'later'
                todo = todo[:limit]
            board.scan(final, show)
            left_behind = [p for b, p in zip(bricks, where) if b not in todo]
            say(f'{len(todo)} brick{"s" if len(todo) != 1 else ""} to move this round: '
                + ', '.join(f'#{b["id"]} {b.get("colour", "")} {b["label"]}' for b in todo))
            board.text(f'Round {round_no}: {len(todo)} bricks to move, {picked} moved before', 'Picking')
            arguments = command(dict(message, bricks=todo), config,
                                ['--countdown', str(args.countdown if round_no == 1 else 0), *extra])
            if args.dry_run:
                finish('DRY RUN: ' + ' '.join([script, *arguments]))
                return 0
            code, current, done = run_pick([sys.executable, '-s', HERE / script, *arguments], todo, board)
            picked += done
            if code == 130:
                raise KeyboardInterrupt
            if code:
                if current is not None:
                    x, y = position[current['id']]
                    spot = next((s for s in failed_at if abs(x - s[0]) < 0.006 and abs(y - s[1]) < 0.006), None)
                    if spot:
                        spot[2] += 1
                    else:
                        failed_at.append([x, y, 1])
                recover(f'{script} failed in round {round_no}' + (f' at #{current["id"]} {current.get("colour", "")}' if current else ''),
                        held=current)
                left_behind = []                                 # the failure may have moved anything: no comparison next round
                continue
            failures = 0
        finish(f'STOPPED after --max-rounds {args.max_rounds} rounds ({picked} bricks moved): the pile is not empty yet.')
        return 1
    except KeyboardInterrupt:
        finish('CANCELLED (Ctrl+C). The arm stays where it is.')
        return 130
    except (RuntimeError, ValueError, OSError) as problem:
        finish(f'STOPPED: {problem}. The arm stays where it is. {picked} bricks moved.')
        return 1
    finally:
        board.close(wait=10.0)                           # the last picture stays on screen a moment
        rclpy.try_shutdown()                             # wakes the spinning thread first: destroying under it aborts
        spinner.join(timeout=2.0)
        del listener
        node.destroy_node()


if __name__ == '__main__':
    sys.exit(main())
