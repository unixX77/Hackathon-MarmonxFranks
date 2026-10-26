#!/usr/bin/env python3
"""Supervised PHYSICAL loose-layer trial. No mock backend or automatic approvals."""
import argparse
import copy
from datetime import datetime
import fcntl
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/lego_sorter'))


def run(c, config_path):
    import math
    import numpy as np
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.time import Time
    from tf2_ros import Buffer,TransformListener
    from lego_sorter.core import Manipulator,Store,matrix,Cancelled
    from lego_sorter.hardware import Hardware
    from lego_sorter.perception import SamProcess
    from lego_sorter.physical_trial import PhysicalStageGate,correct_trial_observation
    from lego_sorter.workflow import LocalTransport,Coordinator

    rclpy.init();node=rclpy.create_node('lego_supervised_physical_trial')
    buffer=Buffer(cache_time=Duration(seconds=180));listener=TransformListener(buffer,node)
    executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    backend=machine=store=worker=engine=None
    try:
        print('Connecting to existing MoveIt/gripper services and checking live Franka health...',flush=True)
        backend=Hardware(node,c,buffer);backend.ready()
        gate=PhysicalStageGate(backend.feedback,backend.mover.health)
        print('\nREAL ROBOT: one loose layer; no stud extraction/seating.\n'
              'Every stage requires Enter. q stops; there is no automatic continuation.\n'
              'Confirm the actual hand is empty and the destination areas are clear.',flush=True)
        gate('CONFIRM_EMPTY_HAND_AND_CLEAR_DESTINATIONS','UNRECONCILED')
        machine=Manipulator(c,backend)
        machine.reconcile_empty('Operator explicitly confirmed empty physical hand at trial startup')
        current=backend.mover.hand_pose()
        if current[2,3]<c['table_top_z']+c['clearance']:
            gate('INITIAL_VERTICAL_CLEARANCE','EMPTY')
            from scipy.spatial.transform import Rotation
            xyz=current[:3,3].copy();xyz[2]=max(xyz[2],c['transfer_z'])
            backend.prepare()
            backend.mover.move_pose('INITIAL_VERTICAL_CLEARANCE',xyz.tolist(),
                                   Rotation.from_matrix(current[:3,:3]).as_quat().tolist())
        print('Loading SAM3 and starting the camera; no further motion until initialization finishes.',flush=True)
        worker=SamProcess(c,str(config_path));worker.start()
        backend.mover.health.check()

        class Transport(LocalTransport):
            def observe(self):
                gate('CAPTURE',self.hand())
                token,epoch=machine.capture_begin()
                observation=None
                try:
                    def lookup(stamp_ns):
                        t=buffer.lookup_transform(c['base_frame'],c['calibration_parent'],
                            Time(nanoseconds=stamp_ns),timeout=Duration(seconds=3)).transform
                        p,q=t.translation,t.rotation
                        return matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
                    observation=worker.observe(epoch,lookup)
                    current=backend.mover.hand_pose()
                    observation=correct_trial_observation(observation,c,math.atan2(current[1,0],current[0,0]))
                    print(f'Detected {len(observation["bricks"])} source brick(s); '
                          f'camera diagnostics: {observation["diagnostics"]}',flush=True)
                    return observation
                finally:
                    machine.capture_end(token,observation['id'] if observation else '',
                                        observation['stamp_ns'] if observation else 0)
            def pick(self,brick,observation):
                target=copy.deepcopy(brick)
                engine.log.event('physical_pick_target',raw_centre=brick['uncorrected_centre'],
                    corrected_centre=target['centre'],correction_tool_m=c['pick_correction_tool_m'])
                print(f'Pick target: {target["centre"]}; colour={target["colour"]}; '
                      f'correction tool XY={c["pick_correction_tool_m"]}',flush=True)
                return machine.run('pick',target,observation['epoch'],feedback=gate)
            def verify(self,slot):
                print(f'Verify {slot["destination"]}: brick released at {slot["centre"]}, hand empty.',flush=True)
                gate('VERIFY_PHYSICAL_PLACEMENT_AND_EMPTY_HAND',self.hand())
                return True,'Operator confirmed physical placement and empty hand by Enter'

        store=Store(c);transport=Transport(machine,worker,gate)
        engine=Coordinator(c,store,transport);gate.stop=lambda:engine.control('stop')
        gate.log=lambda **fields:engine.log.event('physical_stage_approved',**fields)
        engine.control('start')
        while engine.data['completed'] < c['trial_brick_limit']:
            if not engine.tick():break
        if engine.data['completed'] >= c['trial_brick_limit']:
            engine.status('TRIAL_COMPLETE',f'{c["trial_brick_limit"]} operator-verified placements; source-empty not asserted')
        print(json.dumps(engine.data,indent=2))
        print(f'Physical hand state: {machine.state["hand"]}. No automatic return or release.',flush=True)
        return 0 if engine.data['status'] in ('TRIAL_COMPLETE','COMPLETE','STOPPED') else 1
    except (KeyboardInterrupt,Cancelled):
        if engine:engine.control('stop')
        elif machine:machine.stop()
        print('Physical trial stopped. No automatic release or return; inspect hand and saved state.',flush=True)
        return 1
    finally:
        if machine:machine.stop()
        if worker:worker.close()
        if store:store.close()
        if machine:machine.close()
        executor.shutdown(timeout_sec=5);thread.join(timeout=5)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bricks',type=int,default=3,help='Maximum verified placements, 1–30')
    parser.add_argument('--teachings',type=Path,default=ROOT/'runtime/destination-teaching/destinations.json')
    parser.add_argument('--pick-tool-x-mm',type=float,default=0.,help='Explicit pick-only tool-X correction; maximum +/-25 mm')
    parser.add_argument('--execute',action='store_true',help='Enable physical stages; each still requires Enter')
    args=parser.parse_args()
    import yaml
    from lego_sorter.core import load_config
    from lego_sorter.physical_trial import build_trial_config,validate_trial
    try:
        base=yaml.safe_load((ROOT/'src/lego_sorter/config/hardware.yaml').read_text())
        teaching=json.loads(args.teachings.read_text())
        output=ROOT/'runtime'/('physical-one-layer-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        c=build_trial_config(base,teaching,output,args.bricks,args.pick_tool_x_mm)
        validate_trial(c)
        output.mkdir(parents=True)
        config_path=output/'config.yaml';config_path.write_text(yaml.safe_dump(c,sort_keys=False))
        c=load_config(config_path,supervised_trial=True)
    except (ValueError,KeyError,OSError) as exc:
        parser.error(str(exc))
    print(f'Physical trial configuration: {config_path}\nSource boundary: {c["source_boundary"]}\n'
          f'Pick-only correction: {c["pick_correction_tool_m"]} m\n'
          f'Grasp TCP above table: {1000*c["grasp_tcp_height"]:.1f} mm; '
          f'release adds {1000*c["place_margin"]:.1f} mm.\n'
          f'Grasp height source: {c["grasp_height_source"]}\n'
          'Destination clearance allocation check disabled by operator request.\n'
          'Table/self/held/placed collision checks remain; assumed source boxes are omitted.',flush=True)
    if not args.execute:
        print('Configuration check only. No ROS connection, camera capture or robot commands. Add --execute for physical stages.')
        return 0
    with (ROOT/'runtime/physical-one-layer.lock').open('a') as owner:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Another physical one-layer trial is already running')
        return run(c,config_path)


if __name__=='__main__':sys.exit(main())
