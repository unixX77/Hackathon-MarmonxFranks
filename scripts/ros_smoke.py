#!/usr/bin/python3 -s
"""Launch all three actual ROS nodes in mock mode and assert 60-brick completion."""
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
import time
import yaml
import rclpy
from rclpy.node import Node
from lego_sort_interfaces.srv import TaskControl
from std_msgs.msg import String

root=Path('/home/tanay/Documents/Hackathon/project-planner')
runtime=root/'runtime'/f'ros-smoke-{time.time_ns()}'; runtime.mkdir(parents=True)
c=yaml.safe_load(Path(sys.argv[1] if len(sys.argv)>1 else root/'src/lego_sorter/config/sorter.yaml').read_text()); c['runtime']=str(runtime)
if c['mode'] not in ('mock','replay'): raise ValueError('Smoke test forbids hardware')
config=runtime/'config.yaml'; config.write_text(yaml.safe_dump(c))
env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',ROS_DOMAIN_ID='174',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(runtime/'ros_logs'),ROS_HOME=str(runtime/'ros_home'))
os.environ.update({k:env[k] for k in ['ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY','ROS_LOG_DIR','ROS_HOME']})
processes=[]; logs=[]
try:
    for name in ['manipulation','perception','coordinator']:
        log=open(runtime/(name+'.log'),'w'); logs.append(log)
        processes.append(subprocess.Popen(['/usr/bin/python3','-s','-c',f'from lego_sorter.ros_nodes import {name}_main; {name}_main()','--ros-args','-p',f'config:={config}'],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root))
    rclpy.init(); node=Node('lego_smoke_probe'); client=node.create_client(TaskControl,'task_control'); latest={}
    def status(msg): latest.update(json.loads(msg.data))
    sub=node.create_subscription(String,'sorter_status',status,10)
    if not client.wait_for_service(timeout_sec=20): raise RuntimeError('Coordinator not available')
    f=client.call_async(TaskControl.Request(command='start'))
    rclpy.spin_until_future_complete(node,f,timeout_sec=20)
    if not f.done() or not f.result().success: raise RuntimeError('Start failed: '+str(f.result() if f.done() else 'timeout'))
    # Pause finishes the current transaction; resume obtains a new observation.
    f=client.call_async(TaskControl.Request(command='pause')); rclpy.spin_until_future_complete(node,f,timeout_sec=10)
    assert f.done() and f.result().success
    deadline=time.monotonic()+15
    while latest.get('status')!='PAUSED' and time.monotonic()<deadline: rclpy.spin_once(node,timeout_sec=.1)
    assert latest.get('status')=='PAUSED',latest
    f=client.call_async(TaskControl.Request(command='stop')); rclpy.spin_until_future_complete(node,f,timeout_sec=10)
    assert f.done() and f.result().success
    deadline=time.monotonic()+5
    while latest.get('status')!='STOPPED' and time.monotonic()<deadline: rclpy.spin_once(node,timeout_sec=.1)
    assert latest.get('status')=='STOPPED',latest
    f=client.call_async(TaskControl.Request(command='resume')); rclpy.spin_until_future_complete(node,f,timeout_sec=10)
    assert f.done() and f.result().success,str(f.result() if f.done() else 'timeout')
    latest.clear()  # Wait for a post-resume status publication, not cached STOPPED.
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.1)
        if any(p.poll() is not None for p in processes): raise RuntimeError('ROS node exited')
        if latest.get('status') in ('COMPLETE','FAILED','STOPPED'): break
    assert latest.get('status')=='COMPLETE',latest
    assert latest['completed']==60,latest
    slots=json.loads((runtime/'slots.json').read_text())['slots']
    assert sum(s['state']=='PLACED' for s in slots)==60
    assert sum(s['state']=='PLACED' and s['destination']=='discard' for s in slots)==24
    assert json.loads((runtime/'hand.json').read_text())['hand']=='EMPTY'
    (runtime/'result.json').write_text(json.dumps(latest,indent=2)); print(json.dumps(dict(result=latest,runtime=str(runtime)),indent=2))
    node.destroy_node(); rclpy.shutdown()
finally:
    for p in processes:
        if p.poll() is None: p.send_signal(signal.SIGINT)
    for p in processes:
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired: p.kill(); p.wait()
    for log in logs: log.close()
