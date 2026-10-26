"""Inspection/layout preview and full offline mock loop. Never opens robot connections."""
import argparse
import copy
import json
import math
from pathlib import Path
import yaml
from .core import load_config,inside,layout,bounds,Store,Manipulator,Cancelled
from .workflow import MockWorld,LocalTransport,Coordinator
from .teaching import apply_mock_teachings


class StagePrompt:
    """Pause before simulated actions; never insert prompts into hardware execution."""
    descriptions = {
        'OBSERVE_MOVE': 'Move to observation pose', 'CAPTURE': 'Observe simulated bricks',
        'OPEN': 'Open the empty hand', 'APPROACH': 'Approach above the selected brick',
        'INSPECT': 'Inspect above the brick', 'LOWER': 'Lower to grasp height',
        'GRASP': 'Close hand and grasp', 'LIFT': 'Lift the brick',
        'TRANSFER_LIFT': 'Lift to transfer height', 'TRANSFER': 'Transfer above destination',
        'PLACE_LOWER': 'Lower at destination', 'RELEASE': 'Release the brick',
        'RETREAT': 'Retreat from destination', 'VERIFY_PLACEMENT': 'Verify simulated placement',
    }

    def __init__(self, reader=None):
        self.reader = reader or input
        self.on_stop = lambda: None
        self.on_approve = lambda stage: None
        self.number = 0

    def __call__(self, stage, hand):
        self.number += 1
        print(f'\nSIMULATED stage {self.number}: {stage} — '
              f'{self.descriptions.get(stage, stage)}; hand={hand}', flush=True)
        while True:
            try:
                answer = self.reader('Enter = perform this stage; q = stop: ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = 'q'
            if answer == 'q':
                self.on_stop()
                raise Cancelled(f'Operator stopped before {stage}')
            if answer == '':
                self.on_approve(stage)
                return
            print('Press Enter to advance one stage, or type q to stop.', flush=True)


def run_mock(c, step=False, reader=None):
    """The main mission coordinator with simulated backends and optional stage gates."""
    if c['mode'] != 'mock':
        raise ValueError('Offline mock runner requires mode: mock')
    world=MockWorld(c); m=Manipulator(c,world)
    if world.read()['held']: raise RuntimeError('Mock restart has held object; reconcile explicitly')
    m.reconcile_empty('SIMULATED world inspection')
    gate=StagePrompt(reader) if step else None
    store=Store(c)
    try:
        engine=Coordinator(c,store,LocalTransport(m,world,gate))
        if gate:
            gate.on_stop=lambda: engine.control('stop')
            gate.on_approve=lambda stage: engine.log.event('simulated_stage_approved',stage=stage)
        engine.control('start')
        while engine.tick(): pass
        if engine.data['status']=='STOPPED':
            print(f'Stopped. Simulated hand={m.state["hand"]}; state preserved in {c["runtime"]}.')
        return engine.data
    finally:
        store.close()


def preview(c,out):
    out=inside(out); out.mkdir(parents=True,exist_ok=True)
    groups={'source':layout(c['source'],c['block_size'],'source')}
    groups.update({name:layout(spec,c['block_size'],name) for name,spec in c['destinations'].items()})
    (out/'layout.json').write_text(json.dumps(groups,indent=2))
    colours=['#569cd6','#e65656','#b45de0','#e6a356','#5fc6ac','#8ea959','#aaa']
    panels=[]
    layers=max(s['layer'] for slots in groups.values() for s in slots)+1
    for layer in range(layers):
        shapes=[]
        for i,(name,slots) in enumerate(groups.items()):
            for s in slots:
                if s['layer']!=layer: continue
                x0,y0,x1,y1=bounds(s); x=(x0-.1)*850; y=(.55-y1)*650
                shapes.append(f'<rect x="{x}" y="{y}" width="{(x1-x0)*850}" height="{(y1-y0)*650}" fill="{colours[i%len(colours)]}" stroke="#111" stroke-width=".6"><title>{s["id"]}; centre={s["centre"]}; yaw={s["yaw"]:.3f}</title></rect>')
        panels.append(f'<h2>Layer {layer} (zero based)</h2><svg viewBox="0 0 700 715">'+''.join(shapes)+'</svg>')
    html='<!doctype html><meta charset="utf-8"><title>LEGO layout inspection</title><style>body{font:16px system-ui;background:#18212c;color:#eee;margin:30px}svg{background:#eee;max-width:650px;width:100%}h2{margin-top:30px}</style><h1>Structured LEGO layout — assumed geometry</h1><p>Metres in base frame; base +X right, +Y up. Hover over bricks for coordinates. Whole-grid rotation about centre; rows across short axis, five columns along long axis. This preview authorizes no motion.</p><p>'+', '.join(groups)+'</p>'+''.join(panels)
    (out/'layout.html').write_text(html)
    return out/'layout.html'


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['preview','inspect','mock'])
    p.add_argument('--config',default='/home/tanay/Documents/Hackathon/project-planner/src/lego_sorter/config/sorter.yaml')
    p.add_argument('--runtime')
    p.add_argument('--bricks',type=int,help='Number of simulated source bricks (first N configured slots)')
    p.add_argument('--step',action='store_true',help='Pause for Enter before each simulated stage; q stops')
    p.add_argument('--teachings',type=Path,help='Use recorded place/discard centres and observation pose in this mock run')
    args=p.parse_args(); c=load_config(args.config)
    if (args.bricks is not None or args.step or args.teachings is not None) and args.command!='mock':
        p.error('--bricks, --step and --teachings are supported only for mock runs')
    if args.runtime: c['runtime']=str(inside(args.runtime)); inside(c['runtime']).mkdir(parents=True,exist_ok=True)
    if args.command=='preview': print(preview(c,Path(c['runtime'])/'preview')); return
    if args.command=='inspect':
        print(json.dumps({k:c[k] for k in ['mode','source','destinations','block_size','calibration_parent','calibration_file','policy']},indent=2)); return
    if c['mode']!='mock': p.error('Offline command requires mode: mock')
    if args.teachings is not None:
        if (Path(c['runtime'])/'mock_world.json').exists():
            p.error('Use a fresh --runtime to import taught poses; existing state is preserved')
        try:
            c=apply_mock_teachings(c,json.loads(args.teachings.read_text()))
        except (ValueError,KeyError,OSError) as exc:
            p.error(str(exc))
        c['taught_poses_source']=str(args.teachings.resolve())
        print(f'MOCK observation XYZ/yaw: {c["observation_pose"]}',flush=True)
        print(f'MOCK red layer centre: {c["destinations"]["selected"]["origin"]}; '
              f'discard table-area centre: {c["destinations"]["discard"]["origin"]}',flush=True)
    if args.bricks is not None:
        capacity=len(layout(c['source'],c['block_size'],'source'))
        if not 1<=args.bricks<=capacity: p.error(f'--bricks must be between 1 and {capacity}')
        if (Path(c['runtime'])/'mock_world.json').exists():
            p.error('Use a fresh --runtime when specifying --bricks; existing state is preserved')
        c['mock_brick_count']=args.bricks
    (Path(c['runtime'])/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    count=c.get('mock_brick_count',len(layout(c['source'],c['block_size'],'source')))
    print(f'SIMULATION ONLY: configured source count={count}; stage prompts={args.step}.',flush=True)
    data=run_mock(c,step=args.step)
    print(json.dumps(data,indent=2))
    if data['status'] not in ('COMPLETE','STOPPED'): raise SystemExit(1)


if __name__=='__main__': main()
