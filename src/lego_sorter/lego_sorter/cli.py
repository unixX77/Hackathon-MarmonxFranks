"""Inspection/layout preview and full offline mock loop. Never opens robot connections."""
import argparse
import copy
import json
import math
from pathlib import Path
from .core import load_config,inside,layout,bounds,Store,Manipulator
from .workflow import MockWorld,LocalTransport,Coordinator


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
    p.add_argument('--runtime'); args=p.parse_args(); c=load_config(args.config)
    if args.runtime: c['runtime']=str(inside(args.runtime)); inside(c['runtime']).mkdir(parents=True,exist_ok=True)
    if args.command=='preview': print(preview(c,Path(c['runtime'])/'preview')); return
    if args.command=='inspect':
        print(json.dumps({k:c[k] for k in ['mode','source','destinations','block_size','calibration_parent','calibration_file','policy']},indent=2)); return
    if c['mode']!='mock': p.error('Offline command requires mode: mock')
    world=MockWorld(c); m=Manipulator(c,world)
    if world.read()['held']: raise RuntimeError('Mock restart has held object; reconcile explicitly')
    m.reconcile_empty('SIMULATED world inspection')
    store=Store(c); engine=Coordinator(c,store,LocalTransport(m,world)); engine.control('start')
    while engine.tick(): pass
    print(json.dumps(engine.data,indent=2))
    if engine.data['status']!='COMPLETE': raise SystemExit(1)


if __name__=='__main__': main()
