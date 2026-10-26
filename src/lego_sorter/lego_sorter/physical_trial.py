"""Configuration and terminal gates for bounded, supervised physical trials."""
import copy
import math
import numpy as np
from .core import Cancelled


def boundary_from_corners(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a,b]).all():
        raise ValueError('Boundary corners need finite XY pairs')
    low, high = np.minimum(a,b), np.maximum(a,b)
    if np.any(high-low < .04):
        raise ValueError('Source rectangle must be at least 40 mm wide and deep')
    return dict(x=[float(low[0]),float(high[0])], y=[float(low[1]),float(high[1])])


def region_contains(region, xy):
    return region['x'][0] <= xy[0] <= region['x'][1] and region['y'][0] <= xy[1] <= region['y'][1]


def correct_trial_observation(observation, c, current_yaw):
    """Apply the explicit trial compensation once, before source-neighbour checks."""
    from .vendor.ameed_motion import nearest_yaw,hand_frame_to_base
    result=copy.deepcopy(observation)
    for brick in result['bricks']:
        if 'uncorrected_centre' in brick:
            raise ValueError('Trial correction was already applied to this observation')
        brick['uncorrected_centre']=list(brick['centre'])
        heading=nearest_yaw(brick['yaw'],current_yaw)
        offset=hand_frame_to_base(c['pick_correction_tool_m'],heading)
        for i in range(2):
            brick['centre'][i]+=offset[i];brick['top'][i]+=offset[i]
        if not region_contains(c['source_boundary'],brick['centre'][:2]):
            raise ValueError('Corrected pick target is outside source boundary')
    return result


def build_trial_config(base, teaching, runtime, count, pick_x_mm=0.):
    if isinstance(count,bool) or not isinstance(count,int) or not 1 <= count <= 30:
        raise ValueError('Physical trial requires 1 through 30 verified placements')
    if not math.isfinite(pick_x_mm) or abs(pick_x_mm) > 25:
        raise ValueError('Pick correction must be within +/-25 mm')
    records = teaching['destinations']
    names = ('source_corner_a','source_corner_b','place','discard','observation')
    if 'grasp' in records:
        names += ('grasp',)
    for name in names:
        if name not in records:
            raise ValueError(f'Missing taught {name}; run record_destination.py --name {name}')
        r = records[name]
        if r['base_frame'] != base['base_frame'] or r['tcp_frame'] != base['ee_frame']:
            raise ValueError(f'Frame mismatch in taught {name}')
        if not np.isfinite(r['captured_tcp_xyz_m']).all():
            raise ValueError(f'Invalid taught {name} pose')
    region = boundary_from_corners(records['source_corner_a']['xy_m'],records['source_corner_b']['xy_m'])
    for axis,key in [('x','workspace_x'),('y','workspace_y')]:
        if region[axis][0] < base[key][0] or region[axis][1] > base[key][1]:
            raise ValueError('Source boundary is outside configured workspace')
    c = copy.deepcopy(base)
    c.update(mode='hardware',runtime=str(runtime),staged_physical_trial=True,trial_brick_limit=count,
             source_mode='detected',source_boundary=region,destination_clearance_check=False,
             hardware_validated=False,geometry_confirmed=False,verification='operator',
             inspection_mode=False,snap_heights=False,camera_height_bias=0.,
             velocity_scale=min(float(base['velocity_scale']),.02),tcp_segment_speed_limit_m_s=.01,
             pick_correction_tool_m=[pick_x_mm/1000,0.],tcp_correction_hand=[0.,0.],
             camera_serial='829212071248')
    c['source'].update(origin=[sum(region['x'])/2,sum(region['y'])/2,base['table_top_z']],
                       layers=1,stud_connected=False)
    c['max_brick_top_above_table'] = c['block_size'][2]+.008
    # The existing hardware configuration inherits Ameed's TCP-to-brick geometry.
    # A new physical teaching is an optional override, not a prerequisite.
    height = float(base['grasp_tcp_height'])
    c['grasp_height_source'] = 'hardware.yaml:grasp_tcp_height'
    if 'grasp' in records:
        grasp = records['grasp']
        height = float(grasp['captured_tcp_xyz_m'][2])-base['table_top_z']
        q = np.asarray(grasp['captured_tcp_quaternion_xyzw'],float)
        if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.001:
            raise ValueError('Invalid taught grasp quaternion')
        if 1-2*(q[0]**2+q[1]**2) > -math.cos(math.radians(base['arrival_angle_deg'])):
            raise ValueError('Teach grasp height with the tool pointing down within the configured arrival tolerance')
        c['grasp_height_source'] = 'teaching:grasp'
    if not math.isfinite(height) or not base['min_tcp_above_table'] <= height <= .04:
        raise ValueError('Grasp TCP height violates existing table-clearance limits')
    c['grasp_tcp_height'] = height
    obs = records['observation']
    c['observation_pose'] = list(obs['observation_pose_xyz_yaw'])
    c['observation_quaternion_xyzw'] = list(obs['captured_tcp_quaternion_xyzw'])
    c['destinations'] = {}
    for name, key in [('place','selected'),('discard','discard')]:
        # Separate cells, one loose layer. The operator owns destination clearance.
        c['destinations'][key] = dict(origin=[*records[name]['xy_m'],base['table_top_z']],
            rows=6,columns=5,layers=1,yaw_deg=90,brick_gap=0.,row_gap_studs=2,
            stud_pitch=.008,layer_overrides={},stud_connected=False,kind='table_grid')
    c['teaching_evidence'] = {name:records[name] for name in names}
    return c


def validate_trial(c):
    if (not c.get('staged_physical_trial') or c.get('source_mode')!='detected'
            or c['source']['layers']!=1 or c['source'].get('stud_connected')
            or any(d['layers']!=1 or d.get('kind')!='table_grid' or d.get('stud_connected')
                   for d in c['destinations'].values())):
        raise ValueError('Scoped physical runner supports one loose layer and table-grid destinations only')
    limit = c.get('trial_brick_limit')
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=30:
        raise ValueError('Physical trial count must be 1 through 30')
    region = c['source_boundary']
    boundary_from_corners([region['x'][0],region['y'][0]],[region['x'][1],region['y'][1]])
    from .core import layout,bounds,matrix
    obs=c['observation_pose']
    if (len(obs)!=4 or not np.isfinite(obs).all()
            or not c['workspace_x'][0]<=obs[0]<=c['workspace_x'][1]
            or not c['workspace_y'][0]<=obs[1]<=c['workspace_y'][1]
            or not c['table_top_z']+.05<=obs[2]<=c['max_hover_z']):
        raise ValueError('Taught observation pose is outside the hover workspace')
    matrix(obs[:3],c['observation_quaternion_xyzw'])
    for spec in c['destinations'].values():
        for cell in layout(spec,c['block_size'],'destination'):
            x0,y0,x1,y1 = bounds(cell)
            if not (c['workspace_x'][0]<=x0<=x1<=c['workspace_x'][1]
                    and c['workspace_y'][0]<=y0<=y1<=c['workspace_y'][1]):
                raise ValueError('Destination grid extends outside configured workspace')
            # Identity boundary, not a clearance check: prevent re-picking placed bricks.
            if x0<region['x'][1] and x1>region['x'][0] and y0<region['y'][1] and y1>region['y'][0]:
                raise ValueError('A destination cell overlaps the source search region; teach separate regions')
    height = c['grasp_tcp_height']
    if not math.isfinite(height) or not c['min_tcp_above_table'] <= height <= .04:
        raise ValueError('Grasp TCP height violates existing table-clearance limits')
    for name in ('place','discard','observation','source_corner_a','source_corner_b'):
        if name not in c.get('teaching_evidence',{}):
            raise ValueError('Physical trial requires complete teaching evidence')


class PhysicalStageGate:
    def __init__(self, feedback, health, reader=None, log=None):
        self.feedback,self.health = feedback,health
        self.reader = reader or input
        self.log = log or (lambda **_:None)
        self.stop = lambda:None
        self.number = 0

    def __call__(self, stage, hand):
        self.health.check(); self.feedback.stationary()
        before = self.feedback.fresh()
        self.number += 1
        print(f'\nPHYSICAL stage {self.number}: {stage}; hand={hand}',flush=True)
        if stage == 'LOWER':
            print('Continue only if the fingers are physically centred/aligned above the selected brick.',flush=True)
        if stage == 'LIFT':
            print('Continue only if the gripper has securely grasped the brick.',flush=True)
        while True:
            try: answer = self.reader('Enter = perform this stage on the real robot; q = stop: ').strip().lower()
            except (EOFError,KeyboardInterrupt): answer='q'
            if answer == 'q':
                self.stop()
                raise Cancelled(f'Operator stopped before {stage}')
            if answer == '': break
        self.health.check(); self.feedback.stationary()
        if np.max(np.abs(self.feedback.fresh()-before)) > .001:
            self.stop()
            raise Cancelled('Arm moved while waiting for stage approval; stop and replan')
        self.log(stage=stage,hand=hand)
