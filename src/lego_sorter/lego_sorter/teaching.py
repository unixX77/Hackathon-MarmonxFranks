"""Apply taught regions and observation pose to explicitly simulated trials."""
import copy
import math


def apply_mock_teachings(config, teaching):
    if config['mode'] != 'mock':
        raise ValueError('Taught-pose import currently supports mock trials only')
    records = teaching['destinations']
    required = ('place', 'discard', 'observation')
    for name in required:
        if name not in records:
            raise ValueError(f'Missing taught {name} pose; record it first')
        record = records[name]
        if record['base_frame'] != config['base_frame'] or record['tcp_frame'] != config['ee_frame']:
            raise ValueError(f'Taught {name} frame does not match trial configuration')
        xy = record['xy_m']
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError(f'Invalid taught {name} XY')
    result = copy.deepcopy(config)
    for name, destination in (('place', 'selected'), ('discard', 'discard')):
        spec = result['destinations'][destination]
        spec['origin'] = [*records[name]['xy_m'], config['table_top_z']]
        if name == 'discard':
            # The operator chose a table region: use separate placement cells.
            for key in ('kind', 'capacity', 'drop_centre_height'):
                spec.pop(key, None)
            spec.update(rows=config['source']['rows'], columns=config['source']['columns'],
                        layers=1, stud_connected=False)
    observation = records['observation']['observation_pose_xyz_yaw']
    if len(observation) != 4 or not all(math.isfinite(v) for v in observation):
        raise ValueError('Invalid taught observation XYZ/yaw')
    if not (config['workspace_x'][0] <= observation[0] <= config['workspace_x'][1]
            and config['workspace_y'][0] <= observation[1] <= config['workspace_y'][1]
            and config['table_top_z']+.05 <= observation[2] <= config['max_hover_z']):
        raise ValueError('Taught observation pose is outside configured hover workspace')
    result['observation_pose'] = list(observation)
    result['taught_observation_quaternion_xyzw'] = list(records['observation']['captured_tcp_quaternion_xyzw'])
    return result
