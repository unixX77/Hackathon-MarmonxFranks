import math
import numpy as np

def wrap_line_angle(angle):
    """The brick's long axis repeats every pi: report it in (-pi/2, pi/2]."""
    wrapped = (angle + math.pi / 2) % math.pi - math.pi / 2
    return math.pi / 2 if wrapped <= -math.pi / 2 + 1e-12 else wrapped

def plan_pick(c, x, y, z=None, yaw=0.0, pose_is_tcp=False):
    """Everything the mission needs for one brick pose. Raises ValueError for a bad pose."""
    top, height = c['table_top_z'], c['block_size'][2]
    offset = c['grasp_tcp_height'] - height / 2        # TCP sits this far above the brick centre
    if pose_is_tcp and z is None:
        raise ValueError('--pose-is-tcp / --place-is-tcp needs a z value')
    if z is None:
        centre_z = top + height / 2
    elif pose_is_tcp:
        centre_z = z - offset
    else:
        centre_z = z
    grip_z = centre_z + offset
    for name, value, (low, high) in (('x', x, c['workspace_x']), ('y', y, c['workspace_y'])):
        if not low <= value <= high:
            raise ValueError(f'{name}={value:.3f} is outside the allowed window [{low}, {high}] '
                             f'(workspace_{name} in the config)')
    if centre_z - height / 2 < top - 0.002:
        raise ValueError(f'Brick centre z={centre_z:.4f} puts the brick {1000 * (top - (centre_z - height / 2)):.0f} mm '
                         f'below the table top ({top:.3f})')
    if grip_z < top + c['min_tcp_above_table']:
        raise ValueError(f'Grasp height z={grip_z:.4f} would put the fingertips in the table '
                         f'(needs at least {top + c["min_tcp_above_table"]:.4f})')
    if grip_z + c['clearance'] > c['max_hover_z']:
        raise ValueError(f'Hover height {grip_z + c["clearance"]:.3f} is above max_hover_z {c["max_hover_z"]}')
    inspect = c.get('inspect_height', 0.04)
    if not 0 < inspect < c['clearance']:
        raise ValueError('inspect_height must be above the grasp and below the hover clearance')
    return {
        'centre': [float(x), float(y), float(centre_z)],
        'grip': [float(x), float(y), float(grip_z)],
        'inspect': [float(x), float(y), float(grip_z + inspect)],
        'above': [float(x), float(y), float(grip_z + c['clearance'])],
        'yaw': float(wrap_line_angle(yaw)),
        'block_in_tcp': [0.0, 0.0, float(offset)],      # tool Z points down
    }

def plan_place(c, x, y, yaw, z=None, pose_is_tcp=False):
    """Where the carried brick is put down. Same rules as a pick pose (workspace window, table, fingertips);
    the hand stops place_margin above the height at which the brick rests at the given pose."""
    plan = plan_pick(c, x, y, z, yaw, pose_is_tcp)
    margin = float(c.get('place_margin', 0.0))
    if margin < 0:
        raise ValueError('place_margin must not be negative: it would push the brick into what it rests on')
    plan['release'] = [plan['grip'][0], plan['grip'][1], plan['grip'][2] + margin]
    return plan

def layer_centre_z(c, layers):
    """Height of the centre of a brick lying in layer `layers` of a stack (1 = on the table)."""
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError('layers must be a whole number, 1 or more (1 = on the table)')
    return c['table_top_z'] + (layers - 0.5) * c['block_size'][2]

def layer_top_z(c, layers):
    """Height of the top face of a brick lying in layer `layers`."""
    return layer_centre_z(c, layers) + c['block_size'][2] / 2

def nearest_layer(c, top_z):
    """For a measured top-face height: (nearest layer, signed distance in m from that layer's top face, layers
    above the table as a fraction: 1.0 = a brick on the table, 2.0 = one on top of another)."""
    levels = (top_z - c['table_top_z']) / c['block_size'][2]
    n = max(1, int(round(levels)))
    return n, top_z - layer_top_z(c, n), levels
