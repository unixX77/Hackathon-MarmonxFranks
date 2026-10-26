"""Source-region masks in robot-base coordinates; no expected brick cells."""
import numpy as np


def source_pixel_mask(depth, info, base_camera, region):
    intr = info['aligned_depth_intrinsics']
    if any(abs(v)>1e-8 for v in intr.get('distortion_coeffs',[])):
        raise ValueError('Source ROI requires rectified aligned-depth intrinsics')
    v,u = np.indices(depth.shape)
    z = depth.astype(float)*info['depth_scale_m']
    points = np.stack([(u-intr['ppx'])*z/intr['fx'],(v-intr['ppy'])*z/intr['fy'],z],-1)
    world = points @ np.asarray(base_camera)[:3,:3].T + np.asarray(base_camera)[:3,3]
    return ((depth>0) & np.isfinite(world).all(axis=-1)
            & (world[:,:,0]>=region['x'][0]) & (world[:,:,0]<=region['x'][1])
            & (world[:,:,1]>=region['y'][0]) & (world[:,:,1]<=region['y'][1]))
