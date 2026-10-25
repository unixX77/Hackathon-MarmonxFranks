"""Export 6-DoF poses for classified top-layer bricks.

Camera-frame positions come from RGB-D top-plane detection. Orientation uses
the detected long brick axis projected onto the fitted top plane. If a hand-eye
calibration containing ``T_base_camera`` is provided, poses are also transformed
to the robot base frame. No base-frame pose is fabricated without calibration.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def unit(vector):
    vector = np.asarray(vector, np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError('cannot normalize a zero-length vector')
    return vector / norm


def rotation_to_quaternion(rotation):
    """Return an xyzw quaternion from a proper 3x3 rotation matrix."""
    r = np.asarray(rotation, np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        quaternion = [(r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s, .25 * s]
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            s = np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            quaternion = [.25 * s, (r[0, 1] + r[1, 0]) / s,
                          (r[0, 2] + r[2, 0]) / s,
                          (r[2, 1] - r[1, 2]) / s]
        elif index == 1:
            s = np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            quaternion = [(r[0, 1] + r[1, 0]) / s, .25 * s,
                          (r[1, 2] + r[2, 1]) / s,
                          (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            quaternion = [(r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, .25 * s,
                          (r[1, 0] - r[0, 1]) / s]
    quaternion = unit(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def homogeneous(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def pose_dict(frame, transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return {
        'frame_id': frame,
        'translation_m': {
            'x': round(float(translation[0]), 6),
            'y': round(float(translation[1]), 6),
            'z': round(float(translation[2]), 6),
        },
        'quaternion_xyzw': {
            key: round(float(value), 7)
            for key, value in zip('xyzw', rotation_to_quaternion(rotation))
        },
        'rotation_matrix': np.round(rotation, 8).tolist(),
        'T_frame_brick': np.round(transform, 8).tolist(),
    }


def resolve_source(path_string, quality_path, repository_root):
    source = Path(path_string)
    candidates = [source]
    if not source.is_absolute():
        candidates.extend([repository_root / source, quality_path.parent / source])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'cannot resolve segmentation source {path_string!r}')


def load_base_transform(path, camera_info):
    payload = json.loads(path.read_text()) if path else camera_info.get('camera_pose')
    if payload is None:
        return None, None
    if 'T_base_camera' not in payload:
        raise ValueError('camera pose must contain a 4x4 T_base_camera matrix')
    transform = np.asarray(payload['T_base_camera'], np.float64)
    if transform.shape != (4, 4):
        raise ValueError('T_base_camera must have shape 4x4')
    frame = payload.get('frame', payload.get('frame_id', 'robot_base'))
    return transform, frame


def top_surface_center(box, normal, plane_offset, top_height, intrinsics):
    """Intersect the box-centre camera ray with the fitted brick top plane."""
    u, v = map(float, box['center_xy'])
    ray = np.array([(u - intrinsics['ppx']) / intrinsics['fx'],
                    (v - intrinsics['ppy']) / intrinsics['fy'], 1.0])
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-9:
        raise ValueError(f"box {box['id']} centre ray is parallel to the top plane")
    distance = (float(top_height) - float(plane_offset)) / denominator
    if distance <= 0:
        raise ValueError(f"box {box['id']} top-plane intersection is behind the camera")
    return ray * distance


def camera_brick_transform(box, normal, plane_offset, top_height, intrinsics):
    position = top_surface_center(
        box, normal, plane_offset, top_height, intrinsics)
    angle = np.deg2rad(float(box['angle_deg']))
    # Image x points right and y points down in the RealSense optical frame.
    pixel_axis = np.array([np.cos(angle), np.sin(angle)])
    image_tangent = np.array([pixel_axis[0] * position[2] / intrinsics['fx'],
                              pixel_axis[1] * position[2] / intrinsics['fy'],
                              0.0])
    z_axis = unit(normal)
    # The fitted normal is expected to face the camera. Make this explicit so
    # the brick +Z axis always points out of the graspable top surface.
    if z_axis[2] > 0:
        z_axis = -z_axis
    x_axis = image_tangent - np.dot(image_tangent, z_axis) * z_axis
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(z_axis, x_axis))
    x_axis = unit(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return homogeneous(rotation, position)


def camera_box_geometry(camera_brick, size=(.0318, .0158, .0096)):
    """Fitted physical cuboid; corners 0-3 top, 4-7 bottom."""
    length, width, height = map(float, size)
    top_center = camera_brick[:3, 3]
    long_axis, short_axis, outward_normal = camera_brick[:3, :3].T
    signs = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    top_corners = np.asarray([
        top_center + sx * long_axis * length / 2 + sy * short_axis * width / 2
        for sx, sy in signs])
    bottom_corners = top_corners - outward_normal * height
    box_center = top_center - outward_normal * height / 2
    return {
        'top_face_center_camera_m': np.round(top_center, 7).tolist(),
        'box_center_camera_m': np.round(box_center, 7).tolist(),
        'box_size_m': [length, width, height],
        'box_rotation_camera': np.round(camera_brick[:3, :3], 8).tolist(),
        'top_corners_camera_m': np.round(top_corners, 7).tolist(),
        'bottom_corners_camera_m': np.round(bottom_corners, 7).tolist(),
        'box_corners_camera_m': np.round(
            np.vstack((top_corners, bottom_corners)), 7).tolist(),
        'corner_order': [
            'top(+long,+short)', 'top(-long,+short)',
            'top(-long,-short)', 'top(+long,-short)',
            'bottom(+long,+short)', 'bottom(-long,+short)',
            'bottom(-long,-short)', 'bottom(+long,-short)',
        ],
    }


def export_frame(quality_path, output_name, base_transform_path,
                 approach_clearance_m, repository_root):
    quality = json.loads(quality_path.read_text())
    segmentation_path = resolve_source(
        quality['segmentation_source'], quality_path, repository_root)
    segmentation = json.loads(segmentation_path.read_text())
    rgb_path = Path(quality['rgb'])
    camera_info_path = rgb_path.parent.parent / 'camera_info.json'
    camera_info = json.loads(camera_info_path.read_text())
    intrinsics = camera_info['aligned_depth_intrinsics']
    normal = np.asarray(segmentation['support_plane']['normal_camera'], np.float64)
    plane_offset = float(segmentation['support_plane']['d'])
    top_height = float(segmentation['top_height_m'])
    base_camera, base_frame = load_base_transform(base_transform_path, camera_info)

    poses = []
    for box in quality['boxes']:
        camera_brick = camera_brick_transform(
            box, normal, plane_offset, top_height, intrinsics)
        camera_pose = pose_dict('camera_color_optical_frame', camera_brick)
        camera_box = camera_box_geometry(camera_brick)
        surface_normal_camera = camera_brick[:3, 2]
        approach_camera = surface_normal_camera * approach_clearance_m
        record = {
            'id': box['id'],
            'quality': box['quality'],
            'colour': box['colour'],
            'classification_confidence': box['confidence'],
            'detection': {
                'top_coverage': box['top_coverage'],
                'sam_overlap': box['sam_overlap'],
                'source': box['source'],
                'center_pixel_xy': box['center_xy'],
                'corners_pixel_xy': box['corners_xy'],
            },
            'pose_camera': camera_pose,
            'camera_box': camera_box,
            'depth_median_position_camera_m': box['position_camera_m'],
            'surface_normal_camera': np.round(surface_normal_camera, 8).tolist(),
            'approach_direction_camera': np.round(-surface_normal_camera, 8).tolist(),
            'pregrasp_translation_camera_m': np.round(
                camera_brick[:3, 3] + approach_camera, 6).tolist(),
            'nominal_brick_size_m': {'long': 0.0318, 'short': 0.0158,
                                     'height': 0.0096},
        }
        if base_camera is not None:
            base_brick = base_camera @ camera_brick
            surface_normal_base = base_brick[:3, 2]
            record.update({
                'pose_base': pose_dict(base_frame, base_brick),
                'surface_normal_base': np.round(surface_normal_base, 8).tolist(),
                'approach_direction_base': np.round(-surface_normal_base, 8).tolist(),
                'pregrasp_translation_base_m': np.round(
                    base_brick[:3, 3] + surface_normal_base * approach_clearance_m,
                    6).tolist(),
            })
        poses.append(record)

    output = {
        'schema': 'brick-poses-v1',
        'units': {'translation': 'metres', 'rotation': 'unitless'},
        'coordinate_convention_camera': 'RealSense optical: +x right, +y down, +z forward',
        'pose_semantics': 'brick origin is the box-centre camera ray intersected with the fitted top plane; +x follows the long axis; +z is the outward top-surface normal',
        'corner_semantics': 'camera_box corners 0-3 are the fitted top face and 4-7 are the bottom face directly below them',
        'dataset': quality['dataset'],
        'recording': quality['recording'],
        'frame': quality['frame'],
        'rgb': quality['rgb'],
        'camera_info': str(camera_info_path),
        'segmentation_source': str(segmentation_path),
        'classification_source': str(quality_path),
        'robot_base_poses_available': base_camera is not None,
        'missing_for_robot_execution': ([] if base_camera is not None else
                                        ['T_base_camera hand-eye calibration']),
        'approach_clearance_m': approach_clearance_m,
        'bricks': poses,
    }
    destination = quality_path.parent / output_name
    destination.write_text(json.dumps(output, indent=2))
    camera_boxes = {
        'schema': 'camera-frame-brick-boxes-v1',
        'frame_id': 'camera_color_optical_frame',
        'units': 'metres',
        'coordinate_convention': '+x right, +y down, +z forward',
        'dataset': quality['dataset'],
        'recording': quality['recording'],
        'frame': quality['frame'],
        'bricks': [
            {
                'id': record['id'],
                'quality': record['quality'],
                'colour': record['colour'],
                'classification_confidence': record['classification_confidence'],
                'center_pixel_xy': record['detection']['center_pixel_xy'],
                'corners_pixel_xy': record['detection']['corners_pixel_xy'],
                **record['camera_box'],
            }
            for record in poses
        ],
    }
    (quality_path.parent / 'camera_boxes.json').write_text(
        json.dumps(camera_boxes, indent=2))
    return output, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('classified_results', type=Path)
    parser.add_argument('--camera-pose', type=Path,
                        help='JSON containing frame and 4x4 T_base_camera')
    parser.add_argument('--output-name', default='brick_poses.json')
    parser.add_argument('--aggregate-name', default='all_brick_poses.json')
    parser.add_argument('--approach-clearance-m', type=float, default=.08)
    args = parser.parse_args()

    inputs = sorted(args.classified_results.rglob('quality_boxes.json'))
    if not inputs:
        raise SystemExit(f'No quality_boxes.json found under {args.classified_results}')
    repository_root = Path(__file__).resolve().parent.parent
    frames = []
    for index, quality_path in enumerate(inputs, 1):
        print(f'[{index}/{len(inputs)}] {quality_path}', flush=True)
        output, destination = export_frame(
            quality_path, args.output_name, args.camera_pose,
            args.approach_clearance_m, repository_root)
        frames.append({
            'dataset': output['dataset'], 'recording': output['recording'],
            'frame': output['frame'], 'brick_count': len(output['bricks']),
            'robot_base_poses_available': output['robot_base_poses_available'],
            'poses': str(destination),
        })
    aggregate = {
        'schema': 'brick-pose-index-v1',
        'frame_count': len(frames),
        'camera_pose_file': str(args.camera_pose) if args.camera_pose else None,
        'frames': frames,
    }
    (args.classified_results / args.aggregate_name).write_text(
        json.dumps(aggregate, indent=2))
    print(f'Wrote poses for {len(frames)} frames to {args.classified_results}')


if __name__ == '__main__':
    main()
