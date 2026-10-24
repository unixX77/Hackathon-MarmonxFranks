#!/usr/bin/env python3
"""On-demand fresh RGB-D observation; teammate SAM3 pipeline stays loaded."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from std_srvs.srv import Trigger
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
import yaml

from vision_core import BASE, matrix, normalize_bricks, pose_dict, validate_config


class Perception:
    def __init__(self, node, config):
        self.node, self.c = node, config
        self.lock = threading.Lock()
        self.latest = None
        self.busy = threading.Lock()
        self.mock_count = 0
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)
        self.group = ReentrantCallbackGroup()
        self.service = node.create_service(Trigger, 'vision/observe', self.observe, callback_group=self.group)
        if config['backend'] == 'sam3':
            self.load_model()
            from cv_bridge import CvBridge
            from message_filters import Subscriber, ApproximateTimeSynchronizer
            self.bridge = CvBridge()
            topics = config['topics']
            self.rgb = Subscriber(node, Image, topics['rgb'], qos_profile=qos_profile_sensor_data,
                                  callback_group=self.group)
            self.depth = Subscriber(node, Image, topics['aligned_depth'], qos_profile=qos_profile_sensor_data,
                                    callback_group=self.group)
            self.info = Subscriber(node, CameraInfo, topics['camera_info'], qos_profile=qos_profile_sensor_data,
                                   callback_group=self.group)
            self.sync = ApproximateTimeSynchronizer([self.rgb, self.depth, self.info], 10, .03)
            self.sync.registerCallback(self.capture)

    def load_model(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'vision_vendor'))
        import torch
        from transformers import Sam3Model, Sam3Processor
        from segment_lego_sam3_single_prompt import segment
        from top_layer_depth import filter_top
        from top_layer_bricks import apply_size_prior
        self.segment, self.filter_top, self.fit = segment, filter_top, apply_size_prior
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        name = self.c['perception']['model']
        self.processor = Sam3Processor.from_pretrained(name)
        self.model = Sam3Model.from_pretrained(name, torch_dtype=self.dtype).to(self.device).eval()

    def camera_transform(self, stamp):
        ext = self.c['camera_extrinsic']
        tf = self.tf.lookup_transform(BASE, ext['parent_frame'], stamp, timeout=Duration(seconds=.2)).transform
        parent = matrix({'xyz_m': [tf.translation.x, tf.translation.y, tf.translation.z],
                         'quaternion_xyzw': [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]})
        return parent @ matrix(ext)

    def capture(self, rgb, depth, info):
        # Keep the transform associated with this capture, before slow inference.
        try:
            frame = self.c['camera_extrinsic']['optical_frame']
            if any(msg.header.frame_id != frame for msg in (rgb, depth, info)):
                raise ValueError(f'RGB/aligned depth/intrinsics must all use {frame}.')
            if depth.encoding != '16UC1':
                raise ValueError('Expected aligned 16UC1 depth; configure its scale explicitly.')
            t = Time.from_msg(rgb.header.stamp)
            transform = self.camera_transform(t)
            with self.lock:
                self.latest = (t.nanoseconds, time.monotonic(), rgb, depth, info, transform)
        except Exception as exc:
            self.node.get_logger().warning(f'Capture rejected: {exc}', throttle_duration_sec=5.)

    def infer(self, captured):
        stamp, _, rgb, depth, info, transform = captured
        rgb_array = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
        depth_array = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        if depth_array.shape != rgb_array.shape[:2] or info.width != rgb.width or info.height != rgb.height:
            raise RuntimeError('RGB, aligned depth and CameraInfo dimensions differ.')
        if any(abs(v) > 1e-8 for v in info.d):
            raise RuntimeError('Teammate plane geometry requires rectified/zero-distortion images.')
        k = list(info.k)
        calibration = {'depth_scale_m': self.c['perception']['depth_scale_m'],
                       'aligned_depth_intrinsics': {'K': np.array(k).reshape(3, 3).tolist(),
                         'width': info.width, 'height': info.height, 'fx': k[0], 'fy': k[4],
                         'ppx': k[2], 'ppy': k[5], 'distortion_coeffs': list(info.d)}}
        p = self.c['perception']
        args = SimpleNamespace(prompt=p['prompt'], threshold=p['min_score'], mask_threshold=.5,
            max_overlap=.5, min_area=100, min_rect=0., min_rel_area=0., max_colour_dist=float('inf'),
            brick_size_mm=[v*1000 for v in self.c['brick_size_m']], size_tol=p['size_relative_tolerance'],
            block_height_mm=self.c['brick_size_m'][2]*1000, table_roi=p['table_roi'],
            plane_threshold_mm=3., erode=2, min_depth_pixels=30, min_valid_fraction=p['min_valid_fraction'])
        _, instance, records, _, _ = self.segment(rgb_array, self.model, self.processor, args, self.device, self.dtype)
        _, report = self.filter_top(instance, depth_array, calibration, args)
        kept = [r for r in report['instances'] if r['kept']]
        _, bricks = self.fit(instance, kept, records, calibration, report, args)
        sam = {r['id']: r for r in records}
        metrics = {r['id']: r for r in report['instances']}
        for b in bricks:
            ident = b['source_instance']
            b.update({k: sam[ident][k] for k in ('score', 'touches_border')})
            b.update({k: metrics[ident][k] for k in ('valid_fraction', 'height_spread_m')})
        obj = normalize_bricks(bricks, report, transform, self.c)
        return stamp, obj

    def observe(self, request, response):
        if not self.busy.acquire(blocking=False):
            response.success, response.message = False, 'Observation already in progress.'
            return response
        try:
            requested = self.node.get_clock().now().nanoseconds
            if self.c['backend'] == 'mock':
                stamp = requested
                obj = pose_dict(matrix(self.c['mock']['object_base']))
                if self.mock_count == 0:
                    obj['xyz_m'] = (np.asarray(obj['xyz_m']) + np.asarray(self.c['mock'].get('initial_bias_m', [0., 0., 0.]))).tolist()
                self.mock_count += 1
                obj.update(colour='red', size_m=self.c['brick_size_m'], source_id=1)
            else:
                deadline = time.monotonic()+self.c['perception']['capture_timeout_s']
                captured = None
                while time.monotonic() < deadline:
                    with self.lock:
                        sample = self.latest
                    if sample is not None and sample[0] >= requested and time.monotonic()-sample[1] < .5:
                        captured = sample
                        break
                    time.sleep(.01)
                if captured is None:
                    raise RuntimeError('No fresh synchronized RGB-D/TF after observation request.')
                stamp, obj = self.infer(captured)
            response.message = json.dumps({'schema': 1, 'backend': self.c['backend'],
                'frame_id': BASE, 'capture_stamp_ns': stamp, 'objects': [obj]})
            response.success = True
        except Exception as exc:
            response.success, response.message = False, str(exc)
        finally:
            self.busy.release()
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args, ros_args = parser.parse_known_args()
    try:
        with open(args.config, encoding='utf-8') as stream:
            config = validate_config(yaml.safe_load(stream))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Configuration incomplete/invalid: {exc}')
    rclpy.init(args=ros_args)
    node = rclpy.create_node('fr3_vision_perception')
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        worker = Perception(node, config)
        node.get_logger().info(f"Observation service ready; backend={config['backend']}")
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
