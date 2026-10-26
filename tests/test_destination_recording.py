"""Destination teaching records measured poses without commissioning motion."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from lego_sorter.core import ROOT

sys.path.insert(0, str(ROOT/'scripts'))
try:
    from record_destination import capture_record, save_record
finally:
    sys.path.pop(0)


class DestinationRecordingTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(base_frame='fr3_link0',ee_frame='fr3_hand_tcp',table_top_z=-.02,
                           workspace_x=[.2,.85],workspace_y=[-.5,.5])

    def record(self, name='place', position=None):
        return capture_record(name,position or [.6,0.,.3],[1.,0.,0.,0.],
                              [0.]*7,12345,dict(robot_mode=2),self.config)

    def test_xy_is_layer_centre_and_capture_height_is_not_release_height(self):
        record = self.record()
        self.assertEqual(record['meaning'],'centre_of_6x5_red_layer')
        self.assertEqual(record['xy_m'],[.6,0.])
        self.assertEqual(record['captured_tcp_xyz_m'][2],.3)
        self.assertEqual(record['known_table_support_z_m'],-.02)
        self.assertIsNone(record['release_tcp_z_m'])
        self.assertEqual(record['status'],'RECORDED_ONLY_NOT_COMMISSIONED')

    def test_both_destinations_and_previous_capture_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            output = Path(directory)
            save_record(output,self.record())
            save_record(output,self.record('discard',[.4,.3,.3]))
            path = save_record(output,self.record('place',[.61,0.,.3]))
            data = json.loads(path.read_text())
            self.assertEqual(len(data['history']),3)
            self.assertEqual(data['history'][0]['xy_m'],[.6,0.])
            self.assertEqual(data['destinations']['place']['xy_m'],[.61,0.])
            self.assertEqual(data['destinations']['discard']['xy_m'],[.4,.3])

    def test_invalid_or_out_of_workspace_positions_are_rejected(self):
        for position in ([.9,0.,.3],[.6,float('nan'),.3]):
            with self.subTest(position=position),self.assertRaises(ValueError):
                self.record(position=position)

    def test_observation_retains_full_pose_and_does_not_replace_destinations(self):
        record = self.record('observation',[.4,0.,.4])
        self.assertEqual(record['meaning'],'camera_observation_pose')
        self.assertEqual(record['observation_pose_xyz_yaw'],[.4,0.,.4,0.])
        self.assertEqual(record['tool_down_tilt_deg'],0.)
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            save_record(directory,self.record())
            path = save_record(directory,record)
            data = json.loads(path.read_text())
            self.assertEqual(data['destinations']['place']['xy_m'],[.6,0.])
            self.assertEqual(data['destinations']['observation']['captured_tcp_xyz_m'],[.4,0.,.4])

    def test_source_is_grid_centre_and_grasp_height_is_a_separate_measurement(self):
        source = self.record('source',[.4,0.,.3])
        self.assertEqual(source['meaning'],'centre_of_source_grid')
        grasp = self.record('grasp',[.4,0.,-.008])
        self.assertAlmostEqual(grasp['grasp_tcp_above_table_m'],.012)
        self.assertIsNone(grasp['release_tcp_z_m'])
        self.assertEqual(grasp['status'],'RECORDED_ONLY_NOT_COMMISSIONED')
