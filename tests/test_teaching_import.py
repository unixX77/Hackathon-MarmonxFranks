"""Taught coordinates affect explicit mock trials without enabling hardware."""
import copy
import unittest

from lego_sorter.core import ROOT, load_config
from lego_sorter.teaching import apply_mock_teachings


class TeachingImportTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'src/lego_sorter/config/loose_one_layer_mock.yaml')
        self.records = dict(destinations={
            name:dict(base_frame='fr3_link0',tcp_frame='fr3_hand_tcp',xy_m=xy)
            for name,xy in [('place',[.4,-.2]),('discard',[.7,.2]),('observation',[.4,0.])]})
        self.records['destinations']['observation'].update(
            observation_pose_xyz_yaw=[.4,0.,.4,.1],captured_tcp_quaternion_xyzw=[1.,0.,0.,0.])

    def test_import_preserves_original_and_uses_table_cells_for_discard(self):
        before = copy.deepcopy(self.config)
        result = apply_mock_teachings(self.config,self.records)
        self.assertEqual(self.config,before)
        self.assertEqual(result['observation_pose'],[.4,0.,.4,.1])
        self.assertEqual(result['destinations']['selected']['origin'],[.4,-.2,-.02])
        discard = result['destinations']['discard']
        self.assertEqual(discard['origin'],[.7,.2,-.02])
        self.assertNotIn('kind',discard)
        self.assertEqual((discard['rows'],discard['columns'],discard['layers']),(6,5,1))
        self.assertFalse(result['hardware_validated'])
        self.assertFalse(result['geometry_confirmed'])

    def test_missing_observation_and_wrong_frames_are_rejected(self):
        del self.records['destinations']['observation']
        with self.assertRaisesRegex(ValueError,'Missing taught observation'):
            apply_mock_teachings(self.config,self.records)
        self.records['destinations']['place']['base_frame']='other'
        with self.assertRaisesRegex(ValueError,'frame does not match'):
            apply_mock_teachings(self.config,self.records)

    def test_import_cannot_be_used_for_hardware(self):
        self.config['mode']='hardware'
        with self.assertRaisesRegex(ValueError,'mock trials only'):
            apply_mock_teachings(self.config,self.records)
