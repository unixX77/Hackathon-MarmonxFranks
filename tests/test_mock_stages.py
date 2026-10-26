"""Small-count missions and stop-before-stage behavior without hardware."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from lego_sorter.cli import run_mock
from lego_sorter.core import ROOT, load_config


class MockStageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT/'runtime', prefix='test-mock-stages-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.c = load_config(ROOT/'src/lego_sorter/config/loose_one_layer_mock.yaml')
        self.c.update(runtime=str(self.path), mock_brick_count=1)

    def read(self, name):
        return json.loads((self.path/name).read_text())

    def run_trial(self, reader=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_mock(self.c, step=reader is not None, reader=reader)

    def test_three_bricks_complete_with_correct_colour_routing(self):
        self.c['mock_brick_count'] = 3
        result = self.run_trial()
        self.assertEqual((result['status'], result['completed']), ('COMPLETE', 3))
        placed = [s for s in self.read('slots.json')['slots'] if s['state']=='PLACED']
        self.assertEqual(sum(s['destination']=='selected' for s in placed), 1)
        self.assertEqual(sum(s['destination']=='discard' for s in placed), 2)
        self.assertEqual(self.read('mock_world.json')['bricks'], [])

    def test_one_brick_pauses_before_each_action_and_verification(self):
        reader = Mock(return_value='')
        result = self.run_trial(reader)
        self.assertEqual((result['status'], result['completed']), ('COMPLETE', 1))
        # 13 stages in the transaction, then observation move/capture to prove empty.
        self.assertEqual(reader.call_count, 15)
        self.assertEqual(self.read('hand.json')['hand'], 'EMPTY')

    def test_quit_before_grasp_leaves_brick_and_empty_hand(self):
        reader = Mock(side_effect=['']*5 + ['q'])
        result = self.run_trial(reader)
        self.assertEqual(result['status'], 'STOPPED')
        self.assertEqual(len(self.read('mock_world.json')['bricks']), 1)
        self.assertIsNone(self.read('mock_world.json')['held'])
        self.assertEqual(self.read('hand.json')['hand'], 'EMPTY')
        self.assertFalse(any(s['state'] in ('RESERVED','UNCERTAIN') for s in self.read('slots.json')['slots']))

    def test_quit_before_lift_preserves_held_brick_and_uncertain_destination(self):
        reader = Mock(side_effect=['']*6 + ['q'])
        result = self.run_trial(reader)
        self.assertEqual(result['status'], 'STOPPED')
        self.assertIsNotNone(self.read('mock_world.json')['held'])
        self.assertEqual(self.read('hand.json')['hand'], 'HOLDING')
        self.assertEqual(sum(s['state']=='UNCERTAIN' for s in self.read('slots.json')['slots']), 1)

    def test_eof_stops_without_performing_first_stage(self):
        result = self.run_trial(Mock(side_effect=EOFError))
        self.assertEqual(result['status'], 'STOPPED')
        self.assertEqual(self.read('mock_world.json')['steps'], 0)

    def test_invalid_count_is_rejected(self):
        self.c['mock_brick_count'] = 31
        with self.assertRaisesRegex(ValueError, 'between 1 and 30'):
            self.run_trial()
