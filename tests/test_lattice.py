"""Geometry of the requested stud lattice; these tests do not model stud forces."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from lego_sorter.core import ROOT, load_config, layout, bounds, overlap, supported, Store


class LatticeTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / 'src/lego_sorter/config/lattice_red.yaml')
        self.slots = layout(self.c['source'], self.c['block_size'], 'source')

    def test_count_footprints_and_centered_layers(self):
        self.assertEqual(len(self.slots), 300)
        for layer in range(10):
            cells = self.slots[layer * 30:(layer + 1) * 30]
            boxes = [bounds(b) for b in cells]
            size = [max(b[2] for b in boxes) - min(b[0] for b in boxes),
                    max(b[3] for b in boxes) - min(b[1] for b in boxes)]
            np.testing.assert_allclose(size, [.176, .160] if layer % 2 == 0 else [.160, .176])
            np.testing.assert_allclose(np.mean([b['centre'][:2] for b in cells], axis=0), self.c['source']['origin'][:2], atol=1e-12)
            self.assertAlmostEqual(cells[0]['yaw'], math.pi / 2 + layer % 2 * math.pi / 2)
        # Five touching 4-stud bricks along each row; 2-stud air gaps across rows.
        self.assertAlmostEqual(self.slots[1]['centre'][1] - self.slots[0]['centre'][1], .032)
        self.assertAlmostEqual(abs(self.slots[5]['centre'][0] - self.slots[0]['centre'][0]) - .016, .016)

    def test_two_end_contacts_interior_and_outer_rows(self):
        area = np.prod(self.c['block_size'][:2])
        for layer in range(1, 10):
            lower = self.slots[(layer - 1) * 30:layer * 30]
            for b in lower:
                b['state'] = 'PLACED'
            for upper in self.slots[layer * 30:(layer + 1) * 30]:
                # Each end contacts one lower row, sometimes spanning a seam between bricks.
                contacts = {}
                for b in lower:
                    a = overlap(bounds(upper), bounds(b))
                    if a > 1e-10:
                        contacts[b['row']] = contacts.get(b['row'], 0) + a
                self.assertEqual(len(contacts), 2)
                fraction = .25 if upper['row'] in (0, 5) else .5
                np.testing.assert_allclose(list(contacts.values()), [area * fraction / 2] * 2)
                self.assertTrue(supported(upper, self.slots, .25))
                if fraction == .25:
                    self.assertFalse(supported(upper, self.slots, .5))

    def test_selected_layout_matches_source_and_routes_only_red(self):
        selected = layout(self.c['destinations']['selected'], self.c['block_size'], 'selected')
        for a, b in zip(self.slots, selected):
            self.assertEqual(a['yaw'], b['yaw'])
            np.testing.assert_allclose(np.array(a['centre']) - self.c['source']['origin'],
                                       np.array(b['centre']) - self.c['destinations']['selected']['origin'])
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
            c = copy.deepcopy(self.c)
            c['runtime'] = directory
            store = Store(c)
            try:
                for colour, confidence, expected in [('red', 'high', 'selected'), ('dark_red', 'high', 'discard'), ('red', 'low', 'discard')]:
                    slot = store.reserve(dict(colour=colour, confidence=confidence))
                    self.assertEqual(slot['destination'], expected)
                    store.transition(slot['id'], 'EMPTY')
            finally:
                store.close()

    def test_hardware_refuses_stud_connected_lattice(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
            c = copy.deepcopy(self.c)
            c.update(mode='hardware', runtime=directory, hardware_validated=True)
            path = Path(directory) / 'hardware.yaml'
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, 'extraction and seating'):
                load_config(path)

    def test_trash_uses_repeated_release_point_with_bounded_capacity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
            c = copy.deepcopy(self.c)
            c['runtime'] = directory
            c['destinations']['discard']['capacity'] = 2
            store = Store(c)
            try:
                drops = []
                for _ in range(2):
                    slot = store.reserve(dict(colour='dark_red', confidence='high'))
                    drops.append(slot)
                    store.transition(slot['id'], 'PLACED', 'SIMULATED disposal receipt')
                self.assertEqual(drops[0]['centre'], drops[1]['centre'])
                self.assertNotEqual(drops[0]['id'], drops[1]['id'])
                with self.assertRaisesRegex(RuntimeError, 'capacity for discard'):
                    store.reserve(dict(colour='unknown', confidence='low'))
            finally:
                store.close()

    def test_trash_hardware_is_not_enabled_by_removing_stud_flags(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
            c = copy.deepcopy(self.c)
            c.update(mode='hardware', runtime=directory)
            c['source']['stud_connected'] = False
            c['destinations']['selected']['stud_connected'] = False
            path = Path(directory) / 'hardware.yaml'
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, 'Trash-box drops are currently mock-only'):
                load_config(path)
