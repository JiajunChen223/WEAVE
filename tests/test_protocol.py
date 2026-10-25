import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from baselines.adapters.native import LEARNERS
from baselines.source import verify_source
from training.loop import learning_rate_at
from training.update_counter import UpdateCounter


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_learning_methods_match_public_comparison(self):
        comparison = json.loads((ROOT/'configs/experiments/main_comparison.json').read_text())
        traditional = {'cnmf', 'hysure', 'cstf'}
        self.assertEqual(set(LEARNERS), set(comparison['external_methods'])-traditional)
        self.assertEqual(len(comparison['external_methods']), 12)
        self.assertEqual(comparison['external_comparison_cells'], 48)

    def test_method_specific_training_settings_and_cosine_endpoint(self):
        for name in LEARNERS:
            recipe = json.loads((ROOT/f'configs/baselines/{name}.json').read_text())
            self.assertEqual(recipe['successful_optimizer_updates'], 10000)
            self.assertEqual(recipe['seed'], 0)
            self.assertEqual(recipe['effective_batch_size'], 1)
            self.assertEqual(recipe['checkpoint_selection'], 'last')
            phase = recipe['offline_phases'][0]
            self.assertAlmostEqual(learning_rate_at(recipe, 0), phase['optimizer']['lr'])
            expected = 1e-6 if name=='hsc_sampling' else phase['optimizer']['lr']
            self.assertAlmostEqual(learning_rate_at(recipe, 10000), expected)

    def test_cave_scales_are_separate_observation_geometries(self):
        for scale, filename in [(2,'cave_x2'), (4,'cave'), (8,'cave_x8')]:
            d = json.loads((ROOT/f'configs/datasets/{filename}.json').read_text())
            self.assertEqual(d['spatial_observation']['kernel_size'], 2*scale)
            self.assertEqual(d['spatial_observation']['sigma'], scale/2)
            self.assertEqual(d['spatial_observation']['phase'], [scale//2,scale//2])
            self.assertEqual(d['split_counts'], {'train':18,'validation':2,'test':12})

    def test_failed_update_does_not_spend_budget(self):
        counter = UpdateCounter(2)
        counter.record(False)
        counter.record(True)
        self.assertEqual((counter.completed, counter.skipped), (1,1))
        counter.record(True)
        self.assertTrue(counter.done)
        with self.assertRaises(ValueError):counter.record(True)

    def test_upstream_checkout_is_required_and_hash_checked(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            spec=json.loads((ROOT/'baselines/sources.json').read_text())['psrt']
            with patch('baselines.source.subprocess.run', return_value=SimpleNamespace(stdout='wrong')):
                with self.assertRaises(ValueError):verify_source('psrt',folder)
            with patch('baselines.source.subprocess.run', return_value=SimpleNamespace(stdout=spec['commit'])):
                with self.assertRaises(ValueError):verify_source('psrt',folder)


if __name__=='__main__':unittest.main()
