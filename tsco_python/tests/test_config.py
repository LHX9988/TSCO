"""Configuration serialization and empty-input checks."""
import json
from pathlib import Path
import unittest
from tsco.config import DEFAULTS, load_config, missing_fields, OPTIONAL_EMPTY, validate_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_nominal_file_matches_parameters(self):
        self.assertEqual(json.loads((ROOT/'configs/nominal.json').read_text()), DEFAULTS)

    def test_missing_values_stay_empty(self):
        config = load_config(ROOT/'configs/nominal.json')
        self.assertIsNone(config['cpn']['idle_mw'])
        self.assertIsNone(config['cpn']['flops_per_unit_second'])
        self.assertIsNone(config['planning']['min_up_hours'])
        self.assertIsNone(config['drl']['per_beta_steps'])
        self.assertNotIn('provenance', config)
        with self.assertRaisesRegex(ValueError, 'Missing configuration fields'):
            validate_config(config, require_complete=True)

    def test_known_values_and_limits(self):
        config = load_config()
        self.assertEqual(config['cpn']['capacities'], [128, 64, 32])
        self.assertEqual(config['solver']['backend'], 'gurobi')
        self.assertEqual(config['drl']['max_steps'], 2000000)
        missing = missing_fields(config, optional=OPTIONAL_EMPTY)
        self.assertNotIn('solver.time_limit', missing)
        self.assertIn('solver.startup_ramp_mw', missing)

    def test_explicit_empty_lists_are_valid(self):
        self.assertEqual(missing_fields({'predecessor_ids': []}), [])
        self.assertEqual(missing_fields({'predecessor_ids': None}), ['predecessor_ids'])


if __name__ == '__main__':
    unittest.main()
