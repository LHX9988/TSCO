"""Check sensitivity transforms and frozen-policy evaluation invariants."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tsco.config import load_config
from tsco.data import Dataset
from test_grid import small_case
from tsco.simulation import RunResult
from tsco.studies import StudyRunner, dataset_fingerprint


class _Policy:
    def __init__(self, config):
        self.steps, self.updates, self.variant = 30, 4, "tsco"
        self.config = deepcopy(config)
        self.metadata = {"variant": "tsco", "normalization": {
            "feature_scales": {"nodes": [1.]*14, "edges": [1.]*2,
                               "task": [1.]*10, "global_features": [1.]*8},
            "reward_scales": {k: 1. for k in ("revenue", "cost", "carbon", "tardiness")},
            "feature_offsets": {"nodes": [0.]*14, "edges": [0.]*2,
                                "task": [0.]*10, "global_features": [0.]*8},
            "provenance": {"split": "train"},
        }}


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.config = load_config()
        self.config['cpn'].update(idle_mw=[.1, .1, .1], peak_mw=[1., 2., 1.],
                                  flops_per_unit_second=[10., 20., 30.])
        def fixture(split):
            state = {'start_interval': 0, 'commitment': np.array([1]),
                     'generation_mw': np.array([8.]), 'storage_energy_mwh': np.array([]),
                     'initial_status_hours': np.array([3.]), 'cpn_empty': True}
            plans = [{'start_interval': 0, 'hour_dt': 1.,
                      'demand_mw': np.array([[[0., 8.]]]),
                      'cpn_demand_mw': np.array([[[0., 2.]]]),
                      'renewable_mw': np.array([[[2.]]]), 'probabilities': np.array([1.]),
                      'storage_modes': np.empty((1, 1, 0), int)}]
            return Dataset(small_case(), np.tile([0., 8.], (12, 1)), np.full((12, 1), 2.),
                           np.arange(12)*300., [], 300., {}, '2024-01-01T00:00:00Z', split,
                           planning_scenarios=plans, initial_state=deepcopy(state),
                           initial_states=[state], storage_modes=np.empty((12, 0), int))
        self.training, self.test = fixture('train'), fixture('test')
        self.checkpoint = self.path/"policy.pt"
        self.checkpoint.write_bytes(b"UNIT TEST STUB: not a torch checkpoint")

    def tearDown(self):
        self.tmp.cleanup()

    def test_res_scaling_preserves_source_inputs(self):
        runner = StudyRunner(self.config, self.training, self.test, self.checkpoint)
        original = dataset_fingerprint(self.test)
        cfg, train, test, info = runner._setting("res_capacity", .6)
        np.testing.assert_allclose(train.renewable, self.training.renewable*.6)
        np.testing.assert_allclose(test.renewable, self.test.renewable*.6)
        self.assertAlmostEqual(test.case.renewable_capacity.sum(), 12.)
        np.testing.assert_allclose(test.planning_scenarios[0]["renewable_mw"], [[[1.2]]])
        self.assertAlmostEqual(self.test.case.renewable_capacity.sum(), 20.)
        self.assertEqual(dataset_fingerprint(self.test), original)
        self.assertEqual(cfg["reward"], self.config["reward"])

    def test_power_scale_is_explicit_not_resource_scale(self):
        runner = StudyRunner(self.config, self.training, self.test, self.checkpoint)
        cfg, _, test, _ = runner._setting("cpn_power_scale", 1.5)
        np.testing.assert_allclose(test.planning_scenarios[0]["demand_mw"], [[[0., 9.]]])
        np.testing.assert_allclose(self.test.planning_scenarios[0]["demand_mw"], [[[0., 8.]]])
        self.assertAlmostEqual(sum(cfg["cpn"]["peak_mw"]), 6.)
        self.assertEqual(cfg["cpn"]["capacities"], self.config["cpn"]["capacities"])
        self.assertEqual(cfg["cpn"]["flops_per_unit_second"], self.config["cpn"]["flops_per_unit_second"])
        self.assertAlmostEqual(cfg["cpn"]["site_peak_mw"], 6.)

    def test_power_scaling_requires_the_cpn_forecast_component(self):
        self.test.planning_scenarios[0]['cpn_demand_mw'] = None
        runner = StudyRunner(self.config, self.training, self.test, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'cpn_demand_mw'):
            runner._setting('cpn_power_scale', 1.5)

    def test_reward_grid_rejects_single_unchanged_checkpoint(self):
        runner = StudyRunner(self.config, self.training, self.test, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "separately trained"):
            runner.run_grid("reward_carbon", [.5, 1.], [0], self.path/"results", hours=1.)

    def test_reward_grid_rejects_wrong_trained_coefficients(self):
        path = self.path/"different.pt"
        path.write_bytes(b"SECOND UNIT TEST STUB")
        runner = StudyRunner(self.config, self.training, self.test, {".5": path})
        with patch.object(runner, "_load_policy", return_value=_Policy(self.config)):
            with self.assertRaisesRegex(ValueError, "reward weights"):
                runner.run_grid("reward_cost", [.5], [0], self.path/"results", hours=1.)

    def test_seed_grid_replays_the_same_inputs_and_records_unknown_budget(self):
        runner = StudyRunner(self.config, self.training, self.test, self.checkpoint)
        seen = []
        class FakeSimulation:
            def __init__(self, test, train, config, *args):
                self.config, self.test = config, test
                seen.append((config["simulation"]["seed"], dataset_fingerprint(test)))
            def run(self, policy, hours, explore):
                assert not explore
                return RunResult({"seed": self.config["simulation"]["seed"],
                                  "failed": False, "status": "completed", "horizon_hours": hours,
                                  "terminal_budget_violation": None,
                                  "operational_cost_kusd": 1., "emissions_t": 1.})
        with patch.object(runner, "_load_policy", return_value=_Policy(self.config)):
            with patch("tsco.studies.Simulator", FakeSimulation):
                records = runner.run_grid("carbon_budget", [1300., 1450.], [0, 1], self.path/"results", hours=1.)
        self.assertEqual(len(records), 4)
        self.assertEqual(len({h for _, h in seen}), 1)
        summary = json.loads((self.path/"results"/"summary.json").read_text())
        self.assertEqual(summary["1300"]["terminal_budget"]["n_observed"], 0)
        self.assertEqual(summary["1300"]["terminal_budget"]["n_unknown"], 2)
        self.assertTrue((self.path/"results"/"metrics.csv").exists())

    def test_evaluation_cannot_update_frozen_policy(self):
        runner = StudyRunner(self.config, self.training, self.test, self.checkpoint)
        class MutatingSimulation:
            def __init__(self, *args):
                pass
            def run(self, policy, **kwargs):
                policy.steps += 1
                return RunResult({})
        with patch.object(runner, "_load_policy", return_value=_Policy(self.config)):
            with patch("tsco.studies.Simulator", MutatingSimulation):
                with self.assertRaisesRegex(RuntimeError, "frozen policy"):
                    runner.run_grid("carbon_budget", [1450.], [0], self.path/"results", hours=1.)


if __name__ == "__main__":
    unittest.main()
