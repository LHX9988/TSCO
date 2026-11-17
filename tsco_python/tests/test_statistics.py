"""Run-level statistical inference and denominator/censoring checks."""
import json
import unittest
import numpy as np
from scipy.stats import t as student_t
from tsco.statistics import holm_adjust, paired_comparisons, summarize_runs


class StatisticsTests(unittest.TestCase):
    def test_empty_singleton_constant_and_missing(self):
        empty = summarize_runs([], ["cost"])
        self.assertEqual(empty["metrics"]["cost"]["n"], 0)
        self.assertIsNone(empty["metrics"]["cost"]["ci95"])
        one = summarize_runs([{"cost": 4}], ["cost"])["metrics"]["cost"]
        self.assertEqual(one["mean"], 4)
        self.assertIsNone(one["std"])
        constant = summarize_runs([{"cost": 4}, {"cost": 4}, {"cost": None},
                                   {"cost": np.nan}], ["cost"])["metrics"]["cost"]
        self.assertEqual(constant["n"], 2)
        self.assertEqual(constant["n_missing"], 2)
        self.assertEqual(constant["ci95"], 0)
        json.dumps(empty, allow_nan=False)

    def test_t_interval_is_half_width(self):
        values = np.arange(10, dtype=float)
        summary = summarize_runs([{"x": x} for x in values])["metrics"]["x"]
        expected = student_t.ppf(.975, 9) * np.std(values, ddof=1) / np.sqrt(10)
        self.assertAlmostEqual(summary["ci95"], expected)
        self.assertAlmostEqual(summary["ci95_high"], 4.5 + expected)

    def test_failed_diagnostics_and_budget_observations(self):
        runs = [
            {"cost": 10., "failed": False, "terminal_budget_violation": False},
            {"cost": 12., "failed": False, "terminal_budget_violation": True},
            {"cost": 1., "failed": True, "terminal_budget_violation": False},
            {"cost": None, "failed": False, "terminal_budget_violation": None,
             "horizon_hours": 2, "emissions_t": 1, "carbon_budget_t": 1450},
        ]
        result = summarize_runs(runs, ["cost"])
        self.assertEqual(result["n_failed"], 1)
        self.assertEqual(result["metrics"]["cost"]["n_failed_contributing"], 1)
        self.assertEqual(result["metrics"]["cost"]["n"], 3)
        self.assertEqual(result["terminal_budget"]["n_observed"], 2)
        self.assertEqual(result["terminal_budget"]["n_violating"], 1)
        self.assertEqual(result["terminal_budget"]["violation_rate"], .5)

    def test_unknown_budget_does_not_infer_from_partial_emissions(self):
        result = summarize_runs([{"horizon_hours": 168, "emissions_t": 2000,
                                  "carbon_budget_t": 1450, "terminal_budget_violation": None}])
        self.assertEqual(result["terminal_budget"]["n_observed"], 0)
        inferred = summarize_runs([{"horizon_hours": 168, "emissions_t": 2000,
                                    "carbon_budget_t": 1450}])
        self.assertEqual(inferred["terminal_budget"]["n_violating"], 1)

    def test_seed_alignment_and_baseline_denominator(self):
        a = [{"seed": 0, "cost": 8}, {"seed": 1, "cost": 9}]
        b = [{"seed": 1, "cost": 12}, {"seed": 0, "cost": 10}]
        result = paired_comparisons(a, b, ["cost"])["comparisons"]["baseline"]["cost"]
        self.assertAlmostEqual(result["relative_reduction_pct"]["mean"], 22.5)
        self.assertAlmostEqual(result["ratio_of_means_reduction_pct"], 100 * (11 - 8.5) / 11)
        self.assertEqual(result["difference_tsco_minus_baseline"]["mean"], -2.5)
        self.assertNotEqual(result["relative_reduction_pct"]["mean"], result["ratio_of_means_reduction_pct"])

    def test_zero_baseline_guard_and_pair_failures(self):
        a = [{"seed": 0, "x": 1}, {"seed": 1, "x": 8}, {"seed": 2, "x": 2, "failed": True}]
        b = [{"seed": 0, "x": 0}, {"seed": 1, "x": 10}, {"seed": 2, "x": 10}]
        result = paired_comparisons(a, b, ["x"])["comparisons"]["baseline"]["x"]
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["n_excluded_failed"], 1)
        self.assertEqual(result["n_zero_baseline_excluded"], 1)
        self.assertEqual(result["relative_reduction_pct"]["mean"], 20)

    def test_seed_mismatch_and_duplicates_raise(self):
        with self.assertRaises(ValueError):
            paired_comparisons([{"seed": 0}], [{"seed": 1}], ["x"])
        with self.assertRaises(ValueError):
            paired_comparisons([{"seed": 0}, {"seed": 0}], [{"seed": 0}], ["x"])
        with self.assertRaises(ValueError):
            paired_comparisons([{"x": 1}], [{"x": 2}], ["x"])

    def test_holm_monotonicity_and_fixed_family(self):
        p = [.01, .04, .03, .005]
        adjusted = holm_adjust(p)
        np.testing.assert_allclose(adjusted, [.03, .06, .06, .02])
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(np.asarray(adjusted)[order]) >= 0))
        self.assertEqual(holm_adjust([.01, None]), [.02, None])

    def test_multiple_baselines_correct_within_metric_and_welch_option(self):
        a = [{"seed": i, "x": float(i)} for i in range(4)]
        b = [{"seed": i, "x": float(i + 1)} for i in range(4)]
        c = [{"seed": i, "x": float(2 * i)} for i in range(4)]
        out = paired_comparisons(a, {"b": b, "c": c}, ["x"], method="paired")
        self.assertEqual(out["comparisons"]["b"]["x"]["p_raw"], 0)
        self.assertTrue(out["comparisons"]["b"]["x"]["zero_variance_nonzero_difference"])
        json.dumps(out, allow_nan=False)
        welch = paired_comparisons(a, b, ["x"])
        self.assertEqual(welch["method"], "welch")
        self.assertGreater(welch["comparisons"]["baseline"]["x"]["p_raw"], 0)


if __name__ == "__main__":
    unittest.main()
