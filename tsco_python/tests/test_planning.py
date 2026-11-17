"""Small certified checks of the decomposition."""
import unittest
import numpy as np

from tsco.schema import GridCase
from tsco.planning import BendersPlanner


def tiny_case(network_limit=100.0, storage=False):
    a = lambda values: np.asarray(values, dtype=float)
    i = lambda values: np.asarray(values, dtype=int)
    return GridCase('planning-test', 100.0, a([0, 30]), i([0]), i([1]), a([.1]), a([1]), a([network_limit]),
                    i([0, 1]), a([5, 0]), a([40, 30]), a([100, 100]), a([.01, .02]), a([10, 30]), a([2, 1]),
                    a([.8, .2]), a([15, 5]), a([2, 1]), i([1] if storage else []),
                    a([10] if storage else []), a([20] if storage else []), a([.9] if storage else []),
                    a([.9] if storage else []), a([2] if storage else []), a([18] if storage else []),
                    i([1]), a([50]), i([1]))


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'solver': {'backend': 'scipy', 'mip_gap': 1e-8, 'pwl_segments': 4, 'time_limit': 30,
                               'feasibility_tol': 1e-7, 'optimality_tol': 1e-7,
                               'startup_ramp_mw': [40, 30], 'shutdown_ramp_mw': [40, 30]},
                    'simulation': {'carbon_price': 10},
                    'planning': {'relative_gap': 1e-7, 'max_iterations': 100, 'terminal_storage': True,
                                 'time_limit_seconds': 60, 'min_up_hours': 1, 'min_down_hours': 1}}

    def inputs(self, storage=False):
        demand = np.array([[[0, 25], [0, 45], [0, 30]], [[0, 35], [0, 35], [0, 45]]], float)
        res = np.array([[[5], [10], [5]], [[2], [5], [8]]], float)
        return demand, res, np.array([.5, .5]), np.array([0, 0]), np.array([0., 0.]), np.array([10.] if storage else [])

    def test_benders_matches_extensive_and_valid_bounds(self):
        planner = BendersPlanner(tiny_case(network_limit=20), self.cfg)
        try:
            reference = planner.solve_extensive(*self.inputs(), initial_status_hours=np.full(2, 24.0))
            actual = planner.solve(*self.inputs(), initial_status_hours=np.full(2, 24.0))
            self.assertTrue(reference.converged, reference.status)
            self.assertTrue(actual.converged, actual.status)
            self.assertAlmostEqual(reference.objective, actual.objective, places=4)
            self.assertLessEqual(actual.lower_bound, reference.objective + 1e-5)
            self.assertGreaterEqual(actual.objective, reference.objective - 1e-5)
            self.assertTrue(any(h['optimality_cuts'] for h in actual.history))
        finally: planner.close()

    def test_phase_one_feasibility_cut(self):
        # Generator 0 cannot reach bus 1 through the tight line; the aggregate
        # capacity constraint alone does not capture this network bottleneck.
        cfg = {**self.cfg, 'simulation': {'carbon_price': 0}}
        planner = BendersPlanner(tiny_case(network_limit=5), cfg)
        d = np.array([[[0, 25], [0, 25]]], float)
        r = np.zeros((1, 2, 1))
        args = d, r, np.array([1.]), np.zeros(2), np.zeros(2), np.empty(0)
        try:
            result = planner.solve(*args, initial_status_hours=np.full(2, 24.0))
            ref = planner.solve_extensive(*args, initial_status_hours=np.full(2, 24.0))
            self.assertTrue(result.converged, result.status)
            self.assertAlmostEqual(result.objective, ref.objective, places=4)
            self.assertTrue(any(step['feasibility_cuts'] > 0 for step in result.history))
        finally: planner.close()

    def test_storage_energy_and_extensive_agreement(self):
        planner = BendersPlanner(tiny_case(storage=True), self.cfg)
        d, r, prob, u, p, e = self.inputs(storage=True)
        r[:, 0, 0] = 50  # surplus hour charges, later net-load hours discharge
        modes = np.array([[[1], [-1], [-1]], [[1], [-1], [-1]]])
        try:
            result = planner.solve(d, r, prob, u, p, e, storage_modes=modes, initial_status_hours=np.full(2, 24.0))
            reference = planner.solve_extensive(d, r, prob, u, p, e, storage_modes=modes, initial_status_hours=np.full(2, 24.0))
            self.assertTrue(result.converged, result.status)
            self.assertAlmostEqual(result.objective, reference.objective, places=4)
        finally: planner.close()

    def test_iteration_limit_is_not_convergence(self):
        cfg = {**self.cfg, 'planning': {**self.cfg['planning'], 'max_iterations': 1}}
        planner = BendersPlanner(tiny_case(), cfg)
        try:
            result = planner.solve(*self.inputs(), initial_status_hours=np.full(2, 24.0))
            self.assertFalse(result.converged)
            self.assertEqual(result.status, 'iteration_limit')
        finally: planner.close()

    def test_rollover_commitment_avoids_duplicate_startup_cost(self):
        planner = BendersPlanner(tiny_case(), self.cfg)
        demand = np.array([[[0, 20], [0, 20]]], float)
        renewable = np.zeros((1, 2, 1))
        try:
            cold = planner.solve(demand, renewable, np.array([1.]), np.zeros(2), np.zeros(2), np.empty(0), initial_status_hours=np.full(2, 24.0))
            warm = planner.solve(demand, renewable, np.array([1.]), np.array([1, 0]), np.array([20., 0.]), np.empty(0), initial_status_hours=np.full(2, 24.0))
            self.assertTrue(cold.converged and warm.converged)
            self.assertAlmostEqual(cold.startup_cost, 15.0)
            self.assertAlmostEqual(warm.startup_cost, 0.0)
            self.assertAlmostEqual(cold.objective - warm.objective, 15.0)
        finally: planner.close()

    def test_nonfinite_initial_conditions_rejected(self):
        planner = BendersPlanner(tiny_case(storage=True), self.cfg)
        d, r, prob, u, p, energy = self.inputs(storage=True)
        try:
            with self.assertRaises(ValueError):
                planner.solve(d, r, prob, u, np.array([np.nan, 0.]), energy, initial_status_hours=np.full(2, 24.0))
            with self.assertRaises(ValueError):
                planner.solve(d, r, prob, u, p, np.array([np.nan]), initial_status_hours=np.full(2, 24.0))
        finally: planner.close()

    def test_missing_physical_inputs_are_not_inferred(self):
        planner = BendersPlanner(tiny_case(storage=True), self.cfg)
        d, r, prob, u, p, energy = self.inputs(storage=True)
        try:
            with self.assertRaisesRegex(ValueError, 'storage_modes'):
                planner.solve(d, r, prob, u, p, energy, initial_status_hours=np.full(2, 24.0))
            with self.assertRaisesRegex(ValueError, 'initial generator'):
                planner.solve(d, r, prob, u, None, energy, initial_status_hours=np.full(2, 24.0))
        finally: planner.close()

    def test_missing_minimum_up_time_is_not_defaulted(self):
        cfg = {**self.cfg, 'planning': {**self.cfg['planning'], 'min_up_hours': None}}
        planner = BendersPlanner(tiny_case(), cfg)
        try:
            with self.assertRaisesRegex(ValueError, 'min_up_hours'):
                planner.solve(*self.inputs(), initial_status_hours=np.full(2, 24.0))
        finally: planner.close()

    def test_outer_limits_can_be_absent(self):
        cfg = {**self.cfg, 'planning': {**self.cfg['planning'], 'max_iterations': None,
                                      'time_limit_seconds': None}}
        planner = BendersPlanner(tiny_case(), cfg)
        try:
            result = planner.solve(*self.inputs(), initial_status_hours=np.full(2, 24.0))
            self.assertTrue(result.converged, result.status)
        finally: planner.close()

    def test_initial_down_time_obligation_is_enforced(self):
        cfg = {**self.cfg, 'planning': {**self.cfg['planning'], 'min_down_hours': [2, 0]}}
        planner = BendersPlanner(tiny_case(), cfg)
        demand = np.array([[[0, 20]]], float)
        renewable = np.zeros((1, 1, 1))
        try:
            fresh_off = planner.solve(demand, renewable, np.array([1.]), np.zeros(2), np.zeros(2),
                                      np.empty(0), initial_status_hours=np.array([0., 24.]))
            rested_off = planner.solve(demand, renewable, np.array([1.]), np.zeros(2), np.zeros(2),
                                       np.empty(0), initial_status_hours=np.array([2., 24.]))
            self.assertTrue(fresh_off.converged and rested_off.converged)
            self.assertEqual(fresh_off.commitment[0, 0], 0)
            self.assertEqual(rested_off.commitment[0, 0], 1)
        finally: planner.close()

    def test_initial_status_duration_is_required(self):
        planner = BendersPlanner(tiny_case(), self.cfg)
        try:
            with self.assertRaisesRegex(ValueError, 'initial_status_hours'):
                planner.solve(*self.inputs())
        finally: planner.close()

    def test_missing_transition_ramp_is_not_replaced_with_pmax(self):
        cfg = {**self.cfg, 'solver': {**self.cfg['solver'], 'startup_ramp_mw': None}}
        planner = BendersPlanner(tiny_case(), cfg)
        try:
            with self.assertRaisesRegex(ValueError, 'startup_ramp_mw'):
                planner.solve(*self.inputs(), initial_status_hours=np.full(2, 24.0))
        finally: planner.close()

    def test_explicit_startup_ramp_can_make_first_hour_infeasible(self):
        cfg = {**self.cfg, 'solver': {**self.cfg['solver'], 'startup_ramp_mw': [5, 5]}}
        planner = BendersPlanner(tiny_case(), cfg)
        try:
            demand = np.array([[[0, 20]]], float)
            result = planner.solve(demand, np.zeros((1, 1, 1)), np.array([1.]),
                                   np.zeros(2), np.zeros(2), np.empty(0),
                                   initial_status_hours=np.full(2, 24.0))
            self.assertFalse(result.converged)
            self.assertEqual(result.commitment.shape[0], 0)
        finally: planner.close()


if __name__ == '__main__': unittest.main()
