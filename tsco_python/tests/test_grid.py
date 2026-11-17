"""Physical and dimensional invariants; no paper outcomes are hard-coded."""
import unittest
import numpy as np
from scipy import sparse
from tsco.schema import GridCase
from tsco.grid import EconomicDispatch
from tsco.optimization import LinearSolver


def small_case(storage=False):
    a = lambda v: np.array(v, dtype=float)
    i = lambda v: np.array(v, dtype=int)
    return GridCase(
        name='two-bus unit fixture', base_mva=100.0, base_load=a([0, 8]),
        branch_from=i([0]), branch_to=i([1]), branch_x=a([0.1]),
        branch_tap=a([1]), branch_limit=a([10]),
        gen_bus=i([0]), gen_pmin=a([0]), gen_pmax=a([100]),
        gen_ramp=a([120]), gen_a=a([0]), gen_b=a([10]), gen_c=a([0]),
        gen_emission=a([0.8]), gen_startup=a([1]), gen_shutdown=a([1]),
        storage_bus=i([1] if storage else []), storage_power=a([100] if storage else []),
        storage_capacity=a([20] if storage else []),
        storage_eta_c=a([0.9] if storage else []), storage_eta_d=a([0.9] if storage else []),
        storage_min=a([0] if storage else []), storage_max=a([20] if storage else []),
        renewable_bus=i([1]), renewable_capacity=a([20]), cpn_bus=i([1]))


def solver_config():
    return {'solver': {'backend': 'scipy', 'feasibility_tol': 1e-7,
                       'optimality_tol': 1e-7, 'int_feas_tol': 1e-6, 'mip_gap': 1e-4,
                       'threads': 8, 'ed_threads': 1, 'ed_method': 1, 'lp_warm_start': 2,
                       'pwl_segments': 20, 'mci_delta_mw': .1, 'time_limit': None,
                       'startup_ramp_mw': 100., 'shutdown_ramp_mw': 100.},
            'simulation': {'carbon_price': 50.},
            'planning': {'terminal_storage': 'free'}}


class GridTests(unittest.TestCase):
    def setUp(self):
        self.cfg = solver_config()
        self.case = small_case()
        self.ed = EconomicDispatch(self.case, self.cfg)

    def test_balance_cost_lmp_and_finite_difference_units(self):
        result = self.ed.solve([0, 8], [0], [1], [8], [], 1/12, previous_commitment=[1])
        self.assertTrue(result.feasible, result.status)
        self.assertAlmostEqual(result.p_gen[0], 8)
        self.assertAlmostEqual(result.cost_rate, 400)
        self.assertAlmostEqual(result.emission_rate, 6.4)
        self.assertAlmostEqual(result.lmp[1], 50)
        for dt in [1/12, 1.0]:
            mci = self.ed.mci([0, 8], [0], [1], [8], [], dt, buses=[1], previous_commitment=[1])
            self.assertAlmostEqual(mci[1], 0.8, places=7)

    def test_true_network_headroom_and_ramp(self):
        headroom = self.ed.headrooms([0, 8], [0], [1], [8], [], 1/12, buses=[1], previous_commitment=[1])
        self.assertAlmostEqual(headroom[1], 2)
        self.case.gen_ramp[:] = 1
        headroom = self.ed.headrooms([0, 8], [0], [1], [8], [], 1/12, buses=[1], previous_commitment=[1])
        self.assertAlmostEqual(headroom[1], 1/12)

    def test_infeasibility_and_marginal_undefined(self):
        result = self.ed.solve([0, 11], [0], [1], [8], [], 1/12, previous_commitment=[1])
        self.assertFalse(result.feasible)
        marginal = self.ed.mci([0, 10], [0], [1], [8], [], 1/12, buses=[1], previous_commitment=[1])
        self.assertTrue(np.isnan(marginal[1]))

    def test_open_line_does_not_fake_connectivity(self):
        self.case.branch_limit[:] = 0
        result = self.ed.solve([0, 8], [0], [1], [8], [], 1/12, previous_commitment=[1])
        self.assertFalse(result.feasible)
        result = self.ed.solve([0, 8], [8], [0], [0], [], 1/12, previous_commitment=[0])
        self.assertTrue(result.feasible, result.status)

    def test_startup_shutdown_allowance(self):
        on = self.ed.solve([0, 8], [0], [1], [0], [], 1/12, previous_commitment=[0])
        off = self.ed.solve([0, 0], [0], [0], [8], [], 1/12, previous_commitment=[1])
        self.assertTrue(on.feasible)
        self.assertTrue(off.feasible)
        self.assertEqual(off.cost_rate, 0)

    def test_zero_minimum_unit_distinguishes_online_zero_from_startup(self):
        self.case.gen_ramp[:] = 1.0
        online = self.ed.solve([0, 8], [0], [1], [0], [], 1/12,
                               previous_commitment=[1])
        startup = self.ed.solve([0, 8], [0], [1], [0], [], 1/12,
                                previous_commitment=[0])
        self.assertFalse(online.feasible)
        self.assertTrue(startup.feasible)

    def test_bess_energy_is_not_advanced_by_queries(self):
        case = small_case(True)
        ed = EconomicDispatch(case, self.cfg)
        energy = np.array([10.0])
        result = ed.solve([0, 8], [0], [1], [8], energy, 1/12, previous_commitment=[1], storage_mode=[-1])
        self.assertTrue(result.feasible, result.status)
        self.assertEqual(energy[0], 10.0)
        expected = energy + (result.charge * .9 - result.discharge / .9) / 12
        np.testing.assert_allclose(result.energy_next, expected)
        self.assertLessEqual(result.violations['simultaneous_bess_mw'], 1e-8)
        ed.mci([0, 8], [0], [1], [8], energy, 1/12, base=result, buses=[1], previous_commitment=[1])
        self.assertEqual(energy[0], 10.0)

    def test_explicit_battery_mode_cannot_absorb_energy_when_full(self):
        case = small_case(True)
        case.branch_limit[:] = 100
        case.gen_pmin[:] = 20; case.gen_pmax[:] = 20
        ed = EconomicDispatch(case, self.cfg)
        # A full battery cannot physically absorb mandatory thermal surplus.
        result = ed.solve([0, 10], [0], [1], [20], [20], 1.0, previous_commitment=[1], storage_mode=[1])
        self.assertFalse(result.feasible)

    def test_cost_chord_and_off_state(self):
        self.case.gen_a[:] = .1; self.case.gen_c[:] = 7
        self.case.branch_limit[:] = 100
        result = self.ed.solve([0, 7.5], [0], [1], [7.5], [], 1.0, carbon_price=0, previous_commitment=[1])
        # Knots 5 and 10: average f(5)=59.5 and f(10)=117.
        self.assertAlmostEqual(result.fuel_cost_rate, (59.5 + 117)/2)
        result = self.ed.solve([0, 0], [0], [0], [0], [], 1.0, carbon_price=0, previous_commitment=[0])
        self.assertAlmostEqual(result.fuel_cost_rate, 0)

    def test_missing_physical_inputs_are_not_inferred(self):
        with self.assertRaisesRegex(ValueError, 'initial states are not inferred'):
            self.ed.solve([0, 8], [0], [1], None, [], 1/12)
        ed = EconomicDispatch(small_case(True), self.cfg)
        with self.assertRaisesRegex(ValueError, 'storage_mode is not provided'):
            ed.solve([0, 8], [0], [1], [8], [10], 1/12, previous_commitment=[1])

    def test_bess_cost_has_no_unprovided_extra_penalty(self):
        ed = EconomicDispatch(small_case(True), self.cfg)
        result = ed.solve([0, 8], [0], [1], [8], [10], 1/12,
                          previous_commitment=[1], storage_mode=[-1])
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.cost_rate,
                               result.fuel_cost_rate + 50 * result.emission_rate)

    def test_transition_ramps_are_explicit_inputs_not_pmax_fallbacks(self):
        self.cfg['solver']['startup_ramp_mw'] = None
        with self.assertRaisesRegex(ValueError, 'startup_ramp_mw is not provided'):
            self.ed.solve([0, 8], [0], [1], [0], [], 1/12, previous_commitment=[0])
        self.cfg['solver']['startup_ramp_mw'] = 4.
        result = self.ed.solve([0, 8], [0], [1], [0], [], 1/12, previous_commitment=[0])
        self.assertFalse(result.feasible)
        self.cfg['solver']['shutdown_ramp_mw'] = 4.
        result = self.ed.solve([0, 0], [0], [0], [8], [], 1/12, previous_commitment=[1])
        self.assertFalse(result.feasible)


class AdapterTests(unittest.TestCase):
    def test_sparse_duals_and_valid_milp_bound(self):
        solver = LinearSolver(solver_config())
        lp = solver.lp([1], sparse.csr_matrix([[-1.0]]), [-2], None, None, [(0, None)])
        self.assertTrue(lp.success)
        self.assertAlmostEqual(lp.objective, 2)
        self.assertAlmostEqual(lp.dual_ub[0], -1)
        mip = solver.milp([1], [1], sparse.csr_matrix([[1.0]]), [2.2], [np.inf], [(0, 4)])
        self.assertTrue(mip.success)
        self.assertAlmostEqual(mip.objective, 3)
        self.assertLessEqual(mip.bound, mip.objective + 1e-8)

    def test_benders_cuts_support_all_feasible_binary_commitments(self):
        """Independently enumerate a small recourse domain, including infeasibility."""
        from itertools import product
        from tsco.planning import BendersPlanner
        case = small_case()
        case.gen_bus = np.array([0, 0], int)
        case.gen_pmin = np.array([0., 0.])
        case.gen_pmax = np.array([10., 4.])
        case.gen_ramp = np.array([100., 100.])
        case.gen_a = np.array([.1, .2])
        case.gen_b = np.array([10., 5.])
        case.gen_c = np.array([2., 1.])
        case.gen_emission = np.array([.8, .2])
        case.gen_startup = case.gen_shutdown = np.ones(2)
        config = solver_config()
        planner = BendersPlanner(case, config)
        sub = planner._recourse(np.array([[0., 6.]]), np.array([[0.]]),
                                np.ones(2), np.array([6., 0.]), np.empty(0), 1.0, np.empty((1, 0)))
        evaluations = [(np.array(u, float), sub.evaluate(np.array(u, float)))
                       for u in product([0, 1], repeat=2)]
        feasible = [(u, result) for u, result in evaluations if result[0]]
        self.assertEqual(len(feasible), 2)
        for origin, (is_feasible, value, gradient, _) in evaluations:
            for target, (_, objective, _, _) in feasible:
                lower = value + gradient @ (target - origin)
                if is_feasible:
                    self.assertLessEqual(lower, objective + 1e-7)
                else:
                    self.assertLessEqual(lower, 1e-7)
        sub.solver.close(); planner.close()


if __name__ == '__main__': unittest.main()
