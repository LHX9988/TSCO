"""Lossless DC economic dispatch, nodal MCI, and exact demand-headroom LPs.

Units: MW, MWh, hours, dollars/hour, tCO2/hour. In particular, MCI is
``delta emission_rate / delta load`` in tCO2/MWh; multiplying the numerator by
the five-minute duration would introduce an incorrect factor of twelve.

BESS charging/discharging directions and initial physical states must be
provided by the data/controller. No operating mode, initial energy, or added
cycling-cost coefficient is inferred by this module. Subsequent ED, MCI, and
headroom problems hold the supplied mode and physical-start state fixed.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from .optimization import LinearSolver
from .schema import DispatchResult, GridCase


def _rows_to_csr(rows, n):
    rr, cc, dd = [], [], []
    for i, row in enumerate(rows):
        for j, value in row.items():
            if value:
                rr.append(i); cc.append(j); dd.append(value)
    return sparse.csr_matrix((dd, (rr, cc)), shape=(len(rows), n))


def network_matrices(case: GridCase):
    """Return line-angle and bus-angle matrices; zero limit means open line."""
    n, m = case.n_bus, len(case.branch_from)
    f, t = np.asarray(case.branch_from, int), np.asarray(case.branch_to, int)
    if np.any(np.asarray(case.branch_x) <= 0) or np.any(np.asarray(case.branch_tap) <= 0):
        raise ValueError('DC line reactances and tap ratios must be positive')
    active = np.asarray(case.branch_limit) > 0
    b = case.base_mva / (np.asarray(case.branch_x) * np.asarray(case.branch_tap)) * active
    incidence = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)],
                                  (np.r_[np.arange(m), np.arange(m)], np.r_[f, t])),
                                 shape=(m, n))
    line = sparse.diags(b) @ incidence
    bus = incidence.T @ line
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, z, enabled in zip(f, t, active):
        if enabled: parent[find(int(a))] = find(int(z))
    refs = sorted({find(i) for i in range(n)})
    return line.tocsr(), bus.tocsr(), refs


class EconomicDispatch:
    def __init__(self, case: GridCase, config: dict):
        self.case, self.config = case, config
        self.solver = LinearSolver(config)
        self.options = config['solver']
        for key in ('pwl_segments', 'mci_delta_mw', 'feasibility_tol', 'int_feas_tol'):
            if self.options.get(key) is None:
                raise ValueError(f'solver.{key} is not provided')
        if config.get('simulation', {}).get('carbon_price') is None:
            raise ValueError('simulation.carbon_price is not provided')
        self.carbon_price = float(config['simulation']['carbon_price'])
        self.segments = int(self.options['pwl_segments'])
        if self.segments < 1: raise ValueError('pwl_segments must be positive')
        self.delta = float(self.options['mci_delta_mw'])
        if self.delta <= 0: raise ValueError('mci_delta_mw must be positive')
        self.counters = dict(calls=0, total_seconds=0.0, mci_calls=0, headroom_calls=0)

    def solve(self, demand, renewable, commitment, previous_gen, energy,
              dt_hours, carbon_price=None, storage_mode=None, headroom_bus=None,
              previous_commitment=None):
        """Solve one physical interval without advancing any caller state.

        All inner-loop queries must pass the same physical-start generation,
        energy, and elapsed duration. ``storage_mode``: +1 charge, -1 discharge,
        0 idle. A mode must be supplied for each store; no mode is inferred.
        Previous output and commitment must be supplied for every generator,
        including at the beginning of the test window.
        """
        t0 = time.perf_counter()
        demand, renewable = np.asarray(demand, float), np.asarray(renewable, float)
        commitment, energy = np.asarray(commitment, float), np.asarray(energy, float)
        if self.case.n_gen and (previous_gen is None or previous_commitment is None):
            raise ValueError('previous_gen and previous_commitment must be supplied; initial states are not inferred')
        previous_gen = np.asarray([] if previous_gen is None else previous_gen, float)
        previous_commitment = np.asarray([] if previous_commitment is None else previous_commitment, float)
        self._check_inputs(demand, renewable, commitment, previous_gen, energy, dt_hours)
        if previous_commitment.shape != commitment.shape or not np.isin(previous_commitment, [0, 1]).all():
            raise ValueError('previous_commitment must contain one binary entry per generator')
        tol = float(self.options['feasibility_tol'])
        if (np.any(previous_gen < -tol) or
                np.any(previous_gen > self.case.gen_pmax * previous_commitment + tol)):
            raise ValueError('Previous output is incompatible with previous commitment')
        price = self.carbon_price if carbon_price is None else float(carbon_price)
        if headroom_bus is not None and not 0 <= headroom_bus < self.case.n_bus:
            raise ValueError('headroom_bus is out of range')
        if storage_mode is None and self.case.n_storage:
            raise ValueError('storage_mode is not provided; supply the interval charging/discharging modes')
        elif storage_mode is None:
            storage_mode = np.empty(0, dtype=int)
        storage_mode = np.asarray(storage_mode, int)
        if storage_mode.shape != (self.case.n_storage,) or not np.all(np.isin(storage_mode, [-1, 0, 1])):
            raise ValueError('storage_mode must contain one value in {-1,0,1} per store')
        result = self._solve_raw(demand, renewable, commitment, previous_gen,
                                 energy, dt_hours, price, storage_mode, headroom_bus, previous_commitment)
        result.seconds = time.perf_counter() - t0
        return result

    def mci(self, demand, renewable, commitment, previous_gen, energy, dt_hours,
            base=None, buses=None, carbon_price=None, previous_commitment=None, storage_mode=None):
        """Nodal finite differences; infeasible perturbations are NaN, never zero.

        Unrequested buses also remain NaN. A NaN must not be interpreted as a
        clean node: it means no valid marginal estimate exists at this step.
        """
        if base is None:
            base = self.solve(demand, renewable, commitment, previous_gen, energy,
                              dt_hours, carbon_price=carbon_price, previous_commitment=previous_commitment,
                              storage_mode=storage_mode)
        if not base.feasible: raise ValueError('MCI requires a feasible base ED')
        result = np.full(self.case.n_bus, np.nan)
        buses = range(self.case.n_bus) if buses is None else buses
        for bus in buses:
            perturbed = np.asarray(demand, float).copy()
            perturbed[int(bus)] += self.delta
            self.counters['mci_calls'] += 1
            other = self.solve(perturbed, renewable, commitment, previous_gen, energy,
                               dt_hours, carbon_price=carbon_price,
                               storage_mode=base.storage_mode, previous_commitment=previous_commitment)
            if other.feasible:
                result[int(bus)] = (other.emission_rate - base.emission_rate) / self.delta
        return result

    def headrooms(self, demand, renewable, commitment, previous_gen, energy,
                  dt_hours, base=None, buses=None, carbon_price=None, previous_commitment=None,
                  storage_mode=None):
        """Maximize additional demand at each requested bus with physical ED constraints.

        Margins are individual, not simultaneously achievable. Recompute them
        after every accepted assignment; do not sum stale margins.
        """
        if base is None:
            base = self.solve(demand, renewable, commitment, previous_gen, energy,
                              dt_hours, carbon_price=carbon_price, previous_commitment=previous_commitment,
                              storage_mode=storage_mode)
        if not base.feasible: raise ValueError('Headroom requires a feasible base ED')
        result = np.full(self.case.n_bus, np.nan)
        buses = range(self.case.n_bus) if buses is None else buses
        for bus in buses:
            self.counters['headroom_calls'] += 1
            other = self.solve(demand, renewable, commitment, previous_gen, energy,
                               dt_hours, carbon_price=carbon_price,
                               storage_mode=base.storage_mode, headroom_bus=int(bus),
                               previous_commitment=previous_commitment)
            if not other.feasible:
                raise RuntimeError(f'Headroom LP failed from a feasible base: {other.status}')
            result[int(bus)] = max(0.0, other.headroom_mw)
        return result

    def _check_inputs(self, demand, renewable, commitment, previous_gen, energy, dt):
        c = self.case
        tol = float(self.options['feasibility_tol'])
        expected = [(demand, c.n_bus, 'demand'), (renewable, c.n_renewable, 'renewable'),
                    (commitment, c.n_gen, 'commitment'), (energy, c.n_storage, 'energy')]
        if previous_gen is not None: expected.append((previous_gen, c.n_gen, 'previous_gen'))
        for value, size, name in expected:
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be a finite vector of length {size}')
        if not np.isfinite(dt) or dt <= 0: raise ValueError('dt_hours must be positive')
        if np.any(demand < -tol) or np.any(renewable < -tol):
            raise ValueError('Loads and renewable availability cannot be negative')
        if np.any(np.abs(commitment - np.rint(commitment)) > float(self.options['int_feas_tol'])) or np.any((commitment < 0) | (commitment > 1)):
            raise ValueError('Commitments must be binary')
        if np.any(np.asarray(c.gen_a) < 0): raise ValueError('Convex cost epigraph requires gen_a >= 0')
        if np.any(energy < np.asarray(c.storage_min) - tol) or np.any(energy > np.asarray(c.storage_max) + tol):
            raise ValueError('Physical-start BESS energy violates its MWh bounds')

    def _transition_ramp(self, key):
        value = self.options.get(key)
        if value is None:
            raise ValueError(f'solver.{key} is not provided for the generator status change')
        array = np.asarray(value, float)
        if array.ndim == 0:
            array = np.full(self.case.n_gen, float(array))
        if array.shape != (self.case.n_gen,) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f'solver.{key} must be nonnegative MW, scalar or one value per generator')
        return array

    def _solve_raw(self, demand, renewable, u, prev, energy, dt, price, mode, headroom_bus,
                   previous_commitment=None):
        c = self.case
        ng, nr, ns, nb = c.n_gen, c.n_renewable, c.n_storage, c.n_bus
        cursor, idx = 0, {}
        for name, size in [('p', ng), ('r', nr), ('ch', ns), ('dis', ns), ('theta', nb), ('z', ng)]:
            idx[name] = np.arange(cursor, cursor + size); cursor += size
        delta_index = cursor if headroom_bus is not None else None
        nv = cursor + int(headroom_bus is not None)
        objective = np.zeros(nv)
        objective[idx['p']] = price * c.gen_emission
        objective[idx['z']] = 1.0
        if headroom_bus is not None:
            objective[:] = 0.0; objective[delta_index] = -1.0
        bounds = [(0.0, None)] * nv
        starting = (u > .5) & (previous_commitment < .5)
        stopping = (u < .5) & (previous_commitment > .5)
        startup = self._transition_ramp('startup_ramp_mw') if np.any(starting) else None
        shutdown = self._transition_ramp('shutdown_ramp_mw') if np.any(stopping) else None
        for g in range(ng):
            lower, upper = c.gen_pmin[g] * u[g], c.gen_pmax[g] * u[g]
            was_on = previous_commitment[g] > .5
            if u[g] > 0.5 and was_on:
                lower = max(lower, prev[g] - c.gen_ramp[g] * dt)
                upper = min(upper, prev[g] + c.gen_ramp[g] * dt)
            elif starting[g]:
                upper = min(upper, startup[g])
            elif stopping[g]:
                lower = max(lower, prev[g] - shutdown[g])
            bounds[idx['p'][g]] = (float(lower), float(upper))
        for r in range(nr): bounds[idx['r'][r]] = (0.0, float(renewable[r]))
        for b in range(ns):
            maxch = c.storage_power[b] if mode[b] == 1 else 0.0
            maxdis = c.storage_power[b] if mode[b] == -1 else 0.0
            bounds[idx['ch'][b]], bounds[idx['dis'][b]] = (0.0, float(maxch)), (0.0, float(maxdis))
        line, busmat, refs = network_matrices(c)
        for theta in idx['theta']: bounds[theta] = (None, None)
        for ref in refs: bounds[idx['theta'][ref]] = (0.0, 0.0)
        eq = [dict() for _ in range(nb)]
        for g, b in enumerate(c.gen_bus): eq[int(b)][idx['p'][g]] = 1.0
        for r, b in enumerate(c.renewable_bus): eq[int(b)][idx['r'][r]] = 1.0
        for k, b in enumerate(c.storage_bus):
            eq[int(b)][idx['ch'][k]], eq[int(b)][idx['dis'][k]] = -1.0, 1.0
        for b in range(nb):
            row = busmat.getrow(b)
            for col, val in zip(row.indices, row.data): eq[b][idx['theta'][col]] = -val
        if headroom_bus is not None: eq[headroom_bus][delta_index] = -1.0
        ub, rhs, cost_segments = [], [], []
        for ell in range(line.shape[0]):
            row = line.getrow(ell)
            coeff = {idx['theta'][col]: val for col, val in zip(row.indices, row.data)}
            ub.append(coeff); rhs.append(c.branch_limit[ell])
            ub.append({j: -v for j, v in coeff.items()}); rhs.append(c.branch_limit[ell])
        for b in range(ns):
            change = {idx['ch'][b]: dt * c.storage_eta_c[b],
                      idx['dis'][b]: -dt / c.storage_eta_d[b]}
            ub.append(change); rhs.append(c.storage_max[b] - energy[b])
            ub.append({j: -v for j, v in change.items()}); rhs.append(energy[b] - c.storage_min[b])
        for g in range(ng):
            knots = np.linspace(c.gen_pmin[g], c.gen_pmax[g], self.segments + 1)
            slopes = c.gen_a[g] * (knots[:-1] + knots[1:]) + c.gen_b[g]
            intercepts = c.gen_c[g] - c.gen_a[g] * knots[:-1] * knots[1:]
            cost_segments.append((slopes, intercepts))
            for slope, intercept in zip(slopes, intercepts):
                ub.append({idx['p'][g]: slope, idx['z'][g]: -1.0})
                rhs.append(-intercept * u[g])
        aeq, aub = _rows_to_csr(eq, nv), _rows_to_csr(ub, nv)
        solved = self.solver.lp(objective, aub, np.asarray(rhs), aeq, demand, bounds)
        self.counters['calls'] += 1
        self.counters['total_seconds'] += solved.seconds
        if not solved.success:
            return DispatchResult(False, solved.status, seconds=solved.seconds,
                                  storage_mode=np.array(mode if mode is not None else [], dtype=int))
        x = solved.x
        p, ren, ch, dis, angles = [x[idx[k]].copy() for k in ['p', 'r', 'ch', 'dis', 'theta']]
        enext = energy + dt * (c.storage_eta_c * ch - dis / c.storage_eta_d)
        flows = np.asarray(line @ angles)
        fuel = sum(max(0.0, float(np.max(slopes * p[g] + intercepts * u[g])))
                   for g, (slopes, intercepts) in enumerate(cost_segments))
        emissions = float(c.gen_emission @ p)
        violations = {
            'balance_mw': float(np.max(np.abs(aeq @ x - demand), initial=0.0)),
            'line_overload_mw': float(np.max(np.abs(flows) - c.branch_limit, initial=0.0)),
            'soc_mwh': float(max(np.max(c.storage_min - enext, initial=0.0),
                                 np.max(enext - c.storage_max, initial=0.0))),
            'simultaneous_bess_mw': float(np.max(np.minimum(ch, dis), initial=0.0)),
            'inequality_residual': float(np.max(aub @ x - np.asarray(rhs), initial=0.0)),
        }
        feasible = max(violations.values(), default=0.0) <= float(self.options['feasibility_tol'])
        return DispatchResult(
            feasible=feasible, status=solved.status if feasible else 'post-solve constraint residual',
            cost_rate=float(fuel + price * emissions),
            fuel_cost_rate=float(fuel), emission_rate=emissions,
            p_gen=p, renewable_used=ren, charge=ch, discharge=dis,
            energy_next=enext, angles=angles, flows=flows,
            lmp=solved.dual_eq.copy() if headroom_bus is None else np.full(nb, np.nan),
            headroom_mw=0.0 if headroom_bus is None else float(x[delta_index]),
            storage_mode=np.array(mode if mode is not None else [], dtype=int),
            seconds=solved.seconds, violations=violations)
