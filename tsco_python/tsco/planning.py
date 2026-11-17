"""Multi-cut Benders stochastic unit commitment for the day-ahead stage.

The commitment, startup, and shutdown variables are first-stage binaries.
Each scenario has a full time-coupled DC-dispatch/storage LP. All appearances
of commitment in the recourse model are kept in its affine right-hand side,
so dual optimality cuts and elastic phase-I feasibility cuts are genuine
Benders cuts. The master *dual bound*, not its incumbent objective, supplies
the outer lower bound. A deterministic extensive-form MILP is included for
small-case verification.

Unspecified experiment inputs are not inferred: scenario/hour battery modes,
initial physical states, minimum up/down durations, and terminal-storage policy
must be supplied explicitly. Battery modes keep recourse linear and disallow
simultaneous charge/discharge; this does not optimize unrestricted mixed-integer
storage modes. The nominal paper does not provide their numerical trajectories.
Optional outer iteration/time limits may be None (no requested limit).
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .schema import GridCase, Plan
from .optimization import LinearSolver


class _Rows:
    def __init__(self, n):
        self.n = n
        self.rows, self.cols, self.values = [], [], []
        self.lo, self.hi = [], []

    def add(self, entries, lower=-np.inf, upper=np.inf):
        row = len(self.lo)
        for col, value in entries.items():
            if value:
                self.rows.append(row); self.cols.append(int(col)); self.values.append(float(value))
        self.lo.append(float(lower)); self.hi.append(float(upper))

    def matrix(self):
        return sparse.csr_matrix((self.values, (self.rows, self.cols)), shape=(len(self.lo), self.n))


@dataclass
class _Recourse:
    cost: np.ndarray
    aub: sparse.csr_matrix
    b0: np.ndarray
    coupling: sparse.csr_matrix
    aeq: sparse.csr_matrix
    beq: np.ndarray
    bounds: list
    solver: LinearSolver
    storage_mode: np.ndarray

    def evaluate(self, commitment):
        rhs = self.b0 + self.coupling @ commitment
        result = self.solver.lp(self.cost, self.aub, rhs, self.aeq, self.beq, self.bounds)
        if result.success:
            return True, float(result.objective), np.asarray(self.coupling.T @ result.dual_ub), result

        # Each inequality gets a nonnegative slack, and each equality two.
        # All original variable bounds remain. Their feasible product is
        # nonempty by input validation, so the elastic LP itself is feasible.
        n, m, e = len(self.cost), self.aub.shape[0], self.aeq.shape[0]
        phase_cost = np.r_[np.zeros(n), np.ones(m + 2 * e)]
        phase_ub = sparse.hstack([self.aub, -sparse.eye(m), sparse.csr_matrix((m, 2 * e))], format='csr')
        phase_eq = sparse.hstack([self.aeq, sparse.csr_matrix((e, m)), sparse.eye(e), -sparse.eye(e)], format='csr')
        phase = self.solver.lp(phase_cost, phase_ub, rhs, phase_eq, self.beq,
                               self.bounds + [(0.0, None)] * (m + 2 * e))
        if not phase.success:
            raise RuntimeError(f'Benders phase-I LP did not solve: {phase.status}; original: {result.status}')
        if phase.objective <= 1e-7:
            # An interrupted or numerically unsuccessful cost LP must not be
            # misrepresented as infeasibility or used to add an invalid cut.
            raise RuntimeError(f'Recourse is phase-I feasible but cost LP failed: {result.status}')
        return False, float(phase.objective), np.asarray(self.coupling.T @ phase.dual_ub), phase


class BendersPlanner:
    """Scenario-specific stochastic UC; inspect ``Plan.converged`` explicitly.

    Parameters are the shared configuration dictionary. The solver choice is
    explicit: requesting Gurobi never silently falls back to another solver.
    ``planning.scenario_workers`` controls independent LP workers (serial if
    unspecified). This execution choice does not synthesize physical inputs.
    """
    def __init__(self, case: GridCase, config: dict):
        self.case, self.config = case, config
        self.settings = config.get('planning', {})
        self.master_solver = LinearSolver(config)
        self.last_recourse = []

    def _required(self, name):
        value = self.settings.get(name)
        if value is None:
            raise ValueError(f'Missing manuscript implementation input: planning.{name}; supply the actual value explicitly')
        return value

    def _duration_hours(self, name):
        hours = np.asarray(self._required(name), dtype=float)
        if hours.ndim == 0:
            hours = np.full(self.case.n_gen, float(hours))
        if hours.shape != (self.case.n_gen,) or not np.isfinite(hours).all() or np.any(hours < 0):
            raise ValueError(f'planning.{name} must be nonnegative hours, scalar or one value per generator')
        return hours

    def _duration_steps(self, name, dt):
        return np.ceil(self._duration_hours(name) / dt).astype(int)

    def _terminal_policy(self):
        policy = self._required('terminal_storage')
        if policy is True or policy == 'cyclic': return 'cyclic'
        if policy is False or policy == 'free': return 'free'
        raise ValueError("planning.terminal_storage must explicitly be 'free'/False or 'cyclic'/True")

    def _transition_limits(self, name):
        value = self.config.get('solver', {}).get(name)
        if value is None:
            raise ValueError(f'Missing solver.{name}: supply actual startup/shutdown transition limits in MW')
        result = np.asarray(value, dtype=float)
        if result.ndim == 0: result = np.full(self.case.n_gen, float(result))
        if result.shape != (self.case.n_gen,) or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError(f'solver.{name} must be finite nonnegative MW, scalar or one value per generator')
        return result

    def _prepare(self, demand_scenarios, renewable_scenarios, probabilities,
                 initial_commitment, initial_gen, initial_energy, hour_dt, storage_modes=None,
                 initial_status_hours=None):
        case = self.case
        d, r = np.asarray(demand_scenarios, float), np.asarray(renewable_scenarios, float)
        prob = np.asarray(probabilities, float)
        u0 = np.asarray(initial_commitment, float)
        energy = np.asarray(initial_energy, float)
        if initial_gen is None:
            raise ValueError('Missing initial generator outputs; initial_gen must be supplied, not inferred or skipped')
        pg0 = np.asarray(initial_gen, float)
        if d.ndim != 3 or d.shape[2] != case.n_bus or d.shape[0] == 0 or d.shape[1] == 0:
            raise ValueError('Demand must have nonempty shape [scenario,hour,bus]')
        s, h, _ = d.shape
        if r.shape != (s, h, case.n_renewable) or prob.shape != (s,):
            raise ValueError('Renewable or probability dimensions do not match scenarios')
        if not np.isfinite(d).all() or not np.isfinite(r).all() or np.any(d < 0) or np.any(r < 0):
            raise ValueError('Scenario powers must be finite and nonnegative')
        if not np.isfinite(prob).all() or np.any(prob <= 0) or not np.isclose(prob.sum(), 1.0):
            raise ValueError('Positive scenario probabilities must sum to one')
        if u0.shape != (case.n_gen,) or not np.isin(u0, [0, 1]).all():
            raise ValueError('Initial commitment must contain one binary entry per generator')
        if initial_status_hours is None:
            raise ValueError('Missing initial_status_hours: supply elapsed hours in each generator current on/off status')
        ages = np.asarray(initial_status_hours, dtype=float)
        if ages.shape != (case.n_gen,) or not np.isfinite(ages).all() or np.any(ages < 0):
            raise ValueError('initial_status_hours must be one finite nonnegative elapsed duration per generator')
        if energy.shape != (case.n_storage,) or not np.isfinite(energy).all() or np.any(energy < case.storage_min - 1e-7) or np.any(energy > case.storage_max + 1e-7):
            raise ValueError('Initial energy must satisfy the absolute-MWh storage bounds')
        if pg0 is not None and (pg0.shape != (case.n_gen,) or not np.isfinite(pg0).all() or np.any(pg0 < 0) or np.any(pg0 > case.gen_pmax * u0 + 1e-6)):
            raise ValueError('Initial generator output is incompatible with initial commitment')
        if np.any(pg0 < case.gen_pmin * u0 - 1e-6):
            raise ValueError('Initial online generator output is below its minimum output')
        if hour_dt <= 0 or not np.isfinite(hour_dt):
            raise ValueError('hour_dt must be positive')
        for array in (case.gen_a, case.gen_b, case.gen_c, case.gen_emission, case.gen_startup, case.gen_shutdown):
            if np.any(np.asarray(array) < 0):
                raise ValueError('This nonnegative-recourse formulation requires nonnegative cost/emission coefficients')
        if self.config['simulation']['carbon_price'] is None or self.config['simulation']['carbon_price'] < 0:
            raise ValueError('Carbon price must be nonnegative')
        if storage_modes is None:
            if case.n_storage:
                raise ValueError('Missing storage_modes: supply scenario/hour/store modes [S,H,K] in {-1,0,1}; no modes are inferred')
            modes = np.empty((s, h, 0), dtype=int)
        else:
            modes = np.asarray(storage_modes)
            if modes.shape != (s, h, case.n_storage) or not np.isin(modes, [-1, 0, 1]).all():
                raise ValueError('storage_modes must have shape [scenario,hour,store] and contain only -1, 0, +1')
            modes = modes.astype(int)
        self._duration_steps('min_up_hours', hour_dt)
        self._duration_steps('min_down_hours', hour_dt)
        self._transition_limits('startup_ramp_mw')
        self._transition_limits('shutdown_ramp_mw')
        if case.n_storage: self._terminal_policy()
        return d, r, prob, u0, pg0, energy, modes, ages

    def _master(self, demand, renewable, probabilities, initial_commitment, dt, initial_status_hours):
        h, g, s = demand.shape[1], self.case.n_gen, len(probabilities)
        nu, n = h * g, 3 * h * g + s
        objective = np.zeros(n)
        objective[nu:2 * nu] = np.tile(self.case.gen_startup, h)
        objective[2 * nu:3 * nu] = np.tile(self.case.gen_shutdown, h)
        objective[3 * nu:] = probabilities
        integers = np.r_[np.ones(3 * nu, int), np.zeros(s, int)]
        bounds = [(0, 1)] * (3 * nu) + [(0, None)] * s
        rows = _Rows(n)
        min_up = self._duration_steps('min_up_hours', dt)
        min_down = self._duration_steps('min_down_hours', dt)
        required_initial = np.where(initial_commitment > .5,
                                    self._duration_hours('min_up_hours'),
                                    self._duration_hours('min_down_hours'))
        residual_periods = np.ceil(np.maximum(0.0, required_initial - initial_status_hours) / dt).astype(int)
        for t in range(h):
            for j in range(g):
                ij = t * g + j
                changes = {ij: 1, nu + ij: -1, 2 * nu + ij: 1}
                rhs = initial_commitment[j] if t == 0 else 0.0
                if t: changes[(t - 1) * g + j] = -1
                rows.add(changes, rhs, rhs)
                if t < residual_periods[j]:
                    rows.add({ij: 1}, initial_commitment[j], initial_commitment[j])
                rows.add({nu + ij: 1, 2 * nu + ij: 1}, upper=1)
                up = {nu + k * g + j: 1 for k in range(max(0, t - int(min_up[j]) + 1), t + 1)}
                up[ij] = -1
                rows.add(up, upper=0)
                down = {2 * nu + k * g + j: 1 for k in range(max(0, t - int(min_down[j]) + 1), t + 1)}
                down[ij] = 1
                rows.add(down, upper=1)
            # A valid aggregate-capacity necessary condition strengthens the
            # master without replacing scenario/network feasibility cuts.
            for scenario in range(s):
                required = demand[scenario, t].sum() - renewable[scenario, t].sum() - self.case.storage_power.sum()
                rows.add({t * g + j: self.case.gen_pmax[j] for j in range(g)}, lower=required)
        return objective, integers, bounds, rows

    def _recourse(self, demand, available, initial_commitment, initial_gen, initial_energy, dt, storage_modes):
        c = self.case
        h, g, b, r, k = len(demand), c.n_gen, c.n_bus, c.n_renewable, c.n_storage
        # Per-hour blocks: generation, fuel epigraph, used RES, charge,
        # discharge, end-of-hour energy, and voltage angles.
        offsets = np.cumsum([0, g, g, r, k, k, k, b])
        width, nu = int(offsets[-1]), h * g
        n = h * width
        idx = lambda t, block, j: t * width + int(offsets[block]) + j
        cost = np.zeros(n)
        bounds = [(None, None)] * n
        ub, eq = _Rows(n), _Rows(n)
        coupling_rows = []
        modes = np.asarray(storage_modes, dtype=int)
        price = float(self.config['simulation']['carbon_price'])
        segments = int(self.config['solver']['pwl_segments'])
        if segments < 1: raise ValueError('Invalid PWL segment count')
        terminal_policy = self._terminal_policy() if k else 'free'
        startup_limit = self._transition_limits('startup_ramp_mw')
        shutdown_limit = self._transition_limits('shutdown_ramp_mw')
        active = np.asarray(c.branch_limit) > 0
        susceptance = np.where(active, c.base_mva / (c.branch_x * c.branch_tap), 0.0)
        incidence = sparse.csr_matrix((np.r_[np.ones(len(c.branch_from)), -np.ones(len(c.branch_from))],
                                      (np.r_[np.arange(len(c.branch_from)), np.arange(len(c.branch_from))],
                                       np.r_[c.branch_from, c.branch_to])), shape=(len(c.branch_from), b))
        bbus = (incidence.T @ sparse.diags(susceptance) @ incidence).tocsr()
        connectivity = sparse.csr_matrix((np.ones(2 * int(active.sum())),
                     (np.r_[c.branch_from[active], c.branch_to[active]],
                      np.r_[c.branch_to[active], c.branch_from[active]])), shape=(b, b))
        _, components = connected_components(connectivity, directed=False)
        references = {int(np.flatnonzero(components == comp)[0]) for comp in np.unique(components)}

        def add_ub(entries, rhs=0.0, u_coeff=None):
            ub.add(entries, upper=rhs)
            coupling_rows.append(u_coeff or {})

        for t in range(h):
            for j in range(g):
                p, fuel, uj = idx(t, 0, j), idx(t, 1, j), t * g + j
                bounds[p], bounds[fuel] = (0, float(c.gen_pmax[j])), (0, None)
                cost[p], cost[fuel] = price * c.gen_emission[j] * dt, dt
                add_ub({p: 1}, u_coeff={uj: c.gen_pmax[j]})
                add_ub({p: -1}, u_coeff={uj: -c.gen_pmin[j]})
                knots = np.linspace(c.gen_pmin[j], c.gen_pmax[j], segments + 1)
                if np.isclose(knots[-1], knots[0]):
                    slopes = [2 * c.gen_a[j] * knots[0] + c.gen_b[j]]
                    intercepts = [c.gen_c[j] - c.gen_a[j] * knots[0] ** 2]
                else:
                    values = c.gen_a[j] * knots ** 2 + c.gen_b[j] * knots + c.gen_c[j]
                    slopes = np.diff(values) / np.diff(knots)
                    intercepts = values[:-1] - slopes * knots[:-1]
                for slope, intercept in zip(slopes, intercepts):
                    add_ub({p: slope, fuel: -1}, u_coeff={uj: -intercept})
                if t:
                    previous = idx(t - 1, 0, j)
                    # Equivalent on binary commitments to RU*u_prev+SU*z+,
                    # and RD*u_now+SD*z-. For off/off the extra nonnegative
                    # RHS is redundant because both dispatch values are zero.
                    # This form keeps the recourse RHS affine in u alone.
                    add_ub({p: 1, previous: -1}, startup_limit[j],
                           {(t - 1) * g + j: c.gen_ramp[j] * dt - startup_limit[j]})
                    add_ub({previous: 1, p: -1}, shutdown_limit[j],
                           {uj: c.gen_ramp[j] * dt - shutdown_limit[j]})
                elif initial_gen is not None:
                    add_ub({p: 1}, initial_gen[j] + c.gen_ramp[j] * dt * initial_commitment[j]
                           + startup_limit[j] * (1 - initial_commitment[j]))
                    add_ub({p: -1}, shutdown_limit[j] - initial_gen[j],
                           {uj: c.gen_ramp[j] * dt - shutdown_limit[j]})

            for j in range(r): bounds[idx(t, 2, j)] = (0, float(available[t, j]))
            for j in range(k):
                ch, dis, energy = idx(t, 3, j), idx(t, 4, j), idx(t, 5, j)
                bounds[ch] = (0, float(c.storage_power[j]) if modes[t, j] == 1 else 0)
                bounds[dis] = (0, float(c.storage_power[j]) if modes[t, j] == -1 else 0)
                bounds[energy] = (float(c.storage_min[j]), float(c.storage_max[j]))
                entries = {energy: 1, ch: -c.storage_eta_c[j] * dt, dis: dt / c.storage_eta_d[j]}
                rhs = float(initial_energy[j]) if t == 0 else 0.0
                if t: entries[idx(t - 1, 5, j)] = -1
                eq.add(entries, rhs, rhs)
                if t == h - 1 and terminal_policy == 'cyclic':
                    eq.add({energy: 1}, float(initial_energy[j]), float(initial_energy[j]))
            for bus in range(b):
                theta = idx(t, 6, bus)
                bounds[theta] = (0, 0) if bus in references else (None, None)
                entries = {idx(t, 0, j): 1 for j in np.flatnonzero(c.gen_bus == bus)}
                entries.update({idx(t, 2, j): 1 for j in np.flatnonzero(c.renewable_bus == bus)})
                entries.update({idx(t, 3, j): -1 for j in np.flatnonzero(c.storage_bus == bus)})
                entries.update({idx(t, 4, j): 1 for j in np.flatnonzero(c.storage_bus == bus)})
                row = bbus.getrow(bus)
                entries.update({idx(t, 6, int(col)): -value for col, value in zip(row.indices, row.data)})
                eq.add(entries, demand[t, bus], demand[t, bus])
            for line in np.flatnonzero(active):
                f, to, adm = int(c.branch_from[line]), int(c.branch_to[line]), susceptance[line]
                add_ub({idx(t, 6, f): adm, idx(t, 6, to): -adm}, c.branch_limit[line])
                add_ub({idx(t, 6, f): -adm, idx(t, 6, to): adm}, c.branch_limit[line])

        br, bc, bv = [], [], []
        for row, entries in enumerate(coupling_rows):
            for col, value in entries.items():
                br.append(row); bc.append(col); bv.append(value)
        coupling = sparse.csr_matrix((bv, (br, bc)), shape=(len(coupling_rows), nu))
        return _Recourse(cost, ub.matrix(), np.asarray(ub.hi), coupling, eq.matrix(),
                         np.asarray(eq.hi), bounds, LinearSolver(self.config), modes)

    def solve(self, demand_scenarios, renewable_scenarios, probabilities,
              initial_commitment, initial_gen, initial_energy, hour_dt=1.0, storage_modes=None,
              initial_status_hours=None):
        """Solve supplied scenarios; ``storage_modes`` has shape [S,H,K].

        For every actual storage unit its mode must be supplied: +1 charge,
        -1 discharge, 0 idle. No signs or initial states are inferred from
        the demand trace. Missing experiment values raise a clear error.
        """
        start = time.perf_counter()
        d, r, prob, u0, pg0, e0, modes, ages = self._prepare(demand_scenarios, renewable_scenarios,
                                               probabilities, initial_commitment, initial_gen, initial_energy,
                                               hour_dt, storage_modes, initial_status_hours)
        s, h, g = len(prob), d.shape[1], self.case.n_gen
        nu = h * g
        c, integer, bounds, rows = self._master(d, r, prob, u0, hour_dt, ages)
        recourse = [self._recourse(d[i], r[i], u0, pg0, e0, hour_dt, modes[i]) for i in range(s)]
        self.last_recourse = recourse
        ub, lb, best, best_first = np.inf, -np.inf, None, 0.0
        tolerance = float(self._required('relative_gap'))
        limit_value = self.settings.get('max_iterations')
        limit = None if limit_value is None else int(limit_value)
        time_value = self.settings.get('time_limit_seconds')
        time_limit = None if time_value is None else float(time_value)
        if tolerance < 0 or (limit is not None and limit < 1) or (time_limit is not None and time_limit <= 0):
            raise ValueError('Relative gap and optional outer stopping limits must be valid')
        workers = max(1, int(self.settings.get('scenario_workers') or 1))
        history, status, converged = [], 'iteration_limit', False
        iteration = 0
        try:
            while limit is None or iteration < limit:
                iteration += 1
                master = self.master_solver.milp(c, integer, rows.matrix(), rows.lo, rows.hi, bounds)
                if not master.success:
                    status = 'master_without_incumbent: ' + master.status
                    break
                if np.isfinite(master.bound): lb = max(lb, master.bound)
                u = np.rint(master.x[:nu])
                if workers == 1:
                    outcomes = [sub.evaluate(u) for sub in recourse]
                else:
                    with ThreadPoolExecutor(max_workers=min(workers, s)) as pool:
                        outcomes = list(pool.map(lambda sub: sub.evaluate(u), recourse))
                all_feasible = all(item[0] for item in outcomes)
                first_cost = float(c[nu:3 * nu] @ master.x[nu:3 * nu])
                if all_feasible:
                    candidate = first_cost + sum(prob[i] * outcomes[i][1] for i in range(s))
                    if candidate < ub:
                        ub, best, best_first = candidate, u.reshape(h, g).astype(int).copy(), first_cost
                if np.isfinite(ub) and lb > ub + max(1e-5, 1e-7 * abs(ub)):
                    raise RuntimeError(f'Invalid Benders bounds: lower bound {lb} exceeds feasible upper bound {ub}')
                feasibility_cuts = 0
                for scenario, (feasible, value, grad, _) in enumerate(outcomes):
                    cut = {j: float(grad[j]) for j in np.flatnonzero(np.abs(grad) > 1e-12)}
                    if feasible:
                        cut[3 * nu + scenario] = -1
                    else:
                        feasibility_cuts += 1
                    # theta >= Q(u0)+g'(u-u0), or 0 >= phi(u0)+g'(u-u0).
                    rows.add(cut, upper=float(grad @ u - value))
                gap = max(0.0, ub - lb) / max(1.0, abs(ub)) if np.isfinite(ub) and np.isfinite(lb) else np.inf
                history.append({'iteration': iteration, 'lower_bound': float(lb), 'upper_bound': float(ub),
                                'relative_gap': float(gap), 'feasibility_cuts': feasibility_cuts,
                                'optimality_cuts': s - feasibility_cuts,
                                'elapsed_seconds': time.perf_counter() - start,
                                'storage_mode_rule': 'explicit_scenario_input'})
                if best is not None and gap <= tolerance:
                    status, converged = 'relative_gap_satisfied', True
                    break
                if time_limit is not None and time.perf_counter() - start >= time_limit:
                    status = 'time_limit'
                    break
        finally:
            for sub in recourse: sub.solver.close()
        gap = max(0.0, ub - lb) / max(1.0, abs(ub)) if np.isfinite(ub) and np.isfinite(lb) else np.inf
        return Plan(np.empty((0, g), int) if best is None else best, float(ub), float(lb),
                    float(gap), converged, iteration, time.perf_counter() - start,
                    best_first, history, status)

    def solve_extensive(self, demand_scenarios, renewable_scenarios, probabilities,
                        initial_commitment, initial_gen, initial_energy, hour_dt=1.0, storage_modes=None,
                        initial_status_hours=None):
        """Solve exactly the same finite-scenario/PWL model as one MILP.

        Intended for small regression cases, not a purported full-scale oracle.
        The reported relative gap remains the actual MILP solver gap.
        """
        start = time.perf_counter()
        d, r, prob, u0, pg0, e0, modes, ages = self._prepare(demand_scenarios, renewable_scenarios,
                                               probabilities, initial_commitment, initial_gen, initial_energy,
                                               hour_dt, storage_modes, initial_status_hours)
        h, g, s = d.shape[1], self.case.n_gen, len(prob)
        nu = h * g
        c, integer, bounds, master_rows = self._master(d, r, prob, u0, hour_dt, ages)
        nm = len(c)
        recourse = [self._recourse(d[i], r[i], u0, pg0, e0, hour_dt, modes[i]) for i in range(s)]
        lengths = [len(sub.cost) for sub in recourse]
        positions = np.cumsum([nm] + lengths)
        total = int(positions[-1])
        c[3 * nu:] = 0.0
        cfull = np.r_[c, *[prob[i] * sub.cost for i, sub in enumerate(recourse)]]
        ifull = np.r_[integer, np.zeros(total - nm, int)]
        fullbounds = bounds[:3 * nu] + [(0, 0)] * s + sum([sub.bounds for sub in recourse], [])
        matrices = [sparse.hstack([master_rows.matrix(), sparse.csr_matrix((len(master_rows.lo), total - nm))], format='csr')]
        lows, highs = [np.asarray(master_rows.lo)], [np.asarray(master_rows.hi)]
        for scenario, sub in enumerate(recourse):
            begin, end = positions[scenario], positions[scenario + 1]
            left = sparse.hstack([-sub.coupling, sparse.csr_matrix((sub.aub.shape[0], begin - nu))], format='csr')
            matrices.append(sparse.hstack([left, sub.aub, sparse.csr_matrix((sub.aub.shape[0], total - end))], format='csr'))
            lows.append(np.full(sub.aub.shape[0], -np.inf)); highs.append(sub.b0)
            matrices.append(sparse.hstack([sparse.csr_matrix((sub.aeq.shape[0], begin)), sub.aeq,
                                          sparse.csr_matrix((sub.aeq.shape[0], total - end))], format='csr'))
            lows.append(sub.beq); highs.append(sub.beq)
        try:
            result = self.master_solver.milp(cfull, ifull, sparse.vstack(matrices, format='csr'),
                                             np.concatenate(lows), np.concatenate(highs), fullbounds)
        finally:
            for sub in recourse: sub.solver.close()
        u = np.rint(result.x[:nu]).reshape(h, g).astype(int) if result.success else np.empty((0, g), int)
        first = float(c[nu:3 * nu] @ result.x[nu:3 * nu]) if result.success else 0.0
        return Plan(u, float(result.objective), float(result.bound), float(result.gap), bool(result.optimal),
                    0, time.perf_counter() - start, first,
                    [{'storage_mode_rule': 'explicit_scenario_input', 'formulation': 'extensive'}], result.status)

    def close(self):
        self.master_solver.close()
