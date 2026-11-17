"""Small, explicit LP/MILP adapter for SciPy/HiGHS and optional Gurobi.

SciPy is a cold-start portability backend. Gurobi retains compatible LP models
and reuses their simplex bases; it is never silently substituted when a license
or import fails. Bounds and objective values are never confused: ``bound`` is
the MILP solver's dual bound, not its incumbent objective.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import time
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass
class SolveResult:
    success: bool
    status: str
    x: np.ndarray | None = None
    objective: float = float('inf')
    bound: float = float('-inf')
    gap: float = float('inf')
    dual_ub: np.ndarray = field(default_factory=lambda: np.empty(0))
    dual_eq: np.ndarray = field(default_factory=lambda: np.empty(0))
    seconds: float = 0.0
    optimal: bool = False


def _matrix(a, n):
    return sparse.csr_matrix((0, n)) if a is None else sparse.csr_matrix(a, dtype=float)


def _bounds(bounds, n):
    if bounds is None:
        return np.zeros(n), np.full(n, np.inf)
    if isinstance(bounds, Bounds):
        return np.broadcast_to(bounds.lb, (n,)).copy(), np.broadcast_to(bounds.ub, (n,)).copy()
    low = np.array([-np.inf if v[0] is None else v[0] for v in bounds], float)
    high = np.array([np.inf if v[1] is None else v[1] for v in bounds], float)
    if low.shape != (n,) or high.shape != (n,):
        raise ValueError('One (lower, upper) bound pair is required for every variable')
    return low, high


class LinearSolver:
    """Stable interface used by ED and the Benders master/recourse models."""
    def __init__(self, config=None):
        if not isinstance(config, dict):
            raise ValueError('An explicit solver configuration is required')
        self.config = config.get('solver', config)
        backend = self.config.get('backend')
        if not isinstance(backend, str):
            raise ValueError('solver.backend must be provided explicitly')
        self.backend = backend.lower()
        if self.backend not in ('scipy', 'gurobi'):
            raise ValueError('solver.backend must be scipy or gurobi')
        self.lp_calls = self.milp_calls = self.warm_starts = 0
        self.total_seconds = 0.0
        self._cache = OrderedDict()
        self._gp = None
        self._env = None
        if self.backend == 'gurobi':
            try:
                import gurobipy as gp
            except ImportError as exc:
                raise ImportError('Install gurobipy and a valid Gurobi license, or explicitly select solver.backend=scipy') from exc
            self._gp = gp

    def _setting(self, key):
        value = self.config.get(key)
        if value is None:
            raise ValueError(f'solver.{key} is not provided')
        return value

    def _time_limit(self, options, key='time_limit'):
        value = self.config.get('time_limit')
        if value is not None:
            if not np.isfinite(value) or value <= 0:
                raise ValueError('solver.time_limit must be positive or null')
            options[key] = float(value)
        return options

    def lp(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        t0 = time.perf_counter()
        c = np.asarray(c, float)
        aub, aeq = _matrix(A_ub, len(c)), _matrix(A_eq, len(c))
        bub = np.zeros(aub.shape[0]) if b_ub is None else np.asarray(b_ub, float)
        beq = np.zeros(aeq.shape[0]) if b_eq is None else np.asarray(b_eq, float)
        lo, hi = _bounds(bounds, len(c))
        self.lp_calls += 1
        if np.any(lo > hi):
            out = SolveResult(False, 'infeasible variable bounds')
        elif self.backend == 'gurobi':
            out = self._gurobi_lp(c, aub, bub, aeq, beq, lo, hi)
        else:
            result = linprog(
                c, A_ub=aub if aub.shape[0] else None,
                b_ub=bub if aub.shape[0] else None,
                A_eq=aeq if aeq.shape[0] else None,
                b_eq=beq if aeq.shape[0] else None,
                bounds=list(zip(lo, hi)), method='highs-ds',
                options=self._time_limit({
                    'primal_feasibility_tolerance': float(self._setting('feasibility_tol')),
                    'dual_feasibility_tolerance': float(self._setting('optimality_tol'))}))
            out = SolveResult(bool(result.success), str(result.message),
                              x=result.x, optimal=bool(result.success))
            if result.success:
                out.objective = out.bound = float(result.fun)
                out.gap = 0.0
                out.dual_ub = np.asarray(result.ineqlin.marginals)
                out.dual_eq = np.asarray(result.eqlin.marginals)
        out.seconds = time.perf_counter() - t0
        self.total_seconds += out.seconds
        return out

    def milp(self, c, integrality, A, lower, upper, bounds):
        t0 = time.perf_counter()
        c, integers = np.asarray(c, float), np.asarray(integrality, int)
        a = _matrix(A, len(c))
        low, high = np.asarray(lower, float), np.asarray(upper, float)
        lo, hi = _bounds(bounds, len(c))
        self.milp_calls += 1
        if np.any(lo > hi):
            out = SolveResult(False, 'infeasible variable bounds')
        elif self.backend == 'gurobi':
            out = self._gurobi_milp(c, integers, a, low, high, lo, hi)
        else:
            r = milp(c, integrality=integers, bounds=Bounds(lo, hi),
                     constraints=LinearConstraint(a, low, high),
                     options=self._time_limit({'mip_rel_gap': float(self._setting('mip_gap'))}))
            has_solution = r.x is not None and np.all(np.isfinite(r.x))
            out = SolveResult(has_solution, str(r.message), x=r.x,
                              optimal=bool(r.success))
            if has_solution:
                out.objective = float(r.fun)
            bound = getattr(r, 'mip_dual_bound', None)
            gap = getattr(r, 'mip_gap', None)
            if bound is not None: out.bound = float(bound)
            if gap is not None: out.gap = float(gap)
        out.seconds = time.perf_counter() - t0
        self.total_seconds += out.seconds
        return out

    @staticmethod
    def _key(*matrices):
        h = hashlib.sha256()
        for matrix in matrices:
            h.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
            h.update(matrix.indptr.tobytes()); h.update(matrix.indices.tobytes())
            h.update(matrix.data.tobytes())
        return h.digest()

    def _model(self, name, lp):
        gp = self._gp
        # Scenario LPs may run in independent threads. A private environment
        # avoids sharing Gurobi's default environment between those workers.
        if self._env is None:
            self._env = gp.Env(empty=True)
            self._env.setParam('OutputFlag', 0)
            self._env.start()
        model = gp.Model(name, env=self._env)
        model.Params.OutputFlag = 0
        if self.config.get('time_limit') is not None:
            options = self._time_limit({})
            model.Params.TimeLimit = options['time_limit']
        model.Params.FeasibilityTol = float(self._setting('feasibility_tol'))
        model.Params.OptimalityTol = float(self._setting('optimality_tol'))
        model.Params.Threads = int(self._setting('ed_threads' if lp else 'threads'))
        if lp:
            model.Params.Method = int(self._setting('ed_method'))
            model.Params.LPWarmStart = int(self._setting('lp_warm_start'))
        else:
            model.Params.MIPGap = float(self._setting('mip_gap'))
            model.Params.IntFeasTol = float(self._setting('int_feas_tol'))
        return model

    def _gurobi_lp(self, c, aub, bub, aeq, beq, lo, hi):
        gp = self._gp
        key = self._key(aub, aeq)
        entry = self._cache.pop(key, None)
        if entry is None:
            model = self._model('tsco_lp', True)
            x = model.addMVar(len(c), lb=lo, ub=hi, obj=c)
            ubcon = model.addMConstr(aub, x, '<', bub) if aub.shape[0] else None
            eqcon = model.addMConstr(aeq, x, '=', beq) if aeq.shape[0] else None
            model.update()
            entry = (model, x, ubcon, eqcon, None, None)
        model, x, ubcon, eqcon, vbasis, cbasis = entry
        x.LB, x.UB, x.Obj = lo, hi, c
        if ubcon is not None: ubcon.RHS = bub
        if eqcon is not None: eqcon.RHS = beq
        model.update()
        if vbasis is not None:
            model.setAttr('VBasis', model.getVars(), vbasis)
            model.setAttr('CBasis', model.getConstrs(), cbasis)
            self.warm_starts += 1
        model.optimize()
        optimal = model.Status == gp.GRB.OPTIMAL
        out = SolveResult(optimal, f'Gurobi status {model.Status}', optimal=optimal)
        if optimal:
            out.x = np.asarray(x.X)
            out.objective = out.bound = float(model.ObjVal)
            out.gap = 0.0
            out.dual_ub = np.asarray(ubcon.Pi) if ubcon is not None else np.empty(0)
            out.dual_eq = np.asarray(eqcon.Pi) if eqcon is not None else np.empty(0)
            vbasis = model.getAttr('VBasis', model.getVars())
            cbasis = model.getAttr('CBasis', model.getConstrs())
        else:
            vbasis = cbasis = None
        self._cache[key] = (model, x, ubcon, eqcon, vbasis, cbasis)
        return out

    def _gurobi_milp(self, c, integers, a, lower, upper, lo, hi):
        gp = self._gp
        model = self._model('tsco_master', False)
        types = np.array([gp.GRB.INTEGER if v else gp.GRB.CONTINUOUS for v in integers])
        x = model.addMVar(len(c), lb=lo, ub=hi, obj=c, vtype=types)
        equal = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
        lower_rows, upper_rows = np.isfinite(lower) & ~equal, np.isfinite(upper) & ~equal
        if np.any(equal): model.addMConstr(a[equal], x, '=', lower[equal])
        if np.any(lower_rows): model.addMConstr(a[lower_rows], x, '>', lower[lower_rows])
        if np.any(upper_rows): model.addMConstr(a[upper_rows], x, '<', upper[upper_rows])
        model.optimize()
        out = SolveResult(model.SolCount > 0, f'Gurobi status {model.Status}',
                          optimal=model.Status == gp.GRB.OPTIMAL)
        if out.success:
            out.x, out.objective = np.asarray(x.X), float(model.ObjVal)
            out.gap = float(model.MIPGap)
        try: out.bound = float(model.ObjBound)
        except gp.GurobiError: pass
        model.dispose()
        return out

    def close(self):
        for entry in self._cache.values(): entry[0].dispose()
        self._cache.clear()
        if self._env is not None:
            self._env.dispose()
            self._env = None
