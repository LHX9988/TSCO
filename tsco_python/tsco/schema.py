"""Shared, unit-explicit interfaces for the TSCO research implementation."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GridCase:
    """All bus indices are zero-based. Power MW, energy MWh, time hours."""
    name: str
    base_mva: float
    base_load: np.ndarray
    branch_from: np.ndarray
    branch_to: np.ndarray
    branch_x: np.ndarray
    branch_tap: np.ndarray
    branch_limit: np.ndarray
    gen_bus: np.ndarray
    gen_pmin: np.ndarray
    gen_pmax: np.ndarray
    gen_ramp: np.ndarray
    gen_a: np.ndarray
    gen_b: np.ndarray
    gen_c: np.ndarray
    gen_emission: np.ndarray
    gen_startup: np.ndarray
    gen_shutdown: np.ndarray
    storage_bus: np.ndarray
    storage_power: np.ndarray
    storage_capacity: np.ndarray
    storage_eta_c: np.ndarray
    storage_eta_d: np.ndarray
    storage_min: np.ndarray
    storage_max: np.ndarray
    renewable_bus: np.ndarray
    renewable_capacity: np.ndarray
    cpn_bus: np.ndarray

    @property
    def n_bus(self): return len(self.base_load)
    @property
    def n_gen(self): return len(self.gen_bus)
    @property
    def n_storage(self): return len(self.storage_bus)
    @property
    def n_renewable(self): return len(self.renewable_bus)
    @property
    def n_cpn(self): return len(self.cpn_bus)


@dataclass
class DispatchResult:
    feasible: bool
    status: str
    cost_rate: float = float('nan')
    fuel_cost_rate: float = float('nan')
    emission_rate: float = float('nan')
    p_gen: np.ndarray = field(default_factory=lambda: np.empty(0))
    renewable_used: np.ndarray = field(default_factory=lambda: np.empty(0))
    charge: np.ndarray = field(default_factory=lambda: np.empty(0))
    discharge: np.ndarray = field(default_factory=lambda: np.empty(0))
    energy_next: np.ndarray = field(default_factory=lambda: np.empty(0))
    angles: np.ndarray = field(default_factory=lambda: np.empty(0))
    flows: np.ndarray = field(default_factory=lambda: np.empty(0))
    lmp: np.ndarray = field(default_factory=lambda: np.empty(0))
    headroom_mw: float = 0.0
    storage_mode: np.ndarray = field(default_factory=lambda: np.empty(0))
    seconds: float = 0.0
    violations: dict = field(default_factory=dict)


@dataclass
class Plan:
    commitment: np.ndarray
    objective: float
    lower_bound: float
    relative_gap: float
    converged: bool
    iterations: int
    elapsed_seconds: float
    startup_cost: float = 0.0
    history: list = field(default_factory=list)
    status: str = ''


@dataclass
class Observation:
    """Arrays consumed by the graph policy. Action 0=defer, then n*3+h+1."""
    nodes: np.ndarray        # [N, node_dim]
    edges: np.ndarray        # [N,N,edge_dim]; destination,source indexing
    adjacency: np.ndarray    # [N,N] bool, includes self loops
    task: np.ndarray         # [task_dim]
    global_features: np.ndarray  # [global_dim]
    mask: np.ndarray         # [1+3*N], always permits defer


@dataclass
class Transition:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    done: bool
    discount: float = 1.0
