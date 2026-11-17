"""Event-driven heterogeneous CPN and DAG communication model.

Units are seconds, FLOPs, GB, normalized resource units, and MW.  In
particular ``bandwidth_gbps`` is retained as a configuration spelling but
means GB/s (not Gbit/s).  The simulator reserves an assignment's compute
resources immediately until completion, including time spent waiting for
its inputs.  This conservative reservation rule prevents a later assignment
from overbooking a future execution interval.  Electrical demand, however,
includes dynamic power only while computation is actually running.

Every dependency requires an explicit ``edge_data_gb`` volume.  The code does
not infer dependency volumes by dividing aggregate task data.  Same-node
dependencies have no network transfer.  A directed
link serves one full-bandwidth transfer at a time; propagation does not hold
the link after the bytes have finished transmission.  External ingress,
storage I/O, and checkpoint creation are not modeled.

Hardware power allocations, processing rates, link properties, prescribed
resource ceilings, and task revenues must be supplied by the caller.  The
manuscript does not provide enough numbers to derive those inputs.  A
``reservation_fraction`` input explicitly encodes the supplied resource
ceiling divided by hardware capacity; it is not a fitted or default factor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import inf
from typing import Iterable, Mapping

import numpy as np

EPS = 1e-9
HARDWARE = ("CPU", "GPU", "TPU")


@dataclass(frozen=True)
class Task:
    id: str
    job_id: str
    workload: float
    demands: tuple[float, float, float]
    compatible: tuple[int, ...]
    predecessor_ids: tuple[str, ...]
    data_gb: float
    revenue: float | None = None
    edge_data_gb: Mapping[str, float] | None = None


@dataclass
class Job:
    id: str
    arrival_seconds: float
    deadline_seconds: float
    tasks: list[Task]


@dataclass(frozen=True)
class Transfer:
    parent_id: str
    child_id: str
    source: int
    destination: int
    start: float
    end: float
    arrival: float
    data_gb: float


@dataclass(frozen=True)
class Candidate:
    feasible: bool
    reason: str
    start: float = inf
    finish: float = inf
    delta_mw: float = 0.0
    transfers: tuple[Transfer, ...] = ()
    cross_node_inputs: int = 0


@dataclass(frozen=True)
class Execution:
    task_id: str
    node: int
    hardware: int
    resources: float
    reserved_at: float
    start: float
    finish: float
    cross_node_inputs: int
    transfers: tuple[Transfer, ...] = ()


def _matrix(value, n: int, width: int | None = None) -> np.ndarray:
    """Broadcast a scalar, hardware vector, or full array without aliasing."""
    shape = (n, n if width is None else width)
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return np.full(shape, float(a))
    if a.shape == shape:
        return a.copy()
    if width is not None and a.shape == (width,):
        return np.tile(a, (n, 1))
    raise ValueError(f"Expected a scalar or shape {shape}; received {a.shape}")


class CPNSimulator:
    """DAG execution, reservations, power, and job-level metrics.

    ``check`` is read-only and ``assign`` is atomic on invalid input.  Clones
    include every event, transfer reservation, and counter, so a rejected
    closed-loop trial cannot leak an executed transition into its parent.
    The controller, not this class, provides grid feasibility checks.
    """

    def __init__(self, n_sites: int, config: dict, jobs: Iterable[Job],
                 linear_power: bool = False,
                 ignore_communication: bool = False):
        if n_sites < 1:
            raise ValueError("n_sites must be positive")
        self.n_sites = int(n_sites)
        self.config = deepcopy(config)
        c = config.get("cpn", config)
        required = ("idle_mw", "peak_mw", "alpha", "flops_per_unit_second",
                    "reservation_fraction", "bandwidth_gbps", "latency_seconds")
        missing = [key for key in required if c.get(key) is None]
        if missing:
            raise ValueError("Missing required CPN inputs: " + ", ".join(missing))
        self.capacities = _matrix(c.get("capacities", [128, 64, 32]), n_sites, 3)
        self.idle_mw = _matrix(c["idle_mw"], n_sites, 3)
        self.peak_mw = _matrix(c["peak_mw"], n_sites, 3)
        self.alpha = _matrix(c["alpha"], n_sites, 3)
        self.rates = _matrix(c["flops_per_unit_second"], n_sites, 3)
        self.reservation_fraction = _matrix(c["reservation_fraction"], n_sites, 3)
        self.bandwidth = _matrix(c["bandwidth_gbps"], n_sites)
        self.latency = _matrix(c["latency_seconds"], n_sites)
        np.fill_diagonal(self.bandwidth, 0.0)
        np.fill_diagonal(self.latency, 0.0)
        configured_deferral = c.get("deferral_seconds")
        self.deferral_seconds = (None if configured_deferral is None
                                 else float(configured_deferral))
        self.site_power_limit = np.broadcast_to(
            np.asarray(c.get("site_peak_mw", 4.0), dtype=float), (n_sites,)
        ).copy()
        if (np.any(self.capacities <= 0) or np.any(self.rates <= 0)
                or np.any(self.idle_mw < 0) or np.any(self.peak_mw < self.idle_mw)
                or np.any(self.alpha < 0) or np.any(self.alpha > 1)
                or np.any(self.bandwidth < 0) or np.any(self.latency < 0)
                or np.any(self.reservation_fraction < 0)
                or np.any(self.reservation_fraction > 1)
                or (self.deferral_seconds is not None
                    and (not np.isfinite(self.deferral_seconds) or self.deferral_seconds <= 0))):
            raise ValueError("Invalid CPN capacities, powers, rates, or link settings")
        if np.any(self.idle_mw.sum(axis=1) > self.site_power_limit + EPS):
            raise ValueError("Idle power exceeds site rating")
        if not np.isfinite(self.site_power_limit).all() or np.any(self.site_power_limit <= 0):
            raise ValueError("Site ratings must be finite and positive")
        arrays = (self.capacities, self.idle_mw, self.peak_mw, self.alpha,
                  self.rates, self.reservation_fraction, self.bandwidth, self.latency)
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("CPN parameters must be finite")
        self.linear_power = bool(linear_power)
        self.ignore_communication = bool(ignore_communication)
        job_list = list(jobs)
        self.jobs = {j.id: j for j in job_list}
        if len(self.jobs) != len(job_list):
            raise ValueError("Duplicate job identifier")
        self.tasks: dict[str, Task] = {}
        for job in job_list:
            if (not np.isfinite(job.arrival_seconds)
                    or not np.isfinite(job.deadline_seconds)
                    or job.arrival_seconds < 0
                    or job.deadline_seconds < job.arrival_seconds
                    or not job.tasks):
                raise ValueError(f"Invalid arrival/deadline or empty DAG: {job.id}")
            local = {task.id for task in job.tasks}
            if len(local) != len(job.tasks):
                raise ValueError(f"Duplicate task identifier in job {job.id}")
            for task in job.tasks:
                if task.id in self.tasks or task.job_id != job.id:
                    raise ValueError(f"Non-unique task or incorrect job mapping: {task.id}")
                if task.revenue is None:
                    raise ValueError(f"Task {task.id} requires an explicit revenue; the manuscript does not specify its value")
                if (not np.isfinite(task.workload) or task.workload <= 0
                        or not np.isfinite(task.data_gb) or task.data_gb < 0
                        or not np.isfinite(task.revenue)
                        or len(task.demands) != 3 or not task.compatible
                        or len(set(task.compatible)) != len(task.compatible)
                        or any(h not in (0, 1, 2) for h in task.compatible)
                        or any(not np.isfinite(r) or r < 0 for r in task.demands)
                        or any(task.demands[h] <= 0 for h in task.compatible)):
                    raise ValueError(f"Invalid task requirements: {task.id}")
                if (task.id in task.predecessor_ids
                        or not set(task.predecessor_ids).issubset(local)
                        or len(set(task.predecessor_ids)) != len(task.predecessor_ids)):
                    raise ValueError(f"Invalid predecessor set: {task.id}")
                if task.predecessor_ids and task.edge_data_gb is None:
                    raise ValueError(f"Task {task.id} requires explicit edge_data_gb for all predecessors")
                if task.edge_data_gb is not None:
                    if set(task.edge_data_gb) != set(task.predecessor_ids):
                        raise ValueError("edge_data_gb must define every predecessor")
                    if any(not np.isfinite(x) or x < 0 for x in task.edge_data_gb.values()):
                        raise ValueError("Invalid per-edge data volume")
                self.tasks[task.id] = task
            unresolved = set(local)
            while unresolved:
                roots = {tid for tid in unresolved
                         if not set(self.tasks[tid].predecessor_ids) & unresolved}
                if not roots:
                    raise ValueError(f"Job is not acyclic: {job.id}")
                unresolved -= roots
        self.now = 0.0
        self.executions: dict[str, Execution] = {}
        self.task_status = {tid: "pending" for tid in self.tasks}
        self.deferred_until: dict[str, float] = {}
        self.link_reservations: dict[tuple[int, int], list[Transfer]] = {}
        self.completed_jobs: dict[str, float] = {}
        self.energy_mwh = np.zeros(n_sites)
        self.deferrals = 0

    def clone(self) -> "CPNSimulator":
        return deepcopy(self)

    def used_resources(self) -> np.ndarray:
        """Resources held now, including tasks awaiting transferred inputs."""
        result = np.zeros_like(self.capacities)
        for ex in self.executions.values():
            if ex.finish > self.now + EPS:
                result[ex.node, ex.hardware] += ex.resources
        return result

    def free_resources(self) -> np.ndarray:
        return np.maximum(0.0, self.capacities * self.reservation_fraction - self.used_resources())

    def _active_resources(self, at: float) -> np.ndarray:
        used = np.zeros_like(self.capacities)
        for ex in self.executions.values():
            if ex.start <= at + EPS and ex.finish > at + EPS:
                used[ex.node, ex.hardware] += ex.resources
        return used

    def power_from_resources(self, resources: np.ndarray) -> np.ndarray:
        util = np.asarray(resources) / self.capacities
        # Do not clip an invalid utilization to hide a capacity violation.
        if np.any(util < -1e-8) or np.any(util > 1 + 1e-8):
            raise ValueError("Resource utilization is outside [0,1]")
        a = 0.0 if self.linear_power else self.alpha
        return (self.idle_mw + (a * util**2 + (1 - a) * util)
                * (self.peak_mw - self.idle_mw)).sum(axis=1)

    def power(self) -> np.ndarray:
        """Instantaneous electrical power MW; pending transfers consume no compute dynamic power."""
        return self.power_from_resources(self._active_resources(self.now))

    def reserved_power(self) -> np.ndarray:
        """Conservative power envelope if all currently reserved tasks run together."""
        return self.power_from_resources(self.used_resources())

    def ready_tasks(self) -> list[Task]:
        ready = []
        for task in self.tasks.values():
            job = self.jobs[task.job_id]
            if (self.task_status[task.id] == "pending"
                    and job.arrival_seconds <= self.now + EPS
                    and self.deferred_until.get(task.id, 0.0) <= self.now + EPS
                    and all(self.task_status[p] == "completed" for p in task.predecessor_ids)):
                ready.append(task)
        return sorted(ready, key=lambda t: (self.jobs[t.job_id].deadline_seconds,
                                           self.jobs[t.job_id].arrival_seconds, t.id))

    def _transfer_slot(self, source: int, destination: int, duration: float,
                       extra: list[Transfer]) -> float:
        start = self.now
        reservations = self.link_reservations.get((source, destination), [])
        relevant = sorted([*reservations, *(r for r in extra
                                            if r.source == source and r.destination == destination)],
                          key=lambda r: (r.start, r.end))
        for transfer in relevant:
            if transfer.end <= start + EPS:
                continue
            if start + duration <= transfer.start + EPS:
                break
            start = transfer.end
        return start

    def check(self, task_id: str, node: int, hardware: int) -> Candidate:
        """Return a read-only CPN feasibility check and a proposed reservation."""
        if task_id not in self.tasks:
            return Candidate(False, "unknown_task")
        if not (0 <= node < self.n_sites and hardware in (0, 1, 2)):
            return Candidate(False, "invalid_node_or_hardware")
        task = self.tasks[task_id]
        job = self.jobs[task.job_id]
        if self.task_status[task_id] != "pending":
            return Candidate(False, "already_assigned")
        if job.arrival_seconds > self.now + EPS:
            return Candidate(False, "not_arrived")
        if self.deferred_until.get(task_id, 0.0) > self.now + EPS:
            return Candidate(False, "deferred")
        if not all(self.task_status[p] == "completed" for p in task.predecessor_ids):
            return Candidate(False, "predecessor_incomplete")
        if hardware not in task.compatible:
            return Candidate(False, "incompatible_hardware")
        used = self.used_resources()
        if used[node, hardware] + task.demands[hardware] > (
                self.capacities[node, hardware] * self.reservation_fraction[node, hardware] + EPS):
            return Candidate(False, "resource_capacity")
        power_before = self.power_from_resources(used)
        used[node, hardware] += task.demands[hardware]
        power_after = self.power_from_resources(used)
        if power_after[node] > self.site_power_limit[node] + EPS:
            return Candidate(False, "site_power_capacity")
        start = self.now
        transfers: list[Transfer] = []
        cross_inputs = 0
        for parent_id in task.predecessor_ids:
            source = self.executions[parent_id].node
            if source == node:
                continue
            cross_inputs += 1
            if self.ignore_communication:
                continue
            bw = self.bandwidth[source, node]
            if bw <= 0:
                return Candidate(False, "no_communication_link")
            volume = task.edge_data_gb[parent_id]
            duration = volume / bw
            tx_start = self._transfer_slot(source, node, duration, transfers)
            tx_end = tx_start + duration
            arrival = tx_end + self.latency[source, node]
            transfer = Transfer(parent_id, task_id, source, node,
                                tx_start, tx_end, arrival, volume)
            transfers.append(transfer)
            start = max(start, arrival)
        duration = task.workload / (self.rates[node, hardware] * task.demands[hardware])
        return Candidate(True, "feasible", start, start + duration,
                         float(power_after[node] - power_before[node]), tuple(transfers), cross_inputs)

    def assign(self, task_id: str, node: int, hardware: int) -> Candidate:
        candidate = self.check(task_id, node, hardware)
        if not candidate.feasible:
            raise ValueError(f"Infeasible CPN assignment: {candidate.reason}")
        task = self.tasks[task_id]
        ex = Execution(task_id, node, hardware, task.demands[hardware], self.now,
                       candidate.start, candidate.finish, candidate.cross_node_inputs,
                       candidate.transfers)
        self.executions[task_id] = ex
        self.task_status[task_id] = "running" if ex.start <= self.now + EPS else "reserved"
        self.deferred_until.pop(task_id, None)
        for transfer in candidate.transfers:
            self.link_reservations.setdefault((transfer.source, transfer.destination), []).append(transfer)
        return candidate

    def defer(self, task_id: str, until: float | None = None) -> None:
        if task_id not in self.tasks or self.task_status[task_id] != "pending":
            raise ValueError("Only a pending task can be deferred")
        if until is None and self.deferral_seconds is None:
            raise ValueError("Deferral end time is unspecified; supply until or configure deferral_seconds")
        target = self.now + self.deferral_seconds if until is None else float(until)
        if not np.isfinite(target) or target <= self.now + EPS:
            raise ValueError("Deferral must end strictly after the current event")
        self.deferred_until[task_id] = target
        self.deferrals += 1

    def next_event_time(self) -> float:
        events = [job.arrival_seconds for job in self.jobs.values()
                  if job.arrival_seconds > self.now + EPS]
        events.extend(x for ex in self.executions.values() for x in (ex.start, ex.finish)
                      if x > self.now + EPS)
        events.extend(t for t in self.deferred_until.values() if t > self.now + EPS)
        for reservations in self.link_reservations.values():
            events.extend(t for r in reservations for t in (r.start, r.end, r.arrival)
                          if t > self.now + EPS)
        return min(events, default=inf)

    def available_bandwidth(self) -> np.ndarray:
        available = self.bandwidth.copy()
        for (source, destination), reservations in self.link_reservations.items():
            if any(r.start <= self.now + EPS and r.end > self.now + EPS for r in reservations):
                available[source, destination] = 0.0
        return available

    def advance_to(self, target: float) -> dict:
        """Advance physical time once and integrate exact piecewise-constant CPN energy."""
        target = float(target)
        if not np.isfinite(target) or target < self.now - EPS:
            raise ValueError("Physical time cannot move backwards or to infinity")
        before_energy = self.energy_mwh.copy()
        knots = {self.now, target}
        for ex in self.executions.values():
            knots.update(x for x in (ex.start, ex.finish) if self.now < x < target)
        sorted_knots = sorted(knots)
        for left, right in zip(sorted_knots[:-1], sorted_knots[1:]):
            p = self.power_from_resources(self._active_resources((left + right) / 2))
            self.energy_mwh += p * (right - left) / 3600.0
        self.now = max(self.now, target)
        newly_completed = []
        for task_id, ex in self.executions.items():
            if self.task_status[task_id] == "completed":
                continue
            if ex.finish <= self.now + EPS:
                self.task_status[task_id] = "completed"
                newly_completed.append(task_id)
            elif ex.start <= self.now + EPS:
                self.task_status[task_id] = "running"
        new_job_completions = []
        tardiness_events = []
        for job in self.jobs.values():
            if job.id in self.completed_jobs:
                continue
            if all(self.task_status[t.id] == "completed" for t in job.tasks):
                finished = max(self.executions[t.id].finish for t in job.tasks)
                self.completed_jobs[job.id] = finished
                new_job_completions.append(job.id)
                tardiness_events.append(max(0.0, finished - job.deadline_seconds))
        return {
            "completed_task_count": len(newly_completed),
            "completed_task_ids": newly_completed,
            "completed_job_count": len(new_job_completions),
            "completed_job_ids": new_job_completions,
            "revenue": float(sum(self.tasks[tid].revenue for tid in newly_completed)),
            "tardiness_events": tardiness_events,
            "tardiness": float(sum(tardiness_events)),
            "energy_mwh": self.energy_mwh - before_energy,
        }

    def critical_path_seconds(self, task_id: str) -> float:
        """Uncongested compute-only remaining path, for the policy's task feature."""
        task = self.tasks[task_id]
        children = {t.id: [] for t in self.jobs[task.job_id].tasks}
        for t in self.jobs[task.job_id].tasks:
            for parent in t.predecessor_ids:
                children[parent].append(t.id)
        cache: dict[str, float] = {}
        def visit(tid):
            if tid not in cache:
                t = self.tasks[tid]
                duration = min(t.workload / (self.rates[n, h] * t.demands[h])
                               for n in range(self.n_sites) for h in t.compatible)
                cache[tid] = duration + max((visit(c) for c in children[tid]), default=0.0)
            return cache[tid]
        return visit(task_id)

    def metrics(self, test_end: float) -> dict:
        """QoS for jobs whose deadlines fall in [0,test_end].

        Unfinished jobs are unsuccessful and excluded from mean tardiness.
        Transfer and workload fractions are restricted to these completed
        jobs, so an unfinished job cannot inflate a completed-job numerator.
        """
        evaluated = [j for j in self.jobs.values() if j.deadline_seconds <= test_end + EPS]
        done = [j for j in evaluated
                if j.id in self.completed_jobs and self.completed_jobs[j.id] <= test_end + EPS]
        successful = [j for j in done if self.completed_jobs[j.id] <= j.deadline_seconds + EPS]
        tardiness = [max(0.0, self.completed_jobs[j.id] - j.deadline_seconds) for j in done]
        done_tasks = [t for j in done for t in j.tasks]
        total_flops = sum(t.workload for t in done_tasks)
        affected_flops = sum(t.workload for t in done_tasks
                             if self.executions[t.id].cross_node_inputs > 0)
        transfers = sum(self.executions[t.id].cross_node_inputs for t in done_tasks)
        return {
            "evaluated_jobs": len(evaluated), "completed_jobs": len(done),
            "successful_jobs": len(successful), "unfinished_jobs": len(evaluated) - len(done),
            "job_success_pct": 100.0 * len(successful) / len(evaluated) if evaluated else float("nan"),
            "average_tardiness_s": float(np.mean(tardiness)) if tardiness else float("nan"),
            "cross_node_dependency_transfers": transfers,
            "cross_node_transfers_per_completed_job": transfers / len(done) if done else float("nan"),
            "cross_node_input_workload_pct": 100 * affected_flops / total_flops if total_flops else float("nan"),
            "completed_flops": total_flops,
            "deferrals": self.deferrals,
            "cpn_energy_mwh": float(self.energy_mwh.sum()),
        }


def jobs_from_records(records: Iterable[dict]) -> list[Job]:
    """Load the documented nested job JSON schema (no pickle or code execution)."""
    jobs = []
    for record in records:
        tasks = []
        for raw in record["tasks"]:
            tasks.append(Task(
                id=str(raw["id"]), job_id=str(raw.get("job_id", record["id"])),
                workload=float(raw["workload"]), demands=tuple(map(float, raw["demands"])),
                compatible=tuple(map(int, raw["compatible"])),
                predecessor_ids=tuple(map(str, raw.get("predecessor_ids", []))),
                data_gb=float(raw["data_gb"]),
                revenue=None if raw.get("revenue") is None else float(raw["revenue"]),
                edge_data_gb=raw.get("edge_data_gb"),
            ))
        jobs.append(Job(str(record["id"]), float(record["arrival_seconds"]),
                        float(record["deadline_seconds"]), tasks))
    return jobs


def jobs_to_records(jobs: Iterable[Job]) -> list[dict]:
    return [asdict(job) for job in jobs]
