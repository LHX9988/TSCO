"""读取电网参数、作业、时序输入和日前场景。"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .schema import GridCase

_BUS_FIELDS = {"branch_from", "branch_to", "gen_bus", "storage_bus", "renewable_bus", "cpn_bus"}


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing required input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _blank(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    if isinstance(value, (list, tuple)):
        return any(_blank(x) for x in value)
    if isinstance(value, dict):
        return any(_blank(x) for x in value.values())
    return False


def _number(value, name):
    if _blank(value) or isinstance(value, bool):
        raise ValueError(f"Missing numeric input: {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric input: {name}") from exc
    if not np.isfinite(result):
        raise ValueError(f"Nonfinite numeric input: {name}")
    return result


def _array(value, name, shape=None, integer=False):
    if _blank(value):
        raise ValueError(f"Missing numeric array: {name}")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric array: {name}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite numeric array: {name}")
    if shape is not None and result.shape != tuple(shape):
        raise ValueError(f"{name} requires shape {tuple(shape)}, received {result.shape}")
    if integer:
        if np.any(result != np.round(result)):
            raise ValueError(f"{name} must contain integers")
        result = result.astype(int)
    return result


def nominal_case(n_cpn=5, case_path=None):
    """从填写完成的 case.json 读取名义系统。"""
    if case_path is None:
        raise ValueError("电网明细尚未填写；请填写 inputs/case.json，并通过 load_case(path) 读取。")
    case = load_case(case_path)
    if case.n_cpn != n_cpn:
        raise ValueError("Provided CPN mapping does not match the requested site count")
    return case


def validate_case(case):
    if not isinstance(case.name, str) or not case.name.strip():
        raise ValueError("Missing case name")
    if _number(case.base_mva, "base_mva") <= 0:
        raise ValueError("base_mva must be positive")
    for f in fields(GridCase):
        if f.name in {"name", "base_mva"}:
            continue
        array = _array(getattr(case, f.name), f.name)
        if array.ndim != 1:
            raise ValueError(f"{f.name} must be a one-dimensional array")
        if f.name in _BUS_FIELDS:
            if np.any(array != np.round(array)) or np.any(array < 0) or np.any(array >= case.n_bus):
                raise ValueError(f"{f.name} requires valid zero-based integer bus indices")
    if min(case.n_bus, case.n_gen, case.n_cpn) < 1:
        raise ValueError("At least one bus, generator, and CPN site are required")
    for prefix, count in (("branch_", len(case.branch_from)), ("gen_", case.n_gen),
                          ("storage_", case.n_storage), ("renewable_", case.n_renewable)):
        for f in fields(GridCase):
            if f.name.startswith(prefix) and len(getattr(case, f.name)) != count:
                raise ValueError(f"Inconsistent {prefix} field lengths")
    for name in ("branch_x", "branch_tap", "branch_limit", "gen_pmax", "storage_power",
                 "storage_capacity", "renewable_capacity"):
        if np.any(getattr(case, name) <= 0):
            raise ValueError(f"{name} must be positive")
    for name in ("base_load", "gen_pmin", "gen_ramp", "gen_a", "gen_b", "gen_c",
                 "gen_emission", "gen_startup", "gen_shutdown"):
        if np.any(getattr(case, name) < 0):
            raise ValueError(f"{name} must be nonnegative")
    if np.any(case.gen_pmin > case.gen_pmax):
        raise ValueError("Generator minimum output exceeds maximum output")
    if (np.any(case.storage_min < 0) or np.any(case.storage_min > case.storage_max)
            or np.any(case.storage_max > case.storage_capacity)):
        raise ValueError("Storage bounds must be explicit MWh within energy capacity")
    for efficiency in (case.storage_eta_c, case.storage_eta_d):
        if np.any(efficiency <= 0) or np.any(efficiency > 1):
            raise ValueError("Storage charge/discharge efficiencies must lie in (0,1]")
    seen = {0}
    while True:
        old = len(seen)
        for i, j in zip(case.branch_from, case.branch_to):
            if int(i) in seen or int(j) in seen:
                seen.update((int(i), int(j)))
        if len(seen) == old:
            break
    if len(seen) != case.n_bus:
        raise ValueError("The supplied DC network is disconnected")


def load_case(path):
    payload = _read_json(path)
    raw = payload.get("case", payload)
    if not isinstance(raw, dict):
        raise ValueError("case.json requires a case object")
    expected = {f.name for f in fields(GridCase)}
    missing = expected - set(raw)
    blank = [key for key in expected & set(raw) if _blank(raw[key])]
    if missing or blank:
        raise ValueError(f"case.json 尚未填写：{', '.join(sorted(missing | set(blank)))}")
    extra = set(raw) - expected
    if extra:
        raise ValueError(f"Unexpected case fields: {sorted(extra)}")
    converted = dict(raw)
    converted["base_mva"] = _number(raw["base_mva"], "base_mva")
    for key in expected - {"name", "base_mva"}:
        converted[key] = _array(raw[key], key, integer=key in _BUS_FIELDS)
    case = GridCase(**converted)
    validate_case(case)
    known = payload.get("known_paper_settings", {})
    counts = {"buses": case.n_bus, "thermal_generators": case.n_gen,
              "bess_units": case.n_storage, "renewable_sites": case.n_renewable,
              "cpn_sites": case.n_cpn}
    for key, actual in counts.items():
        if key in known and known[key] != actual:
            raise ValueError(f"Case {key}={actual} differs from declared {known[key]}")
    if "renewable_total_mw" in known and not np.isclose(case.renewable_capacity.sum(), known["renewable_total_mw"]):
        raise ValueError("Individual renewable capacities do not sum to the declared installed capacity")
    kinds = payload.get("renewable_types")
    if kinds is not None:
        if len(kinds) != case.n_renewable or any(k not in {"solar", "wind"} for k in kinds):
            raise ValueError("renewable_types must identify every renewable site")
        for kind in ("solar", "wind"):
            key = f"{kind}_total_mw"
            total = sum(p for p, k in zip(case.renewable_capacity, kinds) if k == kind)
            if key in known and not np.isclose(total, known[key]):
                raise ValueError(f"Individual {kind} capacities do not sum to the declared total")
    return case


def save_case(case, path):
    validate_case(case)
    payload = {f.name: (getattr(case, f.name).tolist()
                       if isinstance(getattr(case, f.name), np.ndarray)
                       else getattr(case, f.name)) for f in fields(GridCase)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case": payload}, indent=2), encoding="utf-8")


def to_pypsa(case, voltage_kv=None, initial_energy_mwh=None):
    """将填写的参数导入 PyPSA；电压等级和初始储能必须显式传入。"""
    validate_case(case)
    voltage = _array(voltage_kv, "voltage_kv", (case.n_bus,))
    energy = _array(initial_energy_mwh, "initial_energy_mwh", (case.n_storage,))
    if np.any(voltage <= 0) or not np.allclose(voltage, voltage[0]):
        raise ValueError("This DC export requires an explicitly supplied common voltage base")
    if np.any(energy < case.storage_min) or np.any(energy > case.storage_max):
        raise ValueError("Initial storage energy lies outside supplied bounds")
    try:
        import pypsa
    except ImportError as exc:
        raise ImportError("Install pypsa to export the network") from exc
    network = pypsa.Network()
    for i in range(case.n_bus):
        network.add("Bus", str(i), v_nom=float(voltage[i]))
        network.add("Load", f"base_{i}", bus=str(i), p_set=float(case.base_load[i]))
    for k, (i, j) in enumerate(zip(case.branch_from, case.branch_to)):
        network.add("Line", str(k), bus0=str(i), bus1=str(j), r=0.,
                    x=float(case.branch_x[k]*case.branch_tap[k]*voltage[i]**2/case.base_mva),
                    s_nom=float(case.branch_limit[k]))
    for g, bus in enumerate(case.gen_bus):
        network.add("Generator", f"thermal_{g}", bus=str(bus), p_nom=float(case.gen_pmax[g]),
                    p_min_pu=float(case.gen_pmin[g]/case.gen_pmax[g]),
                    marginal_cost=float(case.gen_b[g]), committable=True)
    for r, bus in enumerate(case.renewable_bus):
        network.add("Generator", f"renewable_{r}", bus=str(bus), p_nom=float(case.renewable_capacity[r]), marginal_cost=0.)
    for k, bus in enumerate(case.storage_bus):
        network.add("StorageUnit", f"bess_{k}", bus=str(bus), p_nom=float(case.storage_power[k]),
                    max_hours=float(case.storage_capacity[k]/case.storage_power[k]),
                    efficiency_store=float(case.storage_eta_c[k]), efficiency_dispatch=float(case.storage_eta_d[k]),
                    state_of_charge_initial=float(energy[k]))
    return network


def _shift_jobs(jobs, left, right):
    from .cpn import Job
    return [Job(id=j.id, arrival_seconds=j.arrival_seconds-left,
                deadline_seconds=j.deadline_seconds-left, tasks=j.tasks)
            for j in jobs if left <= j.arrival_seconds < right and j.deadline_seconds <= right]


def _parse_job_records(payload):
    from .cpn import Job, Task
    rows = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("jobs.json 尚未填写；请提供实际作业记录，空文件不表示零负载。")
    jobs, job_ids, task_ids = [], set(), set()
    for row in rows:
        required = {"id", "arrival_seconds", "deadline_seconds", "tasks"}
        if (not isinstance(row, dict) or required-set(row)
                or any(_blank(row[k]) for k in required-{"tasks"})):
            raise ValueError("Incomplete job record")
        jid = str(row["id"])
        if jid in job_ids:
            raise ValueError(f"Duplicate job ID: {jid}")
        job_ids.add(jid)
        arrival = _number(row["arrival_seconds"], f"{jid}.arrival_seconds")
        deadline = _number(row["deadline_seconds"], f"{jid}.deadline_seconds")
        if not 0 <= arrival < deadline or not row["tasks"]:
            raise ValueError(f"Invalid arrival/deadline or empty DAG for {jid}")
        tasks = []
        for item in row["tasks"]:
            keys = {"id", "workload", "demands", "compatible", "predecessor_ids", "data_gb", "revenue"}
            if not isinstance(item, dict) or keys-set(item) or any(_blank(item[k]) for k in keys):
                raise ValueError(f"Incomplete task record in {jid}")
            tid = str(item["id"])
            if tid in task_ids:
                raise ValueError(f"Task IDs must be globally unique: {tid}")
            task_ids.add(tid)
            predecessors = tuple(str(v) for v in item["predecessor_ids"])
            edge_volumes = item.get("edge_data_gb")
            if predecessors and (not isinstance(edge_volumes, dict) or set(edge_volumes) != set(predecessors)
                                 or any(_blank(v) for v in edge_volumes.values())):
                raise ValueError(f"{tid} requires explicit data volumes for every dependency edge")
            tasks.append(Task(id=tid, job_id=jid, workload=_number(item["workload"], f"{tid}.workload"),
                              demands=tuple(_array(item["demands"], f"{tid}.demands", (3,))),
                              compatible=tuple(_array(item["compatible"], f"{tid}.compatible", integer=True)),
                              predecessor_ids=predecessors, data_gb=_number(item["data_gb"], f"{tid}.data_gb"),
                              revenue=_number(item["revenue"], f"{tid}.revenue"),
                              edge_data_gb=({str(k): _number(v, f"{tid}.edge_data_gb.{k}") for k, v in edge_volumes.items()}
                                            if edge_volumes is not None else {})))
        local = {task.id: task for task in tasks}
        if any(not set(task.predecessor_ids) <= set(local) for task in tasks):
            raise ValueError(f"Missing or cross-job predecessor in {jid}")
        unresolved = set(local)
        while unresolved:
            ready = {tid for tid in unresolved if not set(local[tid].predecessor_ids) & unresolved}
            if not ready:
                raise ValueError(f"Job is not acyclic: {jid}")
            unresolved -= ready
        jobs.append(Job(id=jid, arrival_seconds=arrival, deadline_seconds=deadline, tasks=tasks))
    return sorted(jobs, key=lambda job: (job.arrival_seconds, job.id))


def _read_initial_states(path, case, intervals):
    payload = _read_json(path)
    rows = payload.get("initial_states") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("initial_states.json 尚未填写")
    states, seen = [], set()
    for row in rows:
        start = _number(row.get("start_interval"), "initial_states.start_interval")
        if start != int(start) or not 0 <= start < intervals or start in seen:
            raise ValueError("Initial-state start_interval must be unique and inside the source data")
        seen.add(start)
        commitment = _array(row.get("commitment"), "initial_state.commitment", (case.n_gen,), True)
        generation = _array(row.get("generation_mw"), "initial_state.generation_mw", (case.n_gen,))
        energy = _array(row.get("storage_energy_mwh"), "initial_state.storage_energy_mwh", (case.n_storage,))
        status_hours = _array(row.get("initial_status_hours"), "initial_state.initial_status_hours", (case.n_gen,))
        if not np.isin(commitment, [0, 1]).all():
            raise ValueError("Initial commitment must be binary")
        if np.any(generation < case.gen_pmin*commitment-1e-7) or np.any(generation > case.gen_pmax*commitment+1e-7):
            raise ValueError("Initial generation does not satisfy its commitment limits")
        if np.any(energy < case.storage_min-1e-7) or np.any(energy > case.storage_max+1e-7):
            raise ValueError("Initial storage energy is outside its supplied bounds")
        if np.any(status_hours < 0):
            raise ValueError("Initial generator status durations must be nonnegative hours")
        if not isinstance(row.get("cpn_empty"), bool):
            raise ValueError("Initial state must explicitly specify cpn_empty")
        states.append(dict(row, start_interval=int(start), commitment=commitment,
                           generation_mw=generation, storage_energy_mwh=energy,
                           initial_status_hours=status_hours))
    return sorted(states, key=lambda row: row["start_interval"])


def _read_planning_scenarios(path, case, intervals):
    payload = _read_json(path)
    rows = payload.get("planning_scenarios") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("planning_scenarios.json 尚未填写")
    results, seen = [], set()
    for row in rows:
        start = _number(row.get("start_interval"), "planning_scenarios.start_interval")
        if start != int(start) or not 0 <= start < intervals or start in seen:
            raise ValueError("Planning start_interval must be unique and inside the source data")
        seen.add(start)
        demand = _array(row.get("demand_mw"), "planning_scenarios.demand_mw")
        if demand.ndim != 3 or demand.shape[2] != case.n_bus or min(demand.shape[:2]) < 1:
            raise ValueError("Planning demand requires shape [scenarios,hours,buses]")
        count, hours, _ = demand.shape
        renewable = _array(row.get("renewable_mw"), "planning_scenarios.renewable_mw", (count, hours, case.n_renewable))
        probability = _array(row.get("probabilities"), "planning_scenarios.probabilities", (count,))
        modes = _array(row.get("storage_modes"), "planning_scenarios.storage_modes", (count, hours, case.n_storage), True)
        if (np.any(demand < 0) or np.any(renewable < 0)
                or np.any(renewable > case.renewable_capacity[None, None, :]+1e-7)
                or np.any(probability < 0) or not np.isclose(probability.sum(), 1.)):
            raise ValueError("Invalid planning powers or scenario probabilities")
        if not np.isin(modes, [-1, 0, 1]).all():
            raise ValueError("Storage modes must be -1 (discharge),0 (idle),+1 (charge)")
        hour_dt = _number(row.get("hour_dt"), "planning_scenarios.hour_dt")
        if not np.isclose(hour_dt, 1.):
            raise ValueError("Day-ahead intervals must be one hour")
        expected = row.get("expected_value_storage_modes")
        if expected is not None:
            expected = _array(expected, "expected_value_storage_modes", (hours, case.n_storage), True)
            if not np.isin(expected, [-1, 0, 1]).all():
                raise ValueError("Invalid expected-value storage modes")
        cpn_demand = row.get("cpn_demand_mw")
        if cpn_demand is not None:
            cpn_demand = _array(cpn_demand, "planning_scenarios.cpn_demand_mw", demand.shape)
            if np.any(cpn_demand < 0) or np.any(cpn_demand > demand+1e-7):
                raise ValueError("CPN forecast must be a nonnegative component of total demand")
        results.append(dict(row, start_interval=int(start), demand_mw=demand, renewable_mw=renewable,
                            probabilities=probability, storage_modes=modes, hour_dt=hour_dt,
                            expected_value_storage_modes=expected, cpn_demand_mw=cpn_demand))
    return sorted(results, key=lambda row: row["start_interval"])


def _select_records(rows, left, right, name):
    selected = [dict(row, start_interval=row["start_interval"]-left)
                for row in rows if left <= row["start_interval"] < right]
    if not selected or selected[0]["start_interval"] != 0:
        raise ValueError(f"{name} needs an explicit record at source interval {left}")
    return selected


@dataclass
class Dataset:
    case: GridCase
    base_load: np.ndarray
    renewable: np.ndarray
    times: np.ndarray
    jobs: list
    dt_seconds: float
    manifest: dict
    start_time: str
    split: str
    source_start_index: int = 0
    planning_scenarios: list = field(default_factory=list)
    initial_state: dict = field(default_factory=dict)
    initial_states: list = field(default_factory=list)
    storage_modes: np.ndarray | None = None

    @property
    def duration_seconds(self):
        return len(self.times)*self.dt_seconds

    def window(self, start=0, intervals=None):
        end = len(self.times) if intervals is None else start+intervals
        if start < 0 or end <= start or end > len(self.times):
            raise ValueError("Requested window is outside the partition")
        jobs = _shift_jobs(self.jobs, start*self.dt_seconds, end*self.dt_seconds)
        states = _select_records(self.initial_states, start, end, "initial_states")
        plans = _select_records(self.planning_scenarios, start, end, "planning_scenarios")
        if self.storage_modes is None:
            raise ValueError("Physical interval storage modes have not been provided")
        origin = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))+timedelta(seconds=start*self.dt_seconds)
        manifest = dict(self.manifest, window={"start_interval": start, "intervals": end-start,
                                              "selected_jobs": len(jobs)})
        return Dataset(self.case, self.base_load[start:end].copy(), self.renewable[start:end].copy(),
                       np.arange(end-start)*self.dt_seconds, jobs, self.dt_seconds, manifest,
                       origin.isoformat().replace("+00:00", "Z"), self.split, self.source_start_index+start,
                       plans, states[0], states, self.storage_modes[start:end].copy())


def load_dataset(directory, split="train", config=None):
    directory = Path(directory)
    config = config or {}
    case = load_case(directory/"case.json")
    dt = _number(config.get("simulation", {}).get("dt_seconds", 300.), "dt_seconds")
    if dt != 300:
        raise ValueError("The manuscript uses five-minute physical intervals")
    with (directory/"signals.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("signals.csv 尚未填写足够的实际五分钟时序数据")
    columns = [f"base_bus_{i}_mw" for i in range(case.n_bus)]
    renewable_columns = [f"res_{i}_mw" for i in range(case.n_renewable)]
    mode_columns = [f"storage_{i}_mode" for i in range(case.n_storage)]
    required = ["timestamp"]+columns+renewable_columns+mode_columns
    if set(required)-set(rows[0]):
        raise ValueError(f"Missing signal columns: {sorted(set(required)-set(rows[0]))}")
    timestamps = []
    for row in rows:
        if any(_blank(row[name]) for name in required):
            raise ValueError("signals.csv contains blank required fields")
        stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("Signal timestamps must explicitly use UTC")
        timestamps.append(stamp)
    elapsed = np.array([(stamp-timestamps[0]).total_seconds() for stamp in timestamps])
    if not np.allclose(np.diff(elapsed), dt, rtol=0, atol=1e-7):
        raise ValueError("Signal timestamps must form a regular five-minute UTC sequence")
    base = _array([[row[k] for k in columns] for row in rows], "base_bus_mw")
    renewable = _array([[row[k] for k in renewable_columns] for row in rows], "res_mw")
    storage_modes = _array([[row[k] for k in mode_columns] for row in rows], "storage_modes", integer=True)
    if (np.any(base < 0) or np.any(renewable < 0)
            or np.any(renewable > case.renewable_capacity[None, :]+1e-7)):
        raise ValueError("Invalid nodal demand or renewable availability")
    if not np.isin(storage_modes, [-1, 0, 1]).all():
        raise ValueError("Physical storage modes must be -1,0,+1")
    ratios = _array(config.get("data", {}).get("split", [.70, .15, .15]), "data.split", (3,))
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.):
        raise ValueError("Data split must contain three positive fractions summing to one")
    edges = [0, int(np.floor(len(rows)*ratios[0]+1e-9)),
             int(np.floor(len(rows)*(ratios[0]+ratios[1])+1e-9)), len(rows)]
    if split == "all":
        left, right = 0, len(rows)
    else:
        try:
            k = ["train", "validation", "test"].index(split)
        except ValueError as exc:
            raise ValueError("split must be train, validation, test, or all") from exc
        left, right = edges[k:k+2]
    if right <= left:
        raise ValueError("Selected partition is empty")
    jobs = _parse_job_records(_read_json(directory/"jobs.json"))
    from .cpn import CPNSimulator
    CPNSimulator(case.n_cpn, config, jobs)
    states = _read_initial_states(directory/"initial_states.json", case, len(rows))
    plans = _read_planning_scenarios(directory/"planning_scenarios.json", case, len(rows))
    states = _select_records(states, left, right, "initial_states")
    plans = _select_records(plans, left, right, "planning_scenarios")
    selected_jobs = _shift_jobs(jobs, left*dt, right*dt)
    manifest_path = directory/"manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be an object")
    manifest["partition"] = {"split": split, "source_start_interval": left,
                             "source_end_interval": right, "fractions": ratios.tolist(),
                             "selected_jobs": len(selected_jobs), "source_jobs": len(jobs)}
    return Dataset(case, base[left:right].copy(), renewable[left:right].copy(), np.arange(right-left)*dt,
                   selected_jobs, dt, manifest, timestamps[left].isoformat().replace("+00:00", "Z"),
                   split, left, plans, states[0], states, storage_modes[left:right].copy())
