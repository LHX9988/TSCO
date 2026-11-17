"""Paired sensitivity experiments using supplied datasets and checkpoints."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import json_safe, save_run, write_json
from .data import Dataset, validate_case
from .features import Normalization
from .simulation import Simulator
from .statistics import summarize_runs


DEFAULT_GRIDS = {
    "res_capacity": [.6, .8, 1., 1.2, 1.4],
    "carbon_budget": [1300., 1375., 1450., 1525., 1600.],
    "cpn_power_scale": [.5, 1., 1.5],
    "reward_cost": [.5, .75, 1., 1.25, 1.5],
    "reward_carbon": [.5, .75, 1., 1.25, 1.5],
    "reward_qos": [.5, .75, 1., 1.25, 1.5],
}
_GROUPS = {
    "reward_cost": ("cost",),
    "reward_carbon": ("carbon", "queue"),
    "reward_qos": ("tardiness", "deferral", "failure"),
}
_METRICS = (
    "operational_cost_kusd", "emissions_t", "curtailment_mwh",
    "renewable_utilization_pct", "job_success_pct", "average_tardiness_s",
    "mean_interval_latency_s", "mean_cpn_mw", "cpn_energy_mwh",
    "preferred_action_infeasibility_pct", "selected_action_infeasibility_pct",
)


def _digest_json(data) -> str:
    payload = json.dumps(json_safe(data), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_hash(policy) -> str | None:
    """Audit trainable weights without importing torch for data-only tools."""
    if not hasattr(policy, "online"):
        return None
    digest = hashlib.sha256()
    for parameter in policy.online.parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def dataset_fingerprint(dataset: Dataset) -> str:
    """Hash physical inputs/jobs rather than dates or mutable run metadata."""
    digest = hashlib.sha256()
    for array in (dataset.base_load, dataset.renewable, dataset.times, dataset.storage_modes):
        if array is None:
            raise ValueError("Dataset requires explicit physical storage modes")
        a = np.ascontiguousarray(array)
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(a.tobytes())
    digest.update(_digest_json(asdict(dataset.case)).encode())
    digest.update(_digest_json([asdict(job) for job in dataset.jobs]).encode())
    digest.update(_digest_json(dataset.planning_scenarios).encode())
    digest.update(_digest_json(dataset.initial_states).encode())
    digest.update(_digest_json(dataset.initial_state).encode())
    digest.update(str((dataset.start_time, dataset.dt_seconds, dataset.split)).encode())
    return digest.hexdigest()


class StudyRunner:
    """Reuse identical held-out inputs for each seed and sensitivity setting.

    Parameters
    ----------
    config:
        Nominal evaluation configuration. It must match the trained policy's
        reward coefficients for ordinary frozen-policy checks.
    training_data, test_data:
        Already loaded chronological partitions with supplied planning
        scenarios, physical traces, storage modes, and initial states.
    checkpoint:
        A trusted checkpoint filename, optionally containing ``{seed}``. For
        reward grids this must instead be ``{multiplier: filename}``, with
        separately trained files at every requested multiplier.
    planning_mode:
        ``benders`` or ``extensive``.
    """

    def __init__(self, config: dict, training_data: Dataset, test_data: Dataset,
                 checkpoint, device: str = "cpu", planning_mode: str = "benders"):
        if training_data.split != "train" or test_data.split != "test":
            raise ValueError("Studies require separate training and held-out test partitions")
        if planning_mode not in ("benders", "extensive"):
            raise ValueError("Unknown planning mode")
        if training_data.case.n_bus != test_data.case.n_bus:
            raise ValueError("Training forecast and test data must use the same grid dimension")
        self.config = deepcopy(config)
        self.training_data, self.test_data = training_data, test_data
        self.checkpoint, self.device, self.planning_mode = checkpoint, device, planning_mode
        self.source_hashes = {
            "training": dataset_fingerprint(training_data),
            "test": dataset_fingerprint(test_data),
        }

    def _checkpoint_path(self, gridname, value, seed) -> Path:
        specification = self.checkpoint
        if gridname in _GROUPS:
            if not isinstance(specification, Mapping):
                raise ValueError("Reward-weight sensitivity requires a mapping of separately trained checkpoints")
            matches = [path for key, path in specification.items()
                       if np.isclose(float(key), value, rtol=0., atol=1e-12)]
            if len(matches) != 1:
                raise ValueError(f"Exactly one checkpoint is required for multiplier {value:g}")
            specification = matches[0]
        elif isinstance(specification, Mapping):
            raise ValueError("Frozen-policy physical grids require one checkpoint template, not a value mapping")
        path = Path(str(specification).format(seed=seed)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _load_policy(self, checkpoint: Path):
        import torch
        from .learning import DQNAgent
        torch.set_num_threads(int(self.config.get("drl", {}).get("torch_threads", 1)))
        return DQNAgent.load(checkpoint, self.device)

    def _setting(self, gridname: str, value: float):
        config = deepcopy(self.config)
        # Arrays/cases are copied; immutable Task records are shared read-only.
        def duplicate(data):
            return replace(data, case=deepcopy(data.case), base_load=data.base_load.copy(),
                           renewable=data.renewable.copy(), times=data.times.copy(),
                           manifest=deepcopy(data.manifest),
                           planning_scenarios=deepcopy(data.planning_scenarios),
                           initial_state=deepcopy(data.initial_state),
                           initial_states=deepcopy(data.initial_states),
                           storage_modes=data.storage_modes.copy())
        training, test = duplicate(self.training_data), duplicate(self.test_data)
        if gridname == "res_capacity":
            for data in (training, test):
                data.case.renewable_capacity *= value
                data.renewable *= value
                for record in data.planning_scenarios:
                    record["renewable_mw"] *= value
            meaning = "Scale RES capacities, realized availability, and supplied scenario availability equally"
        elif gridname == "carbon_budget":
            config["simulation"]["carbon_budget_t"] = value
            meaning = "Change the budget and queue normalization with the same frozen policy and reward weights"
        elif gridname == "cpn_power_scale":
            for field in ("idle_mw", "peak_mw", "site_peak_mw"):
                supplied = config["cpn"].get(field)
                if supplied is None:
                    raise ValueError(f"CPN scaling requires explicit cpn.{field}")
                scaled = np.asarray(supplied, dtype=float)*value
                if not np.isfinite(scaled).all():
                    raise ValueError(f"CPN scaling requires finite cpn.{field}")
                config["cpn"][field] = float(scaled) if scaled.ndim == 0 else scaled.tolist()
            for data in (training, test):
                if not data.planning_scenarios:
                    raise ValueError("CPN scaling requires supplied planning scenarios")
                for record in data.planning_scenarios:
                    component = record.get("cpn_demand_mw")
                    if component is None:
                        raise ValueError("CPN scaling requires explicit planning_scenarios.cpn_demand_mw")
                    record["demand_mw"] += (value-1.)*component
                    record["cpn_demand_mw"] = component*value
            meaning = ("Scale idle/peak electrical power, site limits and explicit CPN forecasts; retain CPN site count, "
                       "normalized resource capacity, FLOP throughput, jobs, and communication")
        elif gridname in _GROUPS:
            for weight in _GROUPS[gridname]:
                config["reward"]["weights"][weight] *= value
            meaning = "Evaluate separately retrained reward-group checkpoints with fixed training-fitted normalization"
        else:
            raise ValueError(f"Unsupported grid {gridname}; choose one of {tuple(DEFAULT_GRIDS)}")
        study = {"name": gridname, "value": float(value), "meaning": meaning,
                 "nominal_source_hashes": self.source_hashes}
        for data in (training, test):
            validate_case(data.case)
            data.manifest["sensitivity_setting"] = study
        return config, training, test, study

    def run_grid(self, gridname: str, values, seeds, output: str | Path,
                 hours: float = 168.) -> list[dict]:
        """Run actual simulations and write per-run logs, CSV, and t intervals."""
        if gridname not in DEFAULT_GRIDS:
            raise ValueError(f"Unknown sensitivity grid: {gridname}")
        values = [float(v) for v in (DEFAULT_GRIDS[gridname] if values is None else values)]
        seeds = list(seeds)
        if (not values or not np.isfinite(values).all() or any(v <= 0 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("Sensitivity values must be unique, positive, and finite")
        if (not seeds or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("Seeds must be unique integers")
        seeds = [int(s) for s in seeds]
        intervals = hours*3600/self.test_data.dt_seconds
        if (not np.isfinite(hours) or hours <= 0 or not np.isclose(intervals, round(intervals))
                or intervals > len(self.test_data.times)):
            raise ValueError("Study hours must cover complete available test intervals")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        # Fail before any run if checkpoints are missing or an untrained grid is requested.
        paths = {(value, seed): self._checkpoint_path(gridname, value, seed)
                 for value in values for seed in seeds}
        if gridname in _GROUPS:
            for seed in seeds:
                if len({str(paths[value, seed].resolve()) for value in values}) != len(values):
                    raise ValueError("Reward multipliers must use distinct separately trained checkpoint files")
        runs, summaries, normalization_by_seed = [], {}, {}
        spec = {
            "grid": gridname, "values": values, "seeds": seeds, "hours": float(hours),
            "planning_mode": self.planning_mode, "source_hashes": self.source_hashes,
            "policy_protocol": "separately_retrained_reward_variants" if gridname in _GROUPS else "frozen_policy",
            "data_protocol": "Input datasets are loaded once and reused across seeds; no trace/job regeneration",
            "cpn_power_scale_scope": "Electrical model scaling, not job-count or hardware-resource scaling",
            "budget_note": "Terminal compliance is evaluated only after a completed full budget horizon",
        }
        write_json(output/"study_specification.json", spec)
        for value in values:
            config, training, test, setting = self._setting(gridname, value)
            test = test.window(0, int(round(intervals)))
            physical_hashes = {"training": dataset_fingerprint(training), "test": dataset_fingerprint(test)}
            by_value = []
            for seed in seeds:
                checkpoint = paths[value, seed]
                policy = self._load_policy(checkpoint)
                if policy.steps < 1 or policy.updates < 1:
                    raise ValueError(f"Checkpoint has no completed optimizer updates: {checkpoint}")
                if policy.metadata.get("variant", "tsco") != "tsco" or policy.variant != "tsco":
                    raise ValueError("These sensitivity grids require a full TSCO checkpoint")
                data = policy.metadata.get("normalization")
                if not data or data.get("provenance", {}).get("split") != "train":
                    raise ValueError("Checkpoint must contain training-only normalization metadata")
                normalizer = Normalization.from_dict(data)
                normalizer_hash = _digest_json(data)
                if seed in normalization_by_seed and normalization_by_seed[seed] != normalizer_hash:
                    raise ValueError("Normalization must remain fixed across sensitivity values for a paired seed")
                normalization_by_seed[seed] = normalizer_hash
                expected = config["reward"]["weights"]
                trained = policy.config.get("reward", {}).get("weights", {})
                if set(trained) != set(expected) or any(not np.isclose(trained[k], expected[k], rtol=0, atol=1e-12)
                                                       for k in expected):
                    raise ValueError("Checkpoint reward weights do not match this setting; retrain rather than relabel")
                conf = deepcopy(config)
                conf["simulation"]["seed"] = seed
                conf["drl"]["seed"] = seed
                counters_before = (policy.steps, policy.updates)
                weights_before = _parameter_hash(policy)
                result = Simulator(test, training, conf, "tsco", normalizer, self.planning_mode).run(
                    policy, hours=hours, explore=False)
                if (counters_before != (policy.steps, policy.updates)
                        or weights_before != _parameter_hash(policy)):
                    raise RuntimeError("Evaluation modified the frozen policy")
                result.metrics.update(
                    study_grid=gridname, study_value=value,
                    policy_protocol=spec["policy_protocol"],
                    checkpoint=str(checkpoint), checkpoint_sha256=_file_hash(checkpoint),
                    checkpoint_transitions=policy.steps, checkpoint_updates=policy.updates,
                    frozen_policy_verified=True,
                    checkpoint_full_budget_validated=bool(policy.metadata.get("validated_full_budget_horizon", False)),
                    source_test_hash=self.source_hashes["test"], actual_input_hash=physical_hashes["test"],
                )
                directory = output/f"value_{value:g}"/f"seed_{seed:02d}"
                manifest = dict(test.manifest, study=setting, actual_input_hashes=physical_hashes)
                save_run(result, directory, conf, manifest)
                by_value.append(result.metrics)
                runs.append(result.metrics)
            summaries[f"{value:g}"] = summarize_runs(by_value, _METRICS)
            write_json(output/"runs.json", runs)
            write_json(output/"summary.json", summaries)
            pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list, np.ndarray))}
                          for row in runs]).to_csv(output/"metrics.csv", index=False)
        if (dataset_fingerprint(self.training_data) != self.source_hashes["training"]
                or dataset_fingerprint(self.test_data) != self.source_hashes["test"]):
            raise RuntimeError("A sensitivity run mutated the shared source inputs")
        return runs


def plot_study(directory: str | Path) -> Path:
    """Plot computed run-level means/95% CI; failed runs are not plotted as successes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    spec = json.loads((directory/"study_specification.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    if any(group["n_failed"] for group in summary.values()):
        raise ValueError("Failed runs are present; inspect them before plotting matched performance")
    x = np.array(spec["values"], float)
    order = np.argsort(x)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), constrained_layout=True)
    for ax, metric, label in zip(axes.flat,
        ("operational_cost_kusd", "emissions_t", "curtailment_mwh", "job_success_pct"),
        ("Operational cost (k$)", "CO2 emissions (t)", "Curtailment (MWh)", "Job success (%)")):
        records = [summary[f"{v:g}"]["metrics"][metric] for v in x]
        y = np.array([r["mean"] if r["mean"] is not None else np.nan for r in records])
        ci = np.array([r["ci95"] if r["ci95"] is not None else np.nan for r in records])
        ax.plot(x[order], y[order], "o-", lw=1.3)
        finite = np.isfinite(ci[order])
        if finite.any():
            ax.errorbar(x[order][finite], y[order][finite], yerr=ci[order][finite], fmt="none", capsize=3)
        ax.set(xlabel=spec["grid"].replace("_", " "), ylabel=label, title=label)
        ax.grid(alpha=.2)
    path = directory/"sensitivity.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path
