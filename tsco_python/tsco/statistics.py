"""Run-level inference for TSCO experiments.

``ci95`` is a two-sided Student-t confidence-interval *half-width*, not a
standard deviation or percentile interval.  Runs remain the statistical
units; dispatch intervals and tasks are never pooled as independent runs.

The summary retains any finite numeric result, including a partial failed
run's diagnostic cost, and reports exactly how many failures contribute.
Such summaries must not be interpreted as matched full-horizon performance
when failures are present.  Method comparisons therefore exclude pairs with
a failed member by default, disclose the exclusion count, and still require
identical seed sets.  Failure rates should always be reported alongside any
complete-run comparison.

Terminal-budget violation counts are separate observations.  In particular,
an explicit ``terminal_budget_violation=None`` means unknown (for example a
partial horizon) and is never inferred from a low partial emission sum.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numbers
import numpy as np
from scipy.stats import t as student_t


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _is_failed(run: Mapping) -> bool:
    for field in ("failed", "run_failed", "rollout_failed"):
        value = run.get(field)
        if isinstance(value, (bool, np.bool_)) and value:
            return True
    status = str(run.get("status", "")).strip().lower()
    return (status in {"infeasible", "terminated_infeasible", "rollout_failure", "error"}
            or status.startswith("failed"))


def _interval(values) -> dict:
    array = np.asarray(values, dtype=float)
    n = len(array)
    if n == 0:
        return {"mean": None, "std": None, "n": 0, "ci95": None,
                "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(array))
    if n == 1:
        return {"mean": mean, "std": None, "n": 1, "ci95": None,
                "ci95_low": None, "ci95_high": None}
    std = float(np.std(array, ddof=1))
    width = float(student_t.ppf(.975, n - 1) * std / math.sqrt(n))
    return {"mean": mean, "std": std, "n": n, "ci95": width,
            "ci95_low": mean - width, "ci95_high": mean + width}


def _terminal_violation(run: Mapping):
    """Return bool only for an observed completed budget horizon; else None."""
    if _is_failed(run):
        return None
    if "terminal_budget_violation" in run:
        value = run["terminal_budget_violation"]
        return bool(value) if isinstance(value, (bool, np.bool_)) else None
    # Inference is permitted only with an explicit complete-budget-horizon
    # indicator.  The budget-horizon default is the manuscript's 168 h.
    full = run.get("complete_budget_horizon") is True
    if not full:
        horizon = _number(run.get("horizon_hours"))
        budget_horizon = _number(run.get("budget_horizon_hours", 168.0))
        full = (horizon is not None and budget_horizon is not None
                and math.isclose(horizon, budget_horizon, rel_tol=0, abs_tol=1e-8))
    if not full:
        return None
    emissions = _number(run.get("emissions_t"))
    budget = _number(run.get("carbon_budget_t"))
    if emissions is None or budget is None:
        return None
    return emissions > budget


def summarize_runs(runs: Sequence[Mapping], metrics: Sequence[str] | None = None) -> dict:
    """Summarize independent run-level metrics using mean ± 95% t interval.

    Parameters
    ----------
    runs:
        Run dictionaries, typically including ``seed``, ``failed``, ``status``,
        scalar metrics, and ``terminal_budget_violation`` (bool or None).
    metrics:
        Names to summarize.  If omitted, finite numeric scalar fields other
        than ``seed`` are discovered.  Bool fields are not numeric metrics.

    Returns
    -------
    A JSON-safe dictionary with ``n_total``, ``n_failed``, ``metrics``,
    ``run_completion_rate``, and a separate ``terminal_budget`` count block.
    Every metric reports ``n_failed_contributing`` and ``n_missing``.  Empty
    samples and singleton CI/SD estimates are represented by None, not NaN.
    """
    runs = list(runs)
    if any(not isinstance(run, Mapping) for run in runs):
        raise TypeError("Each run must be a mapping")
    if metrics is None:
        metrics = sorted({key for run in runs for key, value in run.items()
                          if key != "seed" and _number(value) is not None})
    elif len(set(metrics)) != len(metrics):
        raise ValueError("Metric names must be unique")
    failed = [_is_failed(run) for run in runs]
    n_total, n_failed = len(runs), sum(failed)
    result = {"n_total": n_total, "n_failed": n_failed,
              "run_completion_rate": (n_total - n_failed) / n_total if n_total else None,
              "metrics": {},
              "note": ("Finite failed-run values are included for diagnostics; "
                       "do not interpret them as full-horizon matched performance."
                       if n_failed else "Confidence intervals use run-level observations.")}
    for name in metrics:
        observed = [(index, _number(run.get(name))) for index, run in enumerate(runs)]
        observed = [(index, value) for index, value in observed if value is not None]
        summary = _interval([value for _, value in observed])
        summary["n_missing"] = n_total - summary["n"]
        summary["n_failed_contributing"] = sum(failed[index] for index, _ in observed)
        result["metrics"][name] = summary
    violations = [_terminal_violation(run) for run in runs]
    observed = [value for value in violations if value is not None]
    n_violating = sum(observed)
    result["terminal_budget"] = {
        "n_observed": len(observed), "n_violating": n_violating,
        "n_unknown": n_total - len(observed),
        "violation_rate": n_violating / len(observed) if observed else None,
        "violation_pct": 100 * n_violating / len(observed) if observed else None,
    }
    return result


def holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm-adjust a fixed family; missing tests retain None and count in m.

    Counting uncomputed members in the family size is conservative and avoids
    silently shrinking a prespecified family when one comparison is missing.
    """
    values = list(pvalues)
    valid = []
    for index, pvalue in enumerate(values):
        if pvalue is None or (isinstance(pvalue, numbers.Real) and not np.isfinite(pvalue)):
            continue
        value = _number(pvalue)
        if value is None or not 0 <= value <= 1:
            raise ValueError("p-values must be in [0,1], None, or nonfinite")
        valid.append((index, value))
    valid.sort(key=lambda item: item[1])
    adjusted = [None] * len(values)
    running = 0.0
    for rank, (index, pvalue) in enumerate(valid):
        running = min(1.0, max(running, (len(values) - rank) * pvalue))
        adjusted[index] = running
    return adjusted


def _by_seed(runs: Sequence[Mapping], name: str) -> dict:
    indexed = {}
    for run in runs:
        if "seed" not in run:
            raise ValueError(f"{name} run is missing seed; pairing cannot be inferred from row order")
        seed = run["seed"]
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (str, numbers.Integral)):
            raise ValueError("Seed identifiers must be integers or strings")
        seed = int(seed) if isinstance(seed, numbers.Integral) else seed
        if seed in indexed:
            raise ValueError(f"Duplicate seed {seed!r} in {name}")
        indexed[seed] = run
    return indexed


def _test(a: np.ndarray, b: np.ndarray, method: str) -> tuple:
    """Return statistic, p, df; handle zero variance without warning spam."""
    n = len(a)
    if n < 2:
        return None, None, None
    delta = float(np.mean(a - b))
    if method == "paired":
        var_mean = float(np.var(a - b, ddof=1) / n)
        df = float(n - 1)
    else:
        va, vb = float(np.var(a, ddof=1) / n), float(np.var(b, ddof=1) / n)
        var_mean = va + vb
        denominator = (va * va + vb * vb) / (n - 1)
        df = var_mean * var_mean / denominator if denominator else float("inf")
    if var_mean == 0.0:
        # Infinite t is a limiting value for deterministic nonzero differences.
        # Keep JSON-safe statistic=None and disclose the case explicitly below.
        return (0.0, 1.0, df) if delta == 0.0 else (None, 0.0, df)
    statistic = delta / math.sqrt(var_mean)
    return statistic, float(2 * student_t.sf(abs(statistic), df)), df


def paired_comparisons(tsco_runs: Sequence[Mapping], baseline_runs,
                       metrics: Sequence[str], method: str = "welch",
                       include_failed: bool = False,
                       zero_tolerance: float = 1e-12) -> dict:
    """Compare run metrics after explicit seed alignment, with Holm correction.

    ``baseline_runs`` may be one run list or ``{baseline_name: run_list}``.
    Every baseline must have exactly the same unique seed set as TSCO;
    ordering is irrelevant.  The default two-sided Welch analysis matches
    the supplied manuscript's stated statistical protocol.  The optional
    ``method='paired'`` uses a paired t-test and therefore changes that
    protocol; the selected method is preserved in the output.
    Holm correction is across baseline methods *within each metric*.

    Numeric pairs missing a value are excluded, with counts returned.  Pairs
    containing a failed run are also excluded unless ``include_failed=True``.
    Relative reductions are ``100*(baseline-TSCO)/baseline``; near-zero
    denominators are omitted from relative statistics only, never silently
    replaced with epsilon.  Both per-pair reductions and the ratio of means
    are returned because they are different estimands.  A reduction is
    beneficial only for a lower-is-better metric.
    """
    if method not in {"paired", "welch"}:
        raise ValueError("method must be 'paired' or 'welch'")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    if len(set(metrics)) != len(metrics):
        raise ValueError("Metric names must be unique")
    reference = _by_seed(list(tsco_runs), "TSCO")
    groups = baseline_runs if isinstance(baseline_runs, Mapping) else {"baseline": baseline_runs}
    seeds = sorted(reference, key=lambda seed: (str(type(seed)), str(seed)))
    output = {"method": method, "alternative": "two-sided",
              "correction": "Holm across baselines within each metric",
              "n_pairs_total": len(seeds), "include_failed": bool(include_failed),
              "comparisons": {}}
    for name, runs in groups.items():
        baseline = _by_seed(list(runs), str(name))
        if set(reference) != set(baseline):
            missing = set(reference) - set(baseline)
            extra = set(baseline) - set(reference)
            raise ValueError(f"Seed mismatch for {name}: missing={missing}, extra={extra}")
        failed_seeds = [seed for seed in seeds if _is_failed(reference[seed]) or _is_failed(baseline[seed])]
        eligible = seeds if include_failed else [seed for seed in seeds if seed not in failed_seeds]
        by_metric = {}
        for metric in metrics:
            pairs = [(seed, _number(reference[seed].get(metric)), _number(baseline[seed].get(metric)))
                     for seed in eligible]
            pairs = [(seed, a, b) for seed, a, b in pairs if a is not None and b is not None]
            a = np.asarray([p[1] for p in pairs], dtype=float)
            b = np.asarray([p[2] for p in pairs], dtype=float)
            statistic, pvalue, df = _test(a, b, method)
            relative = [100 * (bv - av) / bv for av, bv in zip(a, b) if abs(bv) > zero_tolerance]
            mean_a, mean_b = (float(np.mean(a)), float(np.mean(b))) if len(pairs) else (None, None)
            ratio = (100 * (mean_b - mean_a) / mean_b
                     if mean_b is not None and abs(mean_b) > zero_tolerance else None)
            by_metric[metric] = {
                "n_pairs": len(pairs), "seeds": [seed for seed, _, _ in pairs],
                "n_failed_pairs": len(failed_seeds),
                "n_excluded_failed": 0 if include_failed else len(failed_seeds),
                "n_excluded_missing": len(eligible) - len(pairs),
                "mean_tsco": mean_a, "mean_baseline": mean_b,
                "difference_tsco_minus_baseline": _interval(a - b),
                "relative_reduction_pct": _interval(relative),
                "n_zero_baseline_excluded": len(pairs) - len(relative),
                "ratio_of_means_reduction_pct": ratio,
                "statistic": statistic, "p_raw": pvalue,
                "degrees_of_freedom": df if df is None or math.isfinite(df) else None,
                "zero_variance_nonzero_difference": bool(len(pairs) >= 2 and pvalue == 0.0 and statistic is None),
            }
        output["comparisons"][str(name)] = by_metric
    for metric in metrics:
        comparisons = [group[metric] for group in output["comparisons"].values()]
        corrected = holm_adjust([comparison["p_raw"] for comparison in comparisons])
        for comparison, pvalue in zip(comparisons, corrected):
            comparison["p_holm"] = pvalue
            comparison["significant_005"] = pvalue < .05 if pvalue is not None else None
    return output
