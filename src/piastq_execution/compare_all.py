"""Runs the standard statistical comparisons across every combo
evaluate_all.py produced, and writes a report mapped to this study's
research questions (README.md's Quickstart section):

  RQ1 (quality): per combo, every method compared on qubo_energy /
    probability_of_optimal (single-objective) or hypervolume / igd /
    num_non_dominated (multi-objective).
  RQ2 (effectiveness): per combo, execution_cost and every
    effectiveness.<name> the combo recorded (failure_rate, input_diversity,
    passenger_count, travel_distance -- discovered from the data)
    (single-objective); the multi-objective combos have no separate suite-level
    indicator beyond the Pareto metrics of RQ1.
  RQ3 (cost): billed QPU seconds of the executions and of the calibration,
    per method. These are deterministic (the billing is a function of the
    shots and of the packing, not a random variable), so they are reported
    descriptively -- mean, standard deviation and ratio to raw -- instead of
    being fed to a significance test, which would only report the zero
    variance as an astronomically large effect size. The quality numbers of
    RQ1/RQ2 are related to them by hand, rather than collapsed into one
    invented composite index this module would have to justify.

Reads results/evaluation/<combo>.json (evaluate_all.py's output) and
configs/execution_plan.yaml. Pure
classical statistics -- no backend, no AQT, safe anywhere.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from piastq_execution.evaluate_all import repo_root
from piastq_execution.statistics import ComparisonResult, compare_groups


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_evaluation_results(evaluation_dir: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Loads every results/evaluation/<combo>.json file:
    `{combo_name: {method: [per-repetition metric dict, ...]}}`."""
    results = {}
    if not os.path.isdir(evaluation_dir):
        return results
    for filename in sorted(os.listdir(evaluation_dir)):
        if not filename.endswith(".json") or filename == "comparison_report.json":
            continue
        combo_name = filename[:-len(".json")]
        with open(os.path.join(evaluation_dir, filename)) as f:
            content = json.load(f)
        # only `{method: [per-repetition dict, ...]}` files are combos; this
        # directory also holds the report itself and ideal_reference.json
        if not isinstance(content, dict) or not content or not all(
            isinstance(v, list) and all(isinstance(r, dict) for r in v) for v in content.values()
        ):
            continue
        results[combo_name] = content
    return results


def extract_samples(records: List[Dict[str, Any]], metric_path: str) -> List[float]:
    """Pulls one scalar metric out of a list of per-repetition result
    dicts. `metric_path` supports dotted paths, e.g.
    "mitigation_overhead.calibration_wall_clock_seconds". Skips records
    where the metric is None/missing (e.g. an effectiveness indicator a
    combo did not record) rather than raising, since a report should degrade
    gracefully instead of crashing on one combo's inapplicable metric.
    """
    samples = []
    for record in records:
        value: Any = record
        missing = False
        for part in metric_path.split("."):
            if not isinstance(value, dict) or part not in value or value[part] is None:
                missing = True
                break
            value = value[part]
        if not missing:
            samples.append(float(value))
    return samples


# ---------------------------------------------------------------------------
# Method-vs-raw comparisons (RQ1/RQ2) and cost summary (RQ3)
# ---------------------------------------------------------------------------

def compare_methods_within_combo(
    combo_results: Dict[str, List[Dict[str, Any]]], metric_path: str
) -> Optional[ComparisonResult]:
    """compare_groups() across every method present for this combo (raw,
    mem, m3, trex -- whichever this combo actually has), on one metric.
    Returns None if fewer than 2 methods have non-empty samples for this
    metric (e.g. a multi-objective combo asked about execution_cost), or if
    the underlying stats call can't run on this data (e.g. every sample
    identical across every group -- degenerate but real, e.g. a perfect
    probability_of_optimal=1.0 every repetition -- scipy's kruskal() raises
    rather than returning a trivial result for that case). One metric being
    degenerate shouldn't crash the whole report.
    """
    groups = {}
    for method, records in combo_results.items():
        samples = extract_samples(records, metric_path)
        if samples:
            groups[method] = samples
    if len(groups) < 2:
        return None
    try:
        return compare_groups(groups)
    except ValueError:
        return None


def describe_cost(combo_results: Dict[str, List[Dict[str, Any]]], metric_path: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Descriptive summary of a deterministic cost metric per method: mean,
    population standard deviation, number of repetitions and ratio of the mean
    to raw's mean (None when raw's mean is 0, e.g. the calibration of raw)."""
    means = {}
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for method, records in combo_results.items():
        samples = extract_samples(records, metric_path)
        if not samples:
            continue
        means[method] = sum(samples) / len(samples)
        sd = (sum((x - means[method]) ** 2 for x in samples) / len(samples)) ** 0.5   # population sd
        summary[method] = {"mean": means[method], "sd": sd, "n": len(samples)}
    raw_mean = means.get("raw")
    for method, entry in summary.items():
        entry["ratio_vs_raw"] = (means[method] / raw_mean) if raw_mean else None
    return summary


SINGLE_OBJECTIVE_QUALITY_METRICS = ["qubo_energy", "probability_of_optimal"]
SINGLE_OBJECTIVE_EFFECTIVENESS_METRICS = ["execution_cost"]
MULTI_OBJECTIVE_QUALITY_METRICS = ["hypervolume", "igd", "num_non_dominated"]
COST_METRICS = ["execution_time_seconds", "mitigation_overhead.calibration_wall_clock_seconds"]


def _effectiveness_metric_paths(combo_results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """`effectiveness.<name>` for every effectiveness metric any record of
    this combo actually has (the set differs per dataset)."""
    names: List[str] = []
    for records in combo_results.values():
        for record in records:
            for name in (record.get("effectiveness") or {}):
                path = f"effectiveness.{name}"
                if path not in names:
                    names.append(path)
    return names


def _is_multi_objective_combo(combo_results: Dict[str, List[Dict[str, Any]]]) -> bool:
    for records in combo_results.values():
        if records and "hypervolume" in records[0]:
            return True
    return False


def compare_combo(combo_name: str, combo_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    quality_metrics = (
        MULTI_OBJECTIVE_QUALITY_METRICS if _is_multi_objective_combo(combo_results) else SINGLE_OBJECTIVE_QUALITY_METRICS
    )
    report: Dict[str, Any] = {"combo": combo_name, "rq1_quality": {}, "rq2_effectiveness": {}, "rq3_cost": {}}

    for metric in quality_metrics:
        result = compare_methods_within_combo(combo_results, metric)
        if result is not None:
            report["rq1_quality"][metric] = result

    if not _is_multi_objective_combo(combo_results):
        for metric in SINGLE_OBJECTIVE_EFFECTIVENESS_METRICS + _effectiveness_metric_paths(combo_results):
            result = compare_methods_within_combo(combo_results, metric)
            if result is not None:
                report["rq2_effectiveness"][metric] = result

    for metric in COST_METRICS:
        summary = describe_cost(combo_results, metric)
        if summary:
            report["rq3_cost"][metric] = summary

    return report


# ---------------------------------------------------------------------------
# Top-level driver + rendering
# ---------------------------------------------------------------------------

def compare_all(
    evaluation_dir: str, execution_plan_path: str
) -> Dict[str, Any]:
    all_results = load_evaluation_results(evaluation_dir)

    per_combo = [compare_combo(name, results) for name, results in all_results.items()]
    return {"per_combo": per_combo}


def _render_comparison(result: ComparisonResult, indent: str = "      ") -> List[str]:
    lines = [f"{indent}normal={result.normal}, omnibus={result.omnibus_test}, p={result.omnibus_p_value:.4g}"]
    for pair in result.pairwise:
        lines.append(
            f"{indent}  {pair.group_a} vs {pair.group_b}: p_adj={pair.p_adjusted:.4g}, "
            f"{pair.effect_size_name}={pair.effect_size:.4g}"
        )
    return lines


def render_report(comparison: Dict[str, Any]) -> str:
    lines = ["PIAST-Q comparison report", "=" * 32, ""]

    for combo_report in comparison["per_combo"]:
        lines.append(f"Combo: {combo_report['combo']}")
        for section, title in [
            ("rq1_quality", "RQ1 (quality)"),
            ("rq2_effectiveness", "RQ2 (effectiveness)"),
        ]:
            metrics = combo_report[section]
            if not metrics:
                continue
            lines.append(f"  {title}:")
            for metric_name, result in metrics.items():
                lines.append(f"    {metric_name}:")
                lines.extend(_render_comparison(result))
        if combo_report["rq3_cost"]:
            lines.append("  RQ3 (cost, billed QPU seconds; deterministic, so descriptive):")
            for metric_name, summary in combo_report["rq3_cost"].items():
                lines.append(f"    {metric_name}:")
                for method, entry in summary.items():
                    ratio = "n/a" if entry["ratio_vs_raw"] is None else f"{entry['ratio_vs_raw']:.2f}x raw"
                    lines.append(f"      {method}: mean={entry['mean']:.4g}, sd={entry['sd']:.3g}, n={entry['n']}, {ratio}")
        lines.append("")

    return "\n".join(lines)


def _comparison_result_to_json(result: ComparisonResult) -> Dict[str, Any]:
    return {
        "normal": result.normal,
        "omnibus_test": result.omnibus_test,
        "omnibus_p_value": result.omnibus_p_value,
        "pairwise": [
            {
                "group_a": p.group_a,
                "group_b": p.group_b,
                "p_value": p.p_value,
                "p_adjusted": p.p_adjusted,
                "effect_size_name": p.effect_size_name,
                "effect_size": p.effect_size,
            }
            for p in result.pairwise
        ],
    }


def comparison_to_json_dict(comparison: Dict[str, Any]) -> Dict[str, Any]:
    per_combo_json = []
    for combo_report in comparison["per_combo"]:
        entry = {"combo": combo_report["combo"]}
        for section in ("rq1_quality", "rq2_effectiveness"):
            entry[section] = {
                metric: _comparison_result_to_json(result) for metric, result in combo_report[section].items()
            }
        entry["rq3_cost"] = combo_report["rq3_cost"]
        per_combo_json.append(entry)

    return {"per_combo": per_combo_json}


if __name__ == "__main__":
    root = repo_root()
    evaluation_dir = os.path.join(root, "results", "evaluation")
    execution_plan_path = os.path.join(root, "configs", "execution_plan.yaml")

    comparison = compare_all(evaluation_dir, execution_plan_path)
    report_text = render_report(comparison)
    print(report_text)

    output_dir = os.path.join(root, "results", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "comparison_report.txt"), "w") as f:
        f.write(report_text)
    with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
        json.dump(comparison_to_json_dict(comparison), f, indent=2)
    print(f"\nSaved: {os.path.join(output_dir, 'comparison_report.txt')}")
    print(f"Saved: {os.path.join(output_dir, 'comparison_report.json')}")
