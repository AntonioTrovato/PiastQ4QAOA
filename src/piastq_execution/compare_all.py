"""Runs the standard statistical comparisons across every combo
evaluate_all.py produced, and writes a report mapped to this study's
research questions (README.md's Quickstart section):

  RQ1/RQ3a (mitigation vs. quality, cost-agnostic): per combo, method vs
    raw on qubo_energy / probability_of_optimal (single-objective) or
    hypervolume / igd / num_non_dominated (multi-objective).
  RQ2: per combo, method vs raw on execution_cost / effectiveness.*
    (single-objective) or hypervolume / igd (multi-objective) -- same
    machinery as RQ1, different metric selection.
  RQ3b (mitigation cost/benefit): per combo, method vs raw on
    execution_time_seconds and mitigation_overhead.calibration_wall_clock_seconds
    -- reported alongside RQ1/RQ3a's quality numbers so the two can be
    related by hand, rather than collapsed into one invented composite
    index this module would have to justify methodologically.
  RQ4 (QAOA-TCS vs IGDec-QAOA): for combo pairs sharing the same `dataset`
    across algorithms (gsdtsr/iofrol/paintcontrol pair automatically;
    elevator variants use different naming per algorithm and are not
    auto-paired -- see module docstring in evaluation.py's
    EFFECTIVENESS_METRICS), same-method comparison on execution_time_seconds.

Reads results/evaluation/*.json (evaluate_all.py's output) and
configs/execution_plan.yaml (for RQ4's algorithm/dataset pairing). Pure
classical statistics -- no backend, no AQT, safe anywhere.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from piastq_execution.evaluate_all import load_combo_definitions, repo_root
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
        if not filename.endswith(".json"):
            continue
        combo_name = filename[:-len(".json")]
        with open(os.path.join(evaluation_dir, filename)) as f:
            results[combo_name] = json.load(f)
    return results


def extract_samples(records: List[Dict[str, Any]], metric_path: str) -> List[float]:
    """Pulls one scalar metric out of a list of per-repetition result
    dicts. `metric_path` supports dotted paths, e.g.
    "mitigation_overhead.calibration_wall_clock_seconds". Skips records
    where the metric is None/missing (e.g. execution_cost on IGDec-QAOA's
    non-raw methods) rather than raising, since a report should degrade
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
# Method-vs-raw comparisons (RQ1/RQ2/RQ3a/RQ3b)
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


SINGLE_OBJECTIVE_QUALITY_METRICS = ["qubo_energy", "probability_of_optimal"]
SINGLE_OBJECTIVE_EFFECTIVENESS_METRICS = ["execution_cost"]
MULTI_OBJECTIVE_QUALITY_METRICS = ["hypervolume", "igd", "num_non_dominated"]
COST_METRICS = ["execution_time_seconds", "mitigation_overhead.calibration_wall_clock_seconds"]


def _is_multi_objective_combo(combo_results: Dict[str, List[Dict[str, Any]]]) -> bool:
    for records in combo_results.values():
        if records and "hypervolume" in records[0]:
            return True
    return False


def compare_combo(combo_name: str, combo_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    quality_metrics = (
        MULTI_OBJECTIVE_QUALITY_METRICS if _is_multi_objective_combo(combo_results) else SINGLE_OBJECTIVE_QUALITY_METRICS
    )
    report: Dict[str, Any] = {"combo": combo_name, "rq1_rq3a_quality": {}, "rq2_effectiveness": {}, "rq3b_cost": {}}

    for metric in quality_metrics:
        result = compare_methods_within_combo(combo_results, metric)
        if result is not None:
            report["rq1_rq3a_quality"][metric] = result

    if not _is_multi_objective_combo(combo_results):
        for metric in SINGLE_OBJECTIVE_EFFECTIVENESS_METRICS:
            result = compare_methods_within_combo(combo_results, metric)
            if result is not None:
                report["rq2_effectiveness"][metric] = result

    for metric in COST_METRICS:
        result = compare_methods_within_combo(combo_results, metric)
        if result is not None:
            report["rq3b_cost"][metric] = result

    return report


# ---------------------------------------------------------------------------
# QAOA-TCS vs IGDec-QAOA (RQ4)
# ---------------------------------------------------------------------------

def pair_combos_by_dataset(combo_defs: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Pairs (qaoa_tcs_combo, igdec_qaoa_combo) sharing the same `dataset`
    field -- works directly for gsdtsr/iofrol/paintcontrol (both algorithms
    use the same dataset name); elevator variants use different naming per
    algorithm (QAOA-TCS: "elevator"/"elevator2", IGDec-QAOA:
    "elevator_o2"/"elevator_o3") and are NOT auto-paired here, since there
    is no verified 1:1 correspondence to pair them by (see
    evaluation.py's EFFECTIVENESS_METRICS comment).
    """
    by_dataset: Dict[str, Dict[str, str]] = {}
    for combo_name, cfg in combo_defs.items():
        dataset = cfg["dataset"]
        by_dataset.setdefault(dataset, {})[cfg["algorithm"]] = combo_name

    pairs = []
    for dataset, algos in by_dataset.items():
        if "qaoa_tcs" in algos and "igdec_qaoa" in algos:
            pairs.append({"dataset": dataset, "qaoa_tcs": algos["qaoa_tcs"], "igdec_qaoa": algos["igdec_qaoa"]})
    return pairs


def compare_algorithms(
    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    combo_defs: Dict[str, Dict[str, Any]],
    metric_path: str = "execution_time_seconds",
) -> List[Dict[str, Any]]:
    """RQ4: for each dataset with both a QAOA-TCS and an IGDec-QAOA combo,
    compares `metric_path` between the two algorithms, per shared method
    (raw vs raw, mem vs mem, ...)."""
    reports = []
    for pair in pair_combos_by_dataset(combo_defs):
        qaoa_tcs_results = all_results.get(pair["qaoa_tcs"], {})
        igdec_results = all_results.get(pair["igdec_qaoa"], {})
        shared_methods = set(qaoa_tcs_results.keys()) & set(igdec_results.keys())

        per_method = {}
        for method in sorted(shared_methods):
            groups = {
                "qaoa_tcs": extract_samples(qaoa_tcs_results[method], metric_path),
                "igdec_qaoa": extract_samples(igdec_results[method], metric_path),
            }
            groups = {k: v for k, v in groups.items() if v}
            if len(groups) == 2:
                try:
                    per_method[method] = compare_groups(groups)
                except ValueError:
                    continue

        if per_method:
            reports.append({
                "dataset": pair["dataset"],
                "qaoa_tcs_combo": pair["qaoa_tcs"],
                "igdec_qaoa_combo": pair["igdec_qaoa"],
                "metric": metric_path,
                "by_method": per_method,
            })
    return reports


# ---------------------------------------------------------------------------
# Top-level driver + rendering
# ---------------------------------------------------------------------------

def compare_all(
    evaluation_dir: str, execution_plan_path: str
) -> Dict[str, Any]:
    all_results = load_evaluation_results(evaluation_dir)
    combo_defs = load_combo_definitions(execution_plan_path)

    per_combo = [compare_combo(name, results) for name, results in all_results.items()]
    rq4 = compare_algorithms(all_results, combo_defs)

    return {"per_combo": per_combo, "rq4_algorithm_comparison": rq4}


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
            ("rq1_rq3a_quality", "RQ1/RQ3a (quality, cost-agnostic)"),
            ("rq2_effectiveness", "RQ2 (effectiveness)"),
            ("rq3b_cost", "RQ3b (cost)"),
        ]:
            metrics = combo_report[section]
            if not metrics:
                continue
            lines.append(f"  {title}:")
            for metric_name, result in metrics.items():
                lines.append(f"    {metric_name}:")
                lines.extend(_render_comparison(result))
        lines.append("")

    if comparison["rq4_algorithm_comparison"]:
        lines.append("RQ4: QAOA-TCS vs IGDec-QAOA")
        lines.append("-" * 32)
        for report in comparison["rq4_algorithm_comparison"]:
            lines.append(
                f"  {report['dataset']} ({report['qaoa_tcs_combo']} vs {report['igdec_qaoa_combo']}), "
                f"metric={report['metric']}:"
            )
            for method, result in report["by_method"].items():
                lines.append(f"    method={method}:")
                lines.extend(_render_comparison(result, indent="        "))
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
        for section in ("rq1_rq3a_quality", "rq2_effectiveness", "rq3b_cost"):
            entry[section] = {
                metric: _comparison_result_to_json(result) for metric, result in combo_report[section].items()
            }
        per_combo_json.append(entry)

    rq4_json = []
    for report in comparison["rq4_algorithm_comparison"]:
        rq4_json.append({
            "dataset": report["dataset"],
            "qaoa_tcs_combo": report["qaoa_tcs_combo"],
            "igdec_qaoa_combo": report["igdec_qaoa_combo"],
            "metric": report["metric"],
            "by_method": {
                method: _comparison_result_to_json(result) for method, result in report["by_method"].items()
            },
        })

    return {"per_combo": per_combo_json, "rq4_algorithm_comparison": rq4_json}


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
