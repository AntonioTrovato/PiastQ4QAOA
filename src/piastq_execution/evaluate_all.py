"""Orchestrates piastq_execution.evaluation across every combo x method x
repetition, reading configs/execution_plan.yaml's `combos:` section as the
single source of truth for which combos exist and which methods to
evaluate them with -- the same file budget_planner.py uses.

Reads: trained circuits' persisted QUBOs (piastq_execution.qubo_io), each
combo's `*-raw_counts.jsonl` / `*-trex-raw_counts.jsonl` / `*-subsuites.json`
/ `circuits_metadata.json` under results/ and trained_qaoa_circuits/, and
calibration/ (MEM/M3). Writes one JSON file per combo under
results/evaluation/<combo>.json: `{method: [per-repetition metric dict, ...]}`.

Nothing here calls AQTProvider/AQTSampler/any backend -- this only reads
already-collected files and does classical post-processing, safe to run
anywhere, including in an authoring/review session, once Phase 5 (hardware
execution) and Phase 4 (calibration) data exists.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from piastq_execution.evaluation import (
    ClusterAssignment,
    compute_single_objective_metrics,
    evaluate_multi_objective_combo,
    evaluate_single_objective_combo,
    merge_selected_tests,
)
from piastq_execution.mitigation import CalibrationStore, aggregate_trex_records
from piastq_execution.qubo_io import json_dict_to_cluster_qubo
from piastq_execution.raw_counts import RawCountsRecord
from piastq_execution.sir_metrics import build_pareto_candidates, load_sir_program_data


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def load_combo_definitions(execution_plan_path: str) -> Dict[str, Dict[str, Any]]:
    """Reads configs/execution_plan.yaml's `combos:` section verbatim:
    `{combo_name: {algorithm, objective_mode, dataset, circuits_dir, methods}}`.
    """
    import yaml

    with open(execution_plan_path) as f:
        raw = yaml.safe_load(f)

    combos = {}
    for combo_name, combo_cfg in raw.get("combos", {}).items():
        combos[combo_name] = {
            "algorithm": combo_cfg["algorithm"],
            "objective_mode": combo_cfg.get("objective_mode", "single_objective"),
            "dataset": combo_cfg.get("dataset", combo_name),
            "circuits_dir": combo_cfg.get("circuits_dir", combo_cfg.get("dataset", combo_name)),
            "methods": list(combo_cfg.get("methods", ["raw"])),
        }
    return combos


def planned_combo_names(execution_plan_path: str) -> Optional[set]:
    """Combos covered by a pool in the plan (the ones this run executes), or
    None if the plan has no pools (then every defined combo is evaluated).
    Evaluating combos the run never executed would only hit stale or
    incomplete files from earlier work."""
    import yaml

    with open(execution_plan_path) as f:
        raw = yaml.safe_load(f)
    names = {c for pool in (raw.get("pools") or []) for c in pool.get("combos", [])}
    return names or None


# ---------------------------------------------------------------------------
# Raw-counts I/O
# ---------------------------------------------------------------------------

def load_raw_records(path: str) -> List[RawCountsRecord]:
    """Reads a *-raw_counts.jsonl / *-trex-raw_counts.jsonl file back into
    RawCountsRecord objects. Missing file -> empty list (e.g. a combo with
    no TREx data yet)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(RawCountsRecord(**json.loads(line)))
    return records


def index_raw_records_by_key(records: Sequence[RawCountsRecord]) -> Dict[Tuple[Any, Any, Any], RawCountsRecord]:
    """One raw/mem/m3 record per (cluster_id, iteration_id, subproblem_id) --
    each circuit is executed exactly once per repetition for the raw pass,
    so this key is always 1:1 (unlike TREx's many-per-key, see
    aggregate_trex_records())."""
    return {(r.cluster_id, r.iteration_id, r.subproblem_id): r for r in records}


def cluster_counts_for_method(
    method: str,
    key: Tuple[Any, Any, Any],
    raw_index: Dict[Tuple[Any, Any, Any], RawCountsRecord],
    trex_aggregated: Dict[Tuple[Any, Any, Any], Dict[str, int]],
) -> Dict[str, int]:
    """Looks up one circuit's counts for `method`: raw/mem/m3 all correct
    the SAME raw record (correction happens later, inside
    evaluate_single_objective_cluster's correct_counts() call); trex uses
    its own pre-aggregated (twirl-undone) counts."""
    if method == "trex":
        return trex_aggregated.get(key, {})
    record = raw_index.get(key)
    return record.aggregated_counts if record is not None else {}


# ---------------------------------------------------------------------------
# QAOA-TCS per-test-case data (cost + effectiveness columns), loaded fresh
# from the CSV -- mirrors qaoa_tcs/single_obj.py's get_data() column
# selection exactly, without importing that (heavy, side-effectful at
# import time) module.
# ---------------------------------------------------------------------------

def load_qaoa_tcs_test_case_data(dataset: str, datasets_dir: str) -> Dict[str, List[float]]:
    import pandas as pd

    data = pd.read_csv(os.path.join(datasets_dir, f"{dataset}.csv"), dtype={"time": float, "rate": float})
    data = data[data["rate"] > 0]
    return {"cost": data["time"].tolist(), "failure_rate": data["rate"].tolist()}


# ---------------------------------------------------------------------------
# Calibration lookup (per circuit width, matching run_calibration.py's
# logical-qubits-as-physical-qubits convention)
# ---------------------------------------------------------------------------

def load_calibration_for_width(store: CalibrationStore, method: str, num_qubits: int):
    if method not in ("mem", "m3"):
        return None
    return store.load(method, physical_qubits=list(range(num_qubits)))


def load_calibration_for_record(store: CalibrationStore, method: str, record, num_qubits: int):
    """Calibration for the physical qubits this circuit actually ran on (its
    record's qubit mapping -- circuits run packed at different places of the
    machine). Falls back to qubits 0..w-1 when there is no record. The store
    derives windows from the stored slot/qubit calibration."""
    if method not in ("mem", "m3"):
        return None
    if record is None or not getattr(record, "physical_qubit_mapping", None):
        return store.load_window(method, list(range(num_qubits)))
    mapping = {int(k): int(v) for k, v in record.physical_qubit_mapping.items()}
    return store.load_window(method, [mapping[i] for i in range(num_qubits)])


# ---------------------------------------------------------------------------
# QAOA-TCS single-objective
# ---------------------------------------------------------------------------

def evaluate_qaoa_tcs_single_objective_combo(
    combo_name: str, combo_cfg: Dict[str, Any], root: str, calibration_store: CalibrationStore,
) -> Dict[str, List[Any]]:
    dataset = combo_cfg["dataset"]
    circuits_dir = combo_cfg["circuits_dir"]
    methods = combo_cfg["methods"]

    results_dir = os.path.join(root, "results", "qaoa_tcs", circuits_dir)
    reps_dir = os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", circuits_dir, "rep_1")

    with open(os.path.join(reps_dir, f"{circuits_dir}_rep1_qubos.json")) as f:
        qubos_json = json.load(f)
    qubos_by_idx = {q["cluster_idx"]: json_dict_to_cluster_qubo(q) for q in qubos_json}

    with open(os.path.join(results_dir, f"{circuits_dir}-rep-1-subsuites.json")) as f:
        subsuites = json.load(f)

    raw_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-rep-1-raw_counts.jsonl"))
    trex_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-rep-1-trex-raw_counts.jsonl"))
    raw_index = index_raw_records_by_key(raw_records)
    trex_aggregated = aggregate_trex_records(trex_records) if trex_records else {}

    test_case_data = load_qaoa_tcs_test_case_data(dataset, os.path.join(root, "datasets", "quantum_sota_datasets"))

    results_by_method: Dict[str, List[Any]] = defaultdict(list)

    for exp_key in sorted(subsuites.keys(), key=lambda k: int(k.rsplit("_", 1)[-1])):
        iteration_id = int(exp_key.rsplit("_", 1)[-1])
        assignments_raw = subsuites[exp_key]["cluster_assignments"]
        assignments = [
            ClusterAssignment(cluster_id=c["cluster_id"], cluster_test_cases=c["cluster_test_cases"])
            for c in assignments_raw
        ]
        cluster_qubos = [qubos_by_idx[i] for i in range(len(assignments_raw))]

        for method in methods:
            keys = [(c["cluster_id"], iteration_id, None) for c in assignments_raw]
            cluster_raw_counts = [
                cluster_counts_for_method(method, key, raw_index, trex_aggregated) for key in keys
            ]
            method_records = trex_records if method == "trex" else raw_records
            exp_records = [r for r in method_records if r.iteration_id == iteration_id]

            calibration_records = [
                load_calibration_for_record(calibration_store, method, raw_index.get(key), num_qubits)
                for key, (_linear, _quadratic, num_qubits) in zip(keys, cluster_qubos)
            ]

            result = evaluate_single_objective_combo(
                combo=combo_name,
                method=method,
                dataset=dataset,
                cluster_assignments=assignments,
                cluster_raw_counts=cluster_raw_counts,
                cluster_qubos=cluster_qubos,
                test_case_data=test_case_data,
                raw_records=exp_records,
                calibration_records=calibration_records,
                calibration_shots_per_circuit=200,
            )
            results_by_method[method].append(result)

    return results_by_method


# ---------------------------------------------------------------------------
# QAOA-TCS multi-objective
# ---------------------------------------------------------------------------

def evaluate_qaoa_tcs_multi_objective_combo(
    combo_name: str, combo_cfg: Dict[str, Any], root: str, calibration_store: CalibrationStore,
) -> Dict[str, List[Any]]:
    dataset = combo_cfg["dataset"]
    circuits_dir = combo_cfg["circuits_dir"]
    methods = combo_cfg["methods"]

    results_dir = os.path.join(root, "results", "qaoa_tcs", circuits_dir)
    reps_dir = os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", circuits_dir, "rep_1")

    with open(os.path.join(reps_dir, f"{circuits_dir}_rep1_qubos.json")) as f:
        qubos_json = json.load(f)
    qubos_by_idx = {q["cluster_idx"]: json_dict_to_cluster_qubo(q) for q in qubos_json}

    with open(os.path.join(results_dir, f"{circuits_dir}-rep-1-subsuites.json")) as f:
        subsuites = json.load(f)

    raw_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-rep-1-raw_counts.jsonl"))
    trex_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-rep-1-trex-raw_counts.jsonl"))
    raw_index = index_raw_records_by_key(raw_records)
    trex_aggregated = aggregate_trex_records(trex_records) if trex_records else {}

    sir_data = load_sir_program_data(dataset, os.path.join(root, "datasets", "sir_programs"))

    results_by_method: Dict[str, List[Any]] = defaultdict(list)

    for exp_key in sorted(subsuites.keys(), key=lambda k: int(k.rsplit("_", 1)[-1])):
        iteration_id = int(exp_key.rsplit("_", 1)[-1])
        assignments_raw = subsuites[exp_key]["cluster_assignments"]
        assignments = [
            ClusterAssignment(cluster_id=c["cluster_id"], cluster_test_cases=c["cluster_test_cases"])
            for c in assignments_raw
        ]
        cluster_qubos = [qubos_by_idx[i] for i in range(len(assignments_raw))]

        # Every method's own Pareto-candidate points are needed up front to
        # build the reference (union) frontier -- one pass per method first.
        points_by_method: Dict[str, List[Tuple[float, float, float]]] = {}
        records_by_method: Dict[str, List[RawCountsRecord]] = {}
        calibration_by_method: Dict[str, Any] = {}

        for method in methods:
            keys = [(c["cluster_id"], iteration_id, None) for c in assignments_raw]
            cluster_raw_counts = [
                cluster_counts_for_method(method, key, raw_index, trex_aggregated) for key in keys
            ]
            calibration_records = [
                load_calibration_for_record(calibration_store, method, raw_index.get(key), num_qubits)
                for key, (_linear, _quadratic, num_qubits) in zip(keys, cluster_qubos)
            ]

            # Reuse the single-objective per-cluster correction machinery
            # (selects each cluster's corrected bitstring) but only for the
            # merge -- multi-objective's own metrics come from
            # build_pareto_candidates() below, not from this call's
            # execution_cost/effectiveness (which QAOA-TCS multi-objective
            # doesn't define -- cost is inside the 3-objective points
            # instead, see evaluation.py's module docstring).
            per_cluster_result = evaluate_single_objective_combo(
                combo=combo_name,
                method=method,
                dataset=dataset,
                cluster_assignments=assignments,
                cluster_raw_counts=cluster_raw_counts,
                cluster_qubos=cluster_qubos,
                test_case_data={"cost": [0.0]},  # unused: compute_final_suite_metrics=False
                raw_records=[],
                calibration_records=calibration_records,
                compute_final_suite_metrics=False,
            )
            selected_bitstrings = per_cluster_result.optimal_bitstring.split("+")
            final_selected_tests = merge_selected_tests(assignments, selected_bitstrings)
            points, _solutions = build_pareto_candidates(
                final_selected_tests,
                sir_data["test_cases_costs"],
                sir_data["faults_dictionary"],
                sir_data["test_coverage_line_by_line"],
            )
            points_by_method[method] = points

            method_records = trex_records if method == "trex" else raw_records
            records_by_method[method] = [r for r in method_records if r.iteration_id == iteration_id]
            calibration_by_method[method] = next((c for c in calibration_records if c is not None), None)

        # A reference point strictly worse than every real point in every
        # dimension: coverage/faults are non-negative, so 0 is a natural
        # worst case; cost (negated) needs a margin below the worst observed
        # value, or the worst point would contribute zero hypervolume.
        all_points = [p for pts in points_by_method.values() for p in pts]
        worst_cost_coord = min((p[0] for p in all_points), default=0.0)
        reference_point = (worst_cost_coord - 1.0, 0.0, 0.0)

        for method in methods:
            result = evaluate_multi_objective_combo(
                combo=combo_name,
                method=method,
                pareto_points=points_by_method[method],
                all_methods_points=points_by_method,
                reference_point=reference_point,
                raw_records=records_by_method[method],
                calibration_record=calibration_by_method.get(method),
                calibration_shots_per_circuit=200,
            )
            results_by_method[method].append(result)

    return results_by_method


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _to_json_dict(result: Any) -> Dict[str, Any]:
    d = asdict(result)
    return d


def evaluate_all(execution_plan_path: str, root: Optional[str] = None) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    root = root or repo_root()
    combos = load_combo_definitions(execution_plan_path)
    planned = planned_combo_names(execution_plan_path)
    if planned is not None:
        combos = {name: cfg for name, cfg in combos.items() if name in planned}
    calibration_store = CalibrationStore(base_dir=os.path.join(root, "calibration"))

    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for combo_name, combo_cfg in combos.items():
        algorithm = combo_cfg["algorithm"]
        objective_mode = combo_cfg["objective_mode"]

        try:
            if algorithm == "qaoa_tcs" and objective_mode == "single_objective":
                results_by_method = evaluate_qaoa_tcs_single_objective_combo(
                    combo_name, combo_cfg, root, calibration_store
                )
            elif algorithm == "qaoa_tcs" and objective_mode == "multi_objective":
                results_by_method = evaluate_qaoa_tcs_multi_objective_combo(
                    combo_name, combo_cfg, root, calibration_store
                )
            else:
                raise ValueError(f"Unsupported algorithm/objective_mode: {algorithm}/{objective_mode}")
        except FileNotFoundError as e:
            print(f"Skipping {combo_name}: {e}")
            continue

        all_results[combo_name] = {
            method: [_to_json_dict(r) for r in results]
            for method, results in results_by_method.items()
        }

    return all_results


def write_evaluation_results(all_results: Dict[str, Any], output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for combo_name, results_by_method in all_results.items():
        path = os.path.join(output_dir, f"{combo_name}.json")
        with open(path, "w") as f:
            json.dump(results_by_method, f, indent=2)
        paths.append(path)
    return paths


if __name__ == "__main__":
    root_dir = repo_root()
    config_path = os.path.join(root_dir, "configs", "execution_plan.yaml")
    output_dir = os.path.join(root_dir, "results", "evaluation")

    results = evaluate_all(config_path, root_dir)
    written = write_evaluation_results(results, output_dir)

    print(f"Evaluated {len(results)} combo(s).")
    for path in written:
        print(f"Saved: {path}")
