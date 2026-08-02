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

IGDec-QAOA's non-raw methods (mem/m3/trex) are evaluated with
`compute_final_suite_metrics=False` (see evaluate_single_objective_combo()'s
docstring): its subproblems are solved adaptively, so only per-circuit
metrics (qubo_energy, probability_of_optimal) are valid for a *corrected*
method -- the final-suite cost/effectiveness reported for IGDec-QAOA combos
is only ever raw's, since raw is the trajectory that was actually executed.

IGDec-QAOA's raw final suite is ALSO not computed via
evaluate_single_objective_combo()'s built-in merge_selected_tests()
(union/first-cluster-wins) -- that's correct for QAOA-TCS's genuinely
disjoint clusters, but wrong for IGDec-QAOA, whose subproblems can revisit
the same test case across iterations and must use "last write wins" over
chronological iteration order, matching the real algorithm's
`solution[case_list[i]] = bitstring[i]` update rule exactly (see
merge_igdec_solution() below).
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

    if dataset == "elevator":
        data = pd.read_csv(os.path.join(datasets_dir, "elevator.csv"), dtype={"cost": int, "input_div": float})
        return {"cost": data["cost"].tolist(), "input_diversity": data["input_div"].tolist()}
    if dataset == "elevator2":
        data = pd.read_csv(os.path.join(datasets_dir, "elevator.csv"), dtype={"cost": int, "pcount": int, "dist": int})
        return {
            "cost": data["cost"].tolist(),
            "passenger_count": data["pcount"].tolist(),
            "travel_distance": data["dist"].tolist(),
        }
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
                load_calibration_for_width(calibration_store, method, num_qubits)
                for _linear, _quadratic, num_qubits in cluster_qubos
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

        for method in methods:
            keys = [(c["cluster_id"], iteration_id, None) for c in assignments_raw]
            cluster_raw_counts = [
                cluster_counts_for_method(method, key, raw_index, trex_aggregated) for key in keys
            ]
            calibration_records = [
                load_calibration_for_width(calibration_store, method, num_qubits)
                for _linear, _quadratic, num_qubits in cluster_qubos
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
            )
            results_by_method[method].append(result)

    return results_by_method


# ---------------------------------------------------------------------------
# IGDec-QAOA single-objective
# ---------------------------------------------------------------------------

def load_qaoa_tcs_style_dataset(dataset: str, datasets_dir: str) -> Dict[str, List[float]]:
    """IGDec-QAOA's tcm script (gsdtsr/iofrol/paintcontrol) uses the same
    cost/rate CSV shape as QAOA-TCS's own bootqa datasets. elevator_two/
    elevator_three route through elevator.csv's cost/pcount/dist instead --
    see load_igdec_elevator_test_case_data()."""
    return load_qaoa_tcs_test_case_data(dataset, datasets_dir)


def merge_igdec_solution(
    circuits_metadata: Sequence[Dict[str, Any]], selected_bitstrings: Sequence[str]
) -> List[int]:
    """Replays IGDec-QAOA's actual solution-update rule -- last write wins,
    chronological iteration order -- instead of
    evaluate_single_objective_combo()'s built-in merge_selected_tests()
    (union/first-cluster-wins), which assumes QAOA-TCS's disjoint-cluster
    semantics and does NOT match IGDec-QAOA: `case_list` can revisit the
    same test case across iterations, and the real algorithm always takes
    the MOST RECENT decision (`solution[case_list[i]] = bitstring[i]`,
    exactly mirrored here), not the first one. `circuits_metadata` must
    already be in chronological (iteration) order, as persisted.
    """
    solution: Dict[int, int] = {}
    for meta, bitstring in zip(circuits_metadata, selected_bitstrings):
        bits = [int(b) for b in bitstring[::-1]]
        for case_index, test_id in enumerate(meta["case_list"]):
            if case_index < len(bits):
                solution[test_id] = bits[case_index]
    return sorted(test_id for test_id, bit in solution.items() if bit == 1)


def load_igdec_elevator_test_case_data(datasets_dir: str) -> Dict[str, List[float]]:
    import pandas as pd

    data = pd.read_csv(os.path.join(datasets_dir, "elevator.csv"), dtype={"cost": int, "pcount": int, "dist": int})
    return {
        "cost": data["cost"].tolist(),
        "passenger_count": data["pcount"].tolist(),
        "travel_distance": data["dist"].tolist(),
    }


def evaluate_igdec_qaoa_single_objective_combo(
    combo_name: str, combo_cfg: Dict[str, Any], root: str, calibration_store: CalibrationStore,
) -> Dict[str, List[Any]]:
    dataset = combo_cfg["dataset"]
    circuits_dir = combo_cfg["circuits_dir"]
    methods = combo_cfg["methods"]

    results_dir = os.path.join(root, "results", "igdec_qaoa")
    circuits_base_dir = os.path.join(root, "trained_qaoa_circuits", "igdec_qaoa", circuits_dir)

    raw_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-raw_counts.jsonl"))
    trex_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-trex-raw_counts.jsonl"))
    raw_index = index_raw_records_by_key(raw_records)
    trex_aggregated = aggregate_trex_records(trex_records) if trex_records else {}

    if circuits_dir in ("elevator_two", "elevator_three"):
        test_case_data = load_igdec_elevator_test_case_data(os.path.join(root, "datasets", "quantum_sota_datasets"))
    else:
        test_case_data = load_qaoa_tcs_style_dataset(dataset, os.path.join(root, "datasets", "quantum_sota_datasets"))

    results_by_method: Dict[str, List[Any]] = defaultdict(list)

    sampling_dirs = sorted(
        (d for d in os.listdir(circuits_base_dir) if d.startswith("sampling_")),
        key=lambda d: int(d.rsplit("_", 1)[-1]),
    )

    for sampling_dir_name in sampling_dirs:
        metadata_path = os.path.join(circuits_base_dir, sampling_dir_name, "circuits_metadata.json")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path) as f:
            circuits_metadata = json.load(f)

        assignments = [
            ClusterAssignment(cluster_id=None, cluster_test_cases=m["case_list"])
            for m in circuits_metadata
        ]
        cluster_qubos = [
            (m["linear"], {(int(i), int(j)): float(c) for i, j, c in m["quadratic"]}, m["num_qubits"])
            for m in circuits_metadata
        ]

        for method in methods:
            keys = [(None, m["iteration"], m["subproblem_index"]) for m in circuits_metadata]
            cluster_raw_counts = [
                cluster_counts_for_method(method, key, raw_index, trex_aggregated) for key in keys
            ]
            method_records_pool = trex_records if method == "trex" else raw_records
            sampling_iterations = {m["iteration"] for m in circuits_metadata}
            exp_records = [r for r in method_records_pool if r.iteration_id in sampling_iterations]

            calibration_records = [
                load_calibration_for_width(calibration_store, method, num_qubits)
                for _linear, _quadratic, num_qubits in cluster_qubos
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
                # Never let evaluate_single_objective_combo() do its own
                # merge_selected_tests() -- that's the wrong (union/
                # first-wins) semantics for IGDec-QAOA. raw's final suite is
                # computed correctly below via merge_igdec_solution()
                # instead; corrected methods stay at execution_cost=None
                # (see module docstring).
                compute_final_suite_metrics=False,
            )
            if method == "raw":
                selected_bitstrings = result.optimal_bitstring.split("+")
                final_selected_tests = merge_igdec_solution(circuits_metadata, selected_bitstrings)
                effectiveness_metrics = compute_single_objective_metrics(
                    dataset, final_selected_tests, test_case_data
                )
                result.execution_cost = effectiveness_metrics.pop("execution_cost")
                result.effectiveness = effectiveness_metrics
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
            elif algorithm == "igdec_qaoa" and objective_mode == "single_objective":
                results_by_method = evaluate_igdec_qaoa_single_objective_combo(
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
