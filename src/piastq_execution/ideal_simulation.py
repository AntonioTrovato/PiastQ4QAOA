"""Ideal (noiseless) simulator reference for the PIAST-Q hardware study.

The frozen QAOA-TCS circuits are re-executed on a noiseless simulator in three
configurations, and the results are evaluated exactly like the hardware ones:

  * ``ideal_exact``    -- exact output distribution (statevector, infinite
                          shots): what a perfect machine would return. One
                          deterministic result, repeated for every repetition.
  * ``ideal_shots200`` -- the noiseless distribution sampled with 200 shots
                          (qiskit-aer, no noise model), 10 seeded repetitions:
                          the same shot budget as the hardware raw/MEM/M3 data.
  * ``ideal_shots600`` -- the same with 600 shots: the budget of TREx
                          (3 twirled passes x 200 shots).

Why both: the hardware result differs from the exact optimum for two reasons,
device noise and finite sampling. ``ideal_exact`` measures the total gap;
``ideal_shots200`` removes the device and keeps only the sampling, so
"hardware vs ideal_shots200" isolates what the device itself (and hence what
readout mitigation) can be responsible for. ``ideal_shots600`` does the same
for TREx, whose extra passes are also extra shots.

Nothing here touches a quantum backend: it only needs the frozen circuits,
their QUBOs, the datasets and (optionally) the stored hardware counts.

    cd src/piastq_execution
    python ideal_simulation.py                 # every combo planned in configs/execution_plan.yaml
    python ideal_simulation.py flex_qaoa_tcs   # or only the listed combos

Writes:
  results/ideal_simulation/<subject>/<subject>-rep-1-<label>-counts.jsonl  (the simulated counts)
  results/evaluation/ideal_comparison.json / ideal_comparison_report.txt   (metrics + tests)
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from piastq_execution.evaluate_all import (  # noqa: E402
    cluster_counts_for_method,
    index_raw_records_by_key,
    load_calibration_for_record,
    load_combo_definitions,
    load_qaoa_tcs_test_case_data,
    load_raw_records,
    planned_combo_names,
    repo_root,
)
from piastq_execution.evaluation import (  # noqa: E402
    ClusterAssignment,
    brute_force_optimal,
    evaluate_single_objective_combo,
    merge_selected_tests,
)
from piastq_execution.mitigation import CalibrationStore, aggregate_trex_records  # noqa: E402
from piastq_execution.qubo_io import json_dict_to_cluster_qubo  # noqa: E402
from piastq_execution.statistics import (  # noqa: E402
    build_reference_front,
    compare_two_groups,
    hypervolume,
    inverted_generational_distance,
    pareto_front_indices,
)

DEFAULT_REPETITIONS = 10
SHOT_CONFIGS: Dict[str, Optional[int]] = {"ideal_exact": None, "ideal_shots200": 200, "ideal_shots600": 600}
HARDWARE_METHODS = ["raw", "mem", "m3", "trex"]
# Which ideal configuration has the same shot budget as each hardware method.
SHOT_MATCHED_REFERENCE = {"raw": "ideal_shots200", "mem": "ideal_shots200", "m3": "ideal_shots200",
                          "trex": "ideal_shots600"}


# ---------------------------------------------------------------------------
# Seeds and circuits
# ---------------------------------------------------------------------------

def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from any parts (SHA-256, not hash())."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _reps_dir(root: str, circuits_dir: str) -> str:
    return os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", circuits_dir, "rep_1")


def load_frozen_circuits(root: str, circuits_dir: str):
    """(circuits, qubos) ordered by cluster index, exactly as executed on hardware."""
    from qiskit import qpy

    reps_dir = _reps_dir(root, circuits_dir)
    with open(os.path.join(reps_dir, f"{circuits_dir}_rep1_qubos.json")) as f:
        qubos_json = sorted(json.load(f), key=lambda q: q["cluster_idx"])
    circuits = []
    for entry in qubos_json:
        with open(os.path.join(reps_dir, f"{circuits_dir}_rep1_cluster{entry['cluster_idx']}.qpy"), "rb") as f:
            circuits.append(qpy.load(f)[0])
    return circuits, qubos_json


def _with_measurements(circuit):
    if circuit.num_clbits == 0 or "measure" not in circuit.count_ops():
        circuit = circuit.copy()
        circuit.measure_all()
    return circuit


def exact_distribution(circuit) -> Dict[str, float]:
    """Exact output distribution of a circuit (no noise, infinite shots)."""
    from qiskit.quantum_info import Statevector

    probabilities = Statevector(circuit.remove_final_measurements(inplace=False)).probabilities_dict()
    return {k: float(v) for k, v in probabilities.items() if v > 1e-15}


def sample_counts(circuits: Sequence[Any], shots: int, seed: int) -> List[Dict[str, int]]:
    """Samples every circuit `shots` times on the noiseless qiskit-aer simulator
    (no noise model). One seeded job: deterministic for a given seed."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    simulator = AerSimulator()
    compiled = transpile([_with_measurements(c) for c in circuits], simulator, optimization_level=0)
    result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()
    out = []
    for i in range(len(circuits)):
        counts = result.get_counts(i)
        out.append({k.replace(" ", ""): int(v) for k, v in counts.items()})
    return out


def simulate_combo(root: str, circuits_dir: str, repetitions: int = DEFAULT_REPETITIONS,
                   circuits=None) -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    """Runs every configuration of SHOT_CONFIGS on the combo's frozen circuits.
    Returns {label: {iteration_id: [distribution or counts per cluster_idx]}}
    and writes them to results/ideal_simulation/<subject>/."""
    if circuits is None:
        circuits, _ = load_frozen_circuits(root, circuits_dir)
    out_dir = os.path.join(root, "results", "ideal_simulation", circuits_dir)
    os.makedirs(out_dir, exist_ok=True)

    simulated: Dict[str, Dict[int, List[Dict[str, float]]]] = {}
    exact = [exact_distribution(c) for c in circuits]
    for label, shots in SHOT_CONFIGS.items():
        per_iteration: Dict[int, List[Dict[str, float]]] = {}
        path = os.path.join(out_dir, f"{circuits_dir}-rep-1-{label}-counts.jsonl")
        with open(path, "w") as f:
            for iteration_id in range(1, repetitions + 1):
                if shots is None:
                    counts, seed = exact, None
                else:
                    seed = stable_seed(circuits_dir, label, iteration_id)
                    counts = sample_counts(circuits, shots, seed)
                per_iteration[iteration_id] = counts
                for cluster_idx, c in enumerate(counts):
                    f.write(json.dumps({
                        "circuit_id": f"{circuits_dir}_rep1_cluster{cluster_idx}",
                        "cluster_idx": cluster_idx, "iteration_id": iteration_id,
                        "configuration": label, "shots": shots, "seed": seed,
                        "simulator": "statevector (exact)" if shots is None else "qiskit-aer AerSimulator, no noise",
                        "counts": c,
                    }) + "\n")
        simulated[label] = per_iteration
        print(f"  {label}: saved {path}")
    return simulated


# ---------------------------------------------------------------------------
# Cluster assignments (which test cases each circuit selects among)
# ---------------------------------------------------------------------------

def _assignments_from_subsuites(root: str, circuits_dir: str) -> Optional[List[ClusterAssignment]]:
    path = os.path.join(root, "results", "qaoa_tcs", circuits_dir, f"{circuits_dir}-rep-1-subsuites.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        subsuites = json.load(f)
    first = sorted(subsuites, key=lambda k: int(k.rsplit("_", 1)[-1]))[0]
    return [ClusterAssignment(cluster_id=c["cluster_id"], cluster_test_cases=c["cluster_test_cases"])
            for c in subsuites[first]["cluster_assignments"]]


def _assignments_from_clustering(root: str, dataset: str, objective_mode: str) -> List[ClusterAssignment]:
    """Recomputes the deterministic Ward clustering by importing the QAOA-TCS
    script (used when no hardware subsuites file exists yet). The scripts read
    their data with paths relative to src/qaoa_tcs, hence the chdir."""
    script_dir = os.path.join(root, "src", "qaoa_tcs")
    old_cwd = os.getcwd()
    sys.path.insert(0, script_dir)
    try:
        os.chdir(script_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            if objective_mode == "single_objective":
                import single_obj as module  # noqa: WPS433
                clusters = module.bootqa_clusters[dataset]
            else:
                import multi_obj as module  # noqa: WPS433
                clusters = module.clusters_dictionary[dataset]
    finally:
        os.chdir(old_cwd)
    return [ClusterAssignment(cluster_id=int(cid), cluster_test_cases=list(tests)) for cid, tests in clusters.items()]


def load_cluster_assignments(root: str, combo_cfg: Dict[str, Any], qubos_json: Sequence[Dict[str, Any]]):
    assignments = _assignments_from_subsuites(root, combo_cfg["circuits_dir"])
    if assignments is None:
        assignments = _assignments_from_clustering(root, combo_cfg["dataset"], combo_cfg["objective_mode"])
    if len(assignments) != len(qubos_json):
        raise RuntimeError(f"{combo_cfg['circuits_dir']}: {len(assignments)} clusters but {len(qubos_json)} circuits")
    for a, q in zip(assignments, qubos_json):
        if len(a.cluster_test_cases) != q["num_qubits"]:
            raise RuntimeError(f"{combo_cfg['circuits_dir']} cluster {q['cluster_idx']}: "
                               f"{len(a.cluster_test_cases)} tests but a {q['num_qubits']}-qubit circuit")
        if "cluster_id" in q and int(q["cluster_id"]) != int(a.cluster_id):
            raise RuntimeError(f"{combo_cfg['circuits_dir']} cluster {q['cluster_idx']}: cluster id mismatch")
    return assignments


# ---------------------------------------------------------------------------
# Evaluation: ideal configurations next to the hardware methods
# ---------------------------------------------------------------------------

def _argmax_optimal_fraction(selected: Sequence[str], qubos) -> float:
    hits = 0
    for bitstring, (linear, quadratic, num_qubits) in zip(selected, qubos):
        optimum, _ = brute_force_optimal(linear, quadratic, num_qubits)
        hits += bitstring == optimum
    return hits / len(selected)


def evaluate_combo_against_ideal(root: str, combo_name: str, combo_cfg: Dict[str, Any],
                                 simulated: Dict[str, Dict[int, List[Dict[str, float]]]],
                                 assignments: Sequence[ClusterAssignment],
                                 qubos_json: Sequence[Dict[str, Any]],
                                 calibration_store: Optional[CalibrationStore]) -> Dict[str, List[Dict[str, Any]]]:
    """Per repetition and per source (hardware methods that have data + ideal
    configurations): QUBO energy, P(optimal), fraction of clusters whose
    selected bitstring is the optimum, and either the suite metrics
    (single-objective) or HV/IGD/non-dominated count (multi-objective, against
    the union front of every source of that repetition)."""
    circuits_dir, dataset = combo_cfg["circuits_dir"], combo_cfg["dataset"]
    multi = combo_cfg["objective_mode"] == "multi_objective"
    qubos = [json_dict_to_cluster_qubo(q) for q in qubos_json]

    results_dir = os.path.join(root, "results", "qaoa_tcs", circuits_dir)
    raw_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-rep-1-raw_counts.jsonl"))
    trex_records = load_raw_records(os.path.join(results_dir, f"{circuits_dir}-rep-1-trex-raw_counts.jsonl"))
    raw_index = index_raw_records_by_key(raw_records)
    trex_aggregated = aggregate_trex_records(trex_records) if trex_records else {}
    hardware_iterations = sorted({r.iteration_id for r in raw_records})

    if multi:
        from piastq_execution.sir_metrics import build_pareto_candidates, load_sir_program_data

        sir_data = load_sir_program_data(dataset, os.path.join(root, "datasets", "sir_programs"))
        test_case_data = {"cost": [0.0]}
    else:
        test_case_data = load_qaoa_tcs_test_case_data(dataset, os.path.join(root, "datasets", "quantum_sota_datasets"))

    iterations = sorted(next(iter(simulated.values())).keys())
    results: Dict[str, List[Dict[str, Any]]] = {}

    for iteration_id in iterations:
        sources: Dict[str, Tuple[List[Dict[str, float]], List[Any]]] = {}
        if iteration_id in hardware_iterations:
            keys = [(a.cluster_id, iteration_id, None) for a in assignments]
            for method in combo_cfg["methods"]:
                if method == "trex" and not trex_records:
                    continue
                counts = [cluster_counts_for_method(method, k, raw_index, trex_aggregated) for k in keys]
                calibrations = [load_calibration_for_record(calibration_store, method, raw_index.get(k), q[2])
                                for k, q in zip(keys, qubos)] if calibration_store else [None] * len(keys)
                sources[method] = (counts, calibrations)
        for label, per_iteration in simulated.items():
            sources[label] = (per_iteration[iteration_id], [None] * len(qubos))

        points_by_source: Dict[str, List[Tuple[float, ...]]] = {}
        for source, (counts, calibrations) in sources.items():
            method = source if source in HARDWARE_METHODS else "raw"
            evaluation = evaluate_single_objective_combo(
                combo=combo_name, method=method, dataset=dataset,
                cluster_assignments=assignments, cluster_raw_counts=counts, cluster_qubos=qubos,
                test_case_data=test_case_data, raw_records=[], calibration_records=calibrations,
                compute_final_suite_metrics=not multi,
            )
            selected = evaluation.optimal_bitstring.split("+")
            record: Dict[str, Any] = {
                "iteration_id": iteration_id,
                "qubo_energy": evaluation.qubo_energy,
                "probability_of_optimal": evaluation.probability_of_optimal,
                "argmax_is_optimal_fraction": _argmax_optimal_fraction(selected, qubos),
            }
            if multi:
                tests = merge_selected_tests(assignments, selected)
                points, _ = build_pareto_candidates(tests, sir_data["test_cases_costs"],
                                                    sir_data["faults_dictionary"],
                                                    sir_data["test_coverage_line_by_line"])
                points_by_source[source] = points
                record["num_selected_tests"] = len(tests)
            else:
                record["execution_cost"] = evaluation.execution_cost
                record.update({f"effectiveness.{k}": v for k, v in evaluation.effectiveness.items()})
            results.setdefault(source, []).append(record)

        if multi:
            all_points = [p for pts in points_by_source.values() for p in pts]
            reference_point = (min(p[0] for p in all_points) - 1.0, 0.0, 0.0)
            reference_front = build_reference_front(points_by_source)
            for source, points in points_by_source.items():
                front = [points[i] for i in pareto_front_indices(points)]
                results[source][-1].update({
                    "hypervolume": hypervolume(front, reference_point),
                    "igd": inverted_generational_distance(front, reference_front),
                    "num_non_dominated": sum(1 for p in front if p in reference_front),
                })
    return results


def _describe(values: Sequence[float]) -> Dict[str, float]:
    # numpy, not the stdlib `statistics`: run from this directory, that name
    # resolves to piastq_execution/statistics.py.
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)), "n": int(len(arr))}


def compare_with_shot_matched(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Each hardware method vs the ideal configuration with its shot budget
    (Shapiro-gated Welch t-test + Cohen's d, or Mann-Whitney U + A12)."""
    metrics = [k for k in next(iter(results.values()))[0] if k != "iteration_id"]
    out: Dict[str, Any] = {"descriptive": {}, "tests": {}}
    for source, records in results.items():
        out["descriptive"][source] = {m: _describe([r[m] for r in records]) for m in metrics}
    for method, reference in SHOT_MATCHED_REFERENCE.items():
        if method not in results or reference not in results:
            continue
        for metric in metrics:
            x = [r[metric] for r in results[method]]
            y = [r[metric] for r in results[reference]]
            if len(x) < 3 or len(y) < 3 or (len(set(x)) == 1 and len(set(y)) == 1):
                continue
            c = compare_two_groups(x, y)
            out["tests"].setdefault(f"{method} vs {reference}", {})[metric] = {
                "p_value": c.p_value, "effect_size": c.effect_size, "effect_size_name": c.effect_size_name,
            }
    return out


def render_report(comparison: Dict[str, Any]) -> str:
    lines = ["Ideal-simulator reference vs PIAST-Q hardware", "=" * 46, ""]
    for combo, data in comparison.items():
        lines.append(f"Combo: {combo}")
        for source, metrics in data["descriptive"].items():
            parts = [f"{m}={v['mean']:.6g}+-{v['sd']:.3g}" for m, v in metrics.items()]
            lines.append(f"  {source:15s} " + "  ".join(parts))
        for pair, tests in data["tests"].items():
            lines.append(f"  {pair}:")
            for metric, t in tests.items():
                lines.append(f"    {metric}: p={t['p_value']:.4g}, {t['effect_size_name']}={t['effect_size']:.3g}")
        lines.append("")
    return "\n".join(lines)


def run(combo_names: Optional[Sequence[str]] = None, root: Optional[str] = None,
        repetitions: int = DEFAULT_REPETITIONS) -> Dict[str, Any]:
    root = root or repo_root()
    plan_path = os.path.join(root, "configs", "execution_plan.yaml")
    combos = load_combo_definitions(plan_path)
    if combo_names:
        selected = list(combo_names)
    else:
        planned = planned_combo_names(plan_path)
        selected = [c for c in combos if c.endswith("_qaoa_tcs") and (planned is None or c in planned)]
    calibration_store = CalibrationStore(base_dir=os.path.join(root, "calibration"))

    comparison: Dict[str, Any] = {}
    all_results: Dict[str, Any] = {}
    for combo_name in selected:
        cfg = combos[combo_name]
        if cfg["algorithm"] != "qaoa_tcs":
            print(f"Skipping {combo_name}: only QAOA-TCS is supported")
            continue
        print(f"{combo_name}:")
        circuits, qubos_json = load_frozen_circuits(root, cfg["circuits_dir"])
        assignments = load_cluster_assignments(root, cfg, qubos_json)
        simulated = simulate_combo(root, cfg["circuits_dir"], repetitions, circuits)
        results = evaluate_combo_against_ideal(root, combo_name, cfg, simulated, assignments, qubos_json,
                                               calibration_store)
        all_results[combo_name] = results
        comparison[combo_name] = compare_with_shot_matched(results)

    out_dir = os.path.join(root, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ideal_comparison.json"), "w") as f:
        json.dump({"per_repetition": all_results, "comparison": comparison}, f, indent=2)
    report = render_report(comparison)
    with open(os.path.join(out_dir, "ideal_comparison_report.txt"), "w") as f:
        f.write(report)
    print(report)
    print(f"Saved: {os.path.join(out_dir, 'ideal_comparison.json')}")
    return comparison


if __name__ == "__main__":
    run(sys.argv[1:] or None)
