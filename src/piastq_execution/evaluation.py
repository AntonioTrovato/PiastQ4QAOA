"""Per-combo x per-method evaluation metrics.

Branches on `objective_mode` ("single_objective" | "multi_objective") to pick
the right metric set, per the study design:
  - single-objective (QAOA-TCS and IGDec-QAOA, 5 datasets each): QUBO energy
    of the corrected/selected solution, probability of the optimal bitstring
    (brute-forced -- <=7-qubit subproblems are 128 states, trivial),
    execution cost and the dataset's effectiveness metric(s) of the final
    merged suite, quantum-hardware execution time, mitigation overhead,
    classical post-processing time.
  - multi-objective (flex/grep/gzip/sed): number of non-dominated solutions
    contributed to a reference Pareto frontier, Hypervolume, IGD, plus the
    same execution-time/mitigation-overhead/classical-post-processing-time
    metrics as single-objective. Points are 3-objective
    `(-execution_cost, statement_coverage, fault_coverage)` tuples -- cost
    negated so `piastq_execution.statistics`'s "maximize every objective"
    convention applies uniformly -- matching both SelectQAOA/
    MOQ-Pipeline.ipynb's HV/IGD/Pareto-dominance evaluation
    (`total_cost()`/`total_coverage()`/`total_faults()`/`pareto_dominance()`
    there) and `qaoa_tcs/multi_obj.py`'s `build_pareto_front()`, which uses
    the same three objectives to pick candidates during execution.

`execution_time_seconds` means the same thing, computed the same way, for
every one of the three pipelines this module serves (QAOA-TCS single-
objective, QAOA-TCS multi-objective, IGDec-QAOA single-objective): the sum,
over one experiment repetition, of the wall-clock time spent executing each
subproblem's QAOA circuit (one `RawCountsRecord` per subproblem/cluster).
It's distinct from `execution_cost` (single-objective only -- the selected
test suite's own dataset cost, not hardware time).

Everything here is algorithm-agnostic pure computation over already-collected
data (raw counts, calibration records, selected bitstrings) -- no AQTProvider/
AQTSampler/backend calls, no QAOA optimization loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from piastq_execution.mitigation import CalibrationRecord, correct_counts
from piastq_execution.raw_counts import RawCountsRecord
from piastq_execution.statistics import (
    build_reference_front,
    hypervolume,
    inverted_generational_distance,
    pareto_front_indices,
)

# Which effectiveness metric(s) apply to each single-objective dataset, and
# which column(s) of that dataset's per-test-case data they're summed from.
EFFECTIVENESS_METRICS: Dict[str, List[str]] = {
    "gsdtsr": ["failure_rate"],
    "iofrol": ["failure_rate"],
    "paintcontrol": ["failure_rate"],
    "elevator_o2": ["input_diversity"],
    "elevator_o3": ["passenger_count", "travel_distance"],
}


# ---------------------------------------------------------------------------
# QUBO energy / optimal-bitstring probability
# ---------------------------------------------------------------------------

def qubo_energy(bitstring: Sequence[int], linear: Sequence[float], quadratic: Dict[Tuple[int, int], float]) -> float:
    """Evaluates linear^T x + sum_{i,j} quadratic[i,j] * x_i * x_j for a 0/1
    bitstring, matching how create_linear_qubo()/create_QUBO_problem() encode
    the QUBOs in qaoa_tcs/single_obj.py and qaoa_tcs/multi_obj.py."""
    energy = 0.0
    for i, xi in enumerate(bitstring):
        energy += linear[i] * xi
    for (i, j), coeff in quadratic.items():
        energy += coeff * bitstring[i] * bitstring[j]
    return energy


def brute_force_optimal(
    linear: Sequence[float], quadratic: Dict[Tuple[int, int], float], num_qubits: int
) -> Tuple[str, float]:
    """Brute-forces the true optimal bitstring for a <=7-qubit QUBO subproblem
    (at most 128 states -- trivial). Returns (bitstring, energy), with the
    bitstring in the same little-endian convention measurement results use.
    """
    best_bitstring: Optional[str] = None
    best_energy: Optional[float] = None
    for state in range(2 ** num_qubits):
        label = format(state, f"0{num_qubits}b")
        bits = [int(b) for b in label[::-1]]
        energy = qubo_energy(bits, linear, quadratic)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_bitstring = label
    return best_bitstring, best_energy


def probability_of_optimal(distribution: Dict[str, float], optimal_bitstring: str) -> float:
    return distribution.get(optimal_bitstring, 0.0)


def select_argmax_bitstring(distribution: Dict[str, float]) -> str:
    """Same selection rule the execution tails already use for the raw
    baseline (argmax of the distribution), applied here to any corrected
    distribution so every method is evaluated the same way."""
    if not distribution:
        raise ValueError("Cannot select from an empty distribution")
    return max(distribution.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Selection merging (per-cluster bitstring -> final selected test suite)
# ---------------------------------------------------------------------------

@dataclass
class ClusterAssignment:
    cluster_id: Any
    cluster_test_cases: List[int]


def merge_selected_tests(
    cluster_assignments: Sequence[ClusterAssignment],
    bitstrings: Sequence[str],
) -> List[int]:
    """Merges one selected bitstring per cluster into the final (deduplicated,
    sorted) global test suite, matching the merge logic already used in
    qaoa_tcs/single_obj.py, qaoa_tcs/multi_obj.py, and the IGDec-QAOA scripts.
    """
    if len(cluster_assignments) != len(bitstrings):
        raise ValueError("cluster_assignments and bitstrings must be the same length")

    selected: List[int] = []
    for assignment, bitstring in zip(cluster_assignments, bitstrings):
        bits = [int(b) for b in bitstring[::-1]]
        for index, value in enumerate(bits):
            if value == 1 and index < len(assignment.cluster_test_cases):
                test_id = assignment.cluster_test_cases[index]
                if test_id not in selected:
                    selected.append(test_id)
    return sorted(selected)


# ---------------------------------------------------------------------------
# Single-objective effectiveness/cost metrics
# ---------------------------------------------------------------------------

def compute_single_objective_metrics(
    dataset: str,
    selected_tests: Sequence[int],
    test_case_data: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    """Sums execution cost and the dataset's relevant effectiveness metric(s)
    over the selected test suite.

    `test_case_data` maps column name ("cost", "failure_rate",
    "input_diversity", "passenger_count", "travel_distance") to a
    per-test-case value list, indexed the same way `selected_tests` is.
    """
    if dataset not in EFFECTIVENESS_METRICS:
        raise ValueError(f"Unknown dataset '{dataset}'")

    metrics = {"execution_cost": sum(test_case_data["cost"][i] for i in selected_tests)}
    for metric_name in EFFECTIVENESS_METRICS[dataset]:
        metrics[metric_name] = sum(test_case_data[metric_name][i] for i in selected_tests)
    return metrics


# ---------------------------------------------------------------------------
# Execution time (quantum-hardware wall-clock) and mitigation overhead
# ---------------------------------------------------------------------------

def compute_execution_time_seconds(raw_records: Sequence[RawCountsRecord]) -> float:
    """Sum, over one experiment repetition, of the wall-clock time spent
    executing each subproblem's QAOA circuit (one RawCountsRecord per
    subproblem/cluster). Applies identically to raw and every mitigated
    method -- for TREx, `raw_records` should include every twirl instance,
    so this naturally captures TREx's extra hardware passes too.
    """
    return sum(r.total_wall_clock_seconds for r in raw_records)


@dataclass
class MitigationOverhead:
    calibration_circuits: int
    total_shots: int
    calibration_wall_clock_seconds: float


def compute_mitigation_overhead(
    raw_records: Sequence[RawCountsRecord],
    calibration_record: Optional[CalibrationRecord] = None,
    calibration_shots_per_circuit: int = 0,
) -> MitigationOverhead:
    """Aggregates shot-based overhead: total shots spent on the raw circuit
    executions passed in, plus (if a calibration record is supplied) the
    extra calibration circuits/shots it cost, and the *measured* hardware
    time that calibration actually took
    (`calibration_record.calibration_wall_clock_seconds`, from
    run_calibration.py timing every calibration circuit's execution -- not
    an estimate). Zero for raw/TREx, since neither uses a calibration
    record. See compute_execution_time_seconds() for the QAOA-circuit
    wall-clock-time counterpart (present on every method, including raw).
    """
    total_shots = sum(r.total_shots_returned for r in raw_records)

    calibration_circuits = 0
    calibration_wall_clock_seconds = 0.0
    if calibration_record is not None:
        if calibration_record.method == "mem":
            calibration_circuits = 2 ** calibration_record.num_qubits
        elif calibration_record.method == "m3":
            calibration_circuits = 2 * calibration_record.num_qubits
        total_shots += calibration_circuits * calibration_shots_per_circuit
        calibration_wall_clock_seconds = calibration_record.calibration_wall_clock_seconds

    return MitigationOverhead(
        calibration_circuits=calibration_circuits,
        total_shots=total_shots,
        calibration_wall_clock_seconds=calibration_wall_clock_seconds,
    )


# ---------------------------------------------------------------------------
# Single-objective, per-combo x per-method evaluation
# ---------------------------------------------------------------------------

@dataclass
class SingleObjectiveEvaluation:
    combo: str
    method: str
    qubo_energy: float
    optimal_bitstring: str
    probability_of_optimal: float
    execution_cost: float
    effectiveness: Dict[str, float]
    execution_time_seconds: float
    mitigation_overhead: MitigationOverhead
    classical_post_processing_seconds: float


def evaluate_single_objective_cluster(
    method: str,
    raw_counts: Dict[str, int],
    linear: Sequence[float],
    quadratic: Dict[Tuple[int, int], float],
    num_qubits: int,
    calibration_record: Optional[CalibrationRecord] = None,
    physical_qubits: Optional[Sequence[int]] = None,
) -> Tuple[str, Dict[str, float], float]:
    """Corrects one cluster's raw counts under `method`, brute-forces that
    cluster's true optimum, and returns (selected_bitstring, distribution,
    probability_of_optimal). Classical post-processing time is measured by
    the caller around this function (per-cluster granularity would be noisy
    to time individually).
    """
    distribution = correct_counts(method, raw_counts, calibration_record, num_qubits, physical_qubits)
    optimal_bitstring, _ = brute_force_optimal(linear, quadratic, num_qubits)
    p_optimal = probability_of_optimal(distribution, optimal_bitstring)
    selected = select_argmax_bitstring(distribution)
    return selected, distribution, p_optimal


def evaluate_single_objective_combo(
    combo: str,
    method: str,
    dataset: str,
    cluster_assignments: Sequence[ClusterAssignment],
    cluster_raw_counts: Sequence[Dict[str, int]],
    cluster_qubos: Sequence[Tuple[Sequence[float], Dict[Tuple[int, int], float], int]],
    test_case_data: Dict[str, Sequence[float]],
    raw_records: Sequence[RawCountsRecord],
    calibration_record: Optional[CalibrationRecord] = None,
    calibration_shots_per_circuit: int = 0,
) -> SingleObjectiveEvaluation:
    """Full per-combo x per-method evaluation for one single-objective run:
    corrects every cluster's counts, merges into the final suite, and
    computes every metric in one pass.
    """
    start = time.time()

    selected_bitstrings = []
    total_qubo_energy = 0.0
    optimal_probabilities = []

    for (linear, quadratic, num_qubits), raw_counts in zip(cluster_qubos, cluster_raw_counts):
        selected, _distribution, p_optimal = evaluate_single_objective_cluster(
            method, raw_counts, linear, quadratic, num_qubits, calibration_record,
        )
        selected_bitstrings.append(selected)
        bits = [int(b) for b in selected[::-1]]
        total_qubo_energy += qubo_energy(bits, linear, quadratic)
        optimal_probabilities.append(p_optimal)

    final_selected_tests = merge_selected_tests(cluster_assignments, selected_bitstrings)
    effectiveness_metrics = compute_single_objective_metrics(dataset, final_selected_tests, test_case_data)
    execution_cost = effectiveness_metrics.pop("execution_cost")

    elapsed = time.time() - start

    mean_p_optimal = sum(optimal_probabilities) / len(optimal_probabilities) if optimal_probabilities else 0.0

    return SingleObjectiveEvaluation(
        combo=combo,
        method=method,
        qubo_energy=total_qubo_energy,
        optimal_bitstring="+".join(selected_bitstrings),
        probability_of_optimal=mean_p_optimal,
        execution_cost=execution_cost,
        effectiveness=effectiveness_metrics,
        execution_time_seconds=compute_execution_time_seconds(raw_records),
        mitigation_overhead=compute_mitigation_overhead(raw_records, calibration_record, calibration_shots_per_circuit),
        classical_post_processing_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Multi-objective evaluation
# ---------------------------------------------------------------------------

@dataclass
class MultiObjectiveEvaluation:
    combo: str
    method: str
    num_non_dominated: int
    hypervolume: float
    igd: float
    execution_time_seconds: float
    mitigation_overhead: MitigationOverhead
    classical_post_processing_seconds: float


def evaluate_multi_objective_combo(
    combo: str,
    method: str,
    pareto_points: Sequence[Tuple[float, ...]],
    all_methods_points: Dict[str, Sequence[Tuple[float, ...]]],
    reference_point: Tuple[float, ...],
    raw_records: Sequence[RawCountsRecord],
    calibration_record: Optional[CalibrationRecord] = None,
    calibration_shots_per_circuit: int = 0,
) -> MultiObjectiveEvaluation:
    """Evaluates one method's Pareto front against the a posteriori reference
    frontier built from the union of every compared method's non-dominated
    solutions for this dataset (`all_methods_points` must include `method`'s
    own points under its own key).

    `pareto_points`/`all_methods_points`/`reference_point` are 3-tuples of
    `(-execution_cost, statement_coverage, fault_coverage)` for QAOA-TCS
    multi-objective (cost negated -- see module docstring); `hypervolume()`
    and every other statistics.py function used here are dimension-agnostic,
    so this also works unchanged for a differently-shaped multi-objective
    front if one is ever added.
    """
    start = time.time()

    own_front_indices = pareto_front_indices(pareto_points)
    own_front = [pareto_points[i] for i in own_front_indices]

    reference_front = build_reference_front(all_methods_points)
    num_non_dominated = sum(1 for p in own_front if p in reference_front)

    hv = hypervolume(own_front, reference_point)
    igd = inverted_generational_distance(own_front, reference_front)

    elapsed = time.time() - start

    return MultiObjectiveEvaluation(
        combo=combo,
        method=method,
        num_non_dominated=num_non_dominated,
        hypervolume=hv,
        igd=igd,
        execution_time_seconds=compute_execution_time_seconds(raw_records),
        mitigation_overhead=compute_mitigation_overhead(raw_records, calibration_record, calibration_shots_per_circuit),
        classical_post_processing_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Objective-mode dispatch
# ---------------------------------------------------------------------------

def evaluate_combo(objective_mode: str, **kwargs):
    """Branches on objective_mode to the right evaluation function. Kwargs are
    passed straight through to evaluate_single_objective_combo() or
    evaluate_multi_objective_combo()."""
    if objective_mode == "single_objective":
        return evaluate_single_objective_combo(**kwargs)
    if objective_mode == "multi_objective":
        return evaluate_multi_objective_combo(**kwargs)
    raise ValueError(f"Unknown objective_mode '{objective_mode}'")


# ---------------------------------------------------------------------------
# Bridge to piastq_execution.statistics: turn a list of per-repetition
# evaluation results into the plain float samples compare_groups()/
# compare_two_groups() expect.
# ---------------------------------------------------------------------------

def extract_metric_samples(
    results: Sequence[Any],
    metric: str,
) -> List[float]:
    """Pulls one scalar metric out of a list of per-repetition
    SingleObjectiveEvaluation/MultiObjectiveEvaluation results (one result
    per repetition, e.g. everything evaluate_single_objective_combo() was
    called with a given combo+method across repetitions), ready to hand to
    piastq_execution.statistics.compare_groups()/compare_two_groups().

    `metric` supports dotted paths into nested fields, e.g.
    "mitigation_overhead.total_shots". Common choices: "qubo_energy",
    "probability_of_optimal", "execution_cost" (single-objective only),
    "execution_time_seconds" (both objective modes -- this is the one to use
    for comparing algorithms/methods on quantum-hardware execution cost),
    "hypervolume" / "igd" / "num_non_dominated" (multi-objective only),
    "classical_post_processing_seconds".
    """
    samples = []
    for result in results:
        value: Any = result
        for part in metric.split("."):
            value = getattr(value, part)
        samples.append(float(value))
    return samples
