"""Readout-error mitigation: full MEM, M3, and TREx.

Algorithm/dataset-agnostic by design: every function here operates on a raw
counts dict (bitstring -> count) plus a circuit width / physical-qubit list.
None of it knows or cares whether the circuit came from QAOA-TCS or
IGDec-QAOA, single- or multi-objective, or which dataset -- that keeps the
same mitigation code usable across all 14 combos without forking per
algorithm or dataset, per the study's requirements.

Nothing here calls AQTProvider/AQTSampler or any backend: circuit *builders*
(build_mem_calibration_circuits, build_m3_calibration_circuits,
build_trex_twirled_circuit) just return QuantumCircuit objects for the caller
to submit through whatever sampler it already has; the *correction* functions
work purely on already-collected counts dicts.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Full MEM (measurement error mitigation via the full 2**n confusion matrix)
# ---------------------------------------------------------------------------

def build_mem_calibration_circuits(num_qubits: int):
    """One circuit per computational basis state |b>, b in 0..2**num_qubits-1.

    Returns [(bitstring_label, QuantumCircuit), ...]. `bitstring_label` uses
    qiskit's little-endian convention (rightmost character = qubit 0), matching
    how measurement results are reported.
    """
    from qiskit import QuantumCircuit

    circuits = []
    for state in range(2 ** num_qubits):
        bitstring = format(state, f"0{num_qubits}b")
        qc = QuantumCircuit(num_qubits, num_qubits)
        for i, bit in enumerate(reversed(bitstring)):
            if bit == "1":
                qc.x(i)
        qc.measure(range(num_qubits), range(num_qubits))
        circuits.append((bitstring, qc))
    return circuits


def build_confusion_matrix(calibration_counts: Dict[str, Dict[str, int]], num_qubits: int) -> np.ndarray:
    """Builds the full 2**n x 2**n assignment/confusion matrix from calibration
    circuit results.

    Args:
        calibration_counts: {prepared_bitstring: {measured_bitstring: count, ...}, ...},
            one entry per circuit from build_mem_calibration_circuits().
        num_qubits: circuit width.

    Returns:
        A of shape (2**n, 2**n) with A[measured_idx, prepared_idx] = P(measured | prepared).
    """
    dim = 2 ** num_qubits
    matrix = np.zeros((dim, dim))
    for prepared_bitstring, counts in calibration_counts.items():
        prepared_idx = int(prepared_bitstring, 2)
        total = sum(counts.values())
        if total == 0:
            continue
        for measured_bitstring, count in counts.items():
            measured_idx = int(measured_bitstring, 2)
            matrix[measured_idx, prepared_idx] = count / total
    return matrix


def correct_counts_with_matrix(
    raw_counts: Dict[str, int],
    confusion_matrix: np.ndarray,
    num_qubits: int,
    method: str = "nnls",
) -> Dict[str, float]:
    """Inverts `confusion_matrix` against the noisy distribution derived from
    `raw_counts`, returning a corrected probability distribution that is
    guaranteed non-negative and normalized to 1.

    method:
      "nnls" (default) -- non-negative least squares (scipy.optimize.nnls).
          Numerically stable and enforces non-negativity directly; the
          preferred approach.
      "pinv" -- Moore-Penrose pseudo-inverse, with negative entries clipped to
          0 and the result renormalized. Documented fallback for when NNLS
          fails to converge or for cross-checking.
    """
    dim = 2 ** num_qubits
    total_shots = sum(raw_counts.values())
    if total_shots == 0:
        raise ValueError("raw_counts is empty (zero total shots)")

    p_noisy = np.zeros(dim)
    for bitstring, count in raw_counts.items():
        p_noisy[int(bitstring, 2)] = count / total_shots

    if method == "nnls":
        from scipy.optimize import nnls

        x, _residual = nnls(confusion_matrix, p_noisy)
    elif method == "pinv":
        x = np.linalg.pinv(confusion_matrix) @ p_noisy
        x = np.clip(x, 0, None)
    else:
        raise ValueError(f"Unknown correction method '{method}'")

    total = x.sum()
    if total <= 0:
        # Degenerate calibration (e.g. singular matrix) -- fall back to the
        # raw (uncorrected) distribution rather than dividing by zero.
        x = p_noisy
        total = x.sum()
        if total <= 0:
            x = np.ones(dim) / dim
            total = 1.0
    x = x / total

    return {format(i, f"0{num_qubits}b"): float(p) for i, p in enumerate(x) if p > 0}


# ---------------------------------------------------------------------------
# M3 (per-qubit independent/marginal calibration, corrected via mthree when
# available, with a reduced-support NNLS fallback otherwise -- see
# _correct_counts_m3_reduced())
# ---------------------------------------------------------------------------

def build_m3_calibration_circuits(num_qubits: int):
    """Two circuits per qubit (prepare |0>, prepare |1> on that qubit alone,
    all others held at |0>), measuring every qubit each time so the target
    qubit's marginal readout confusion can be extracted from the full-width
    counts. Returns [(qubit_index, prepared_bit, QuantumCircuit), ...].
    """
    from qiskit import QuantumCircuit

    circuits = []
    for q in range(num_qubits):
        for prepared_bit in (0, 1):
            qc = QuantumCircuit(num_qubits, num_qubits)
            if prepared_bit == 1:
                qc.x(q)
            qc.measure(range(num_qubits), range(num_qubits))
            circuits.append((q, prepared_bit, qc))
    return circuits


def build_m3_single_qubit_cals(
    calibration_counts: Dict[Tuple[int, int], Dict[str, int]], num_qubits: int
) -> List[np.ndarray]:
    """Builds one 2x2 per-qubit calibration matrix per qubit from
    build_m3_calibration_circuits() results.

    Args:
        calibration_counts: {(qubit_index, prepared_bit): {measured_bitstring: count}, ...}
        num_qubits: circuit width.

    Returns:
        List of length num_qubits; cals[q][measured_bit, prepared_bit] = P(measured_bit on q | prepared_bit on q).
    """
    cals = []
    for q in range(num_qubits):
        mat = np.zeros((2, 2))
        for prepared_bit in (0, 1):
            counts = calibration_counts[(q, prepared_bit)]
            total = sum(counts.values())
            if total == 0:
                mat[prepared_bit, prepared_bit] = 1.0  # no data: assume ideal readout
                continue
            for bitstring, count in counts.items():
                measured_bit = int(bitstring[::-1][q])
                mat[measured_bit, prepared_bit] += count / total
        cals.append(mat)
    return cals


def _correct_counts_m3_reduced(
    raw_counts: Dict[str, int],
    single_qubit_cals: Sequence[np.ndarray],
    num_qubits: int,
    method: str = "nnls",
) -> Dict[str, float]:
    """Fallback used by correct_counts_m3() when the `mthree` package isn't
    installed.

    Restricts the assignment matrix to just the bitstrings actually observed
    in `raw_counts` -- typically far fewer than 2**num_qubits for a QAOA
    output distribution concentrated on a handful of low-energy states --
    computing each entry as a product of per-qubit calibration probabilities,
    instead of materializing the full 2**num_qubits x 2**num_qubits matrix
    the way MEM does. This mirrors the real M3 algorithm's approach (Nation
    et al. 2021, "Scalable Mitigation of Measurement Errors on Quantum
    Computers"): restrict correction to the noisy distribution's support
    rather than every possible bitstring, which is what keeps M3
    sub-exponential in practice. It's still a simplification of the actual
    `mthree` implementation (which also expands the support to nearby
    bitstrings reachable by low-order bit flips, and uses an iterative
    solver rather than dense NNLS) -- installing `mthree` (already in
    requirements.txt) is preferred over relying on this path.
    """
    bitstrings = sorted(raw_counts.keys())
    n = len(bitstrings)
    total_shots = sum(raw_counts.values())
    if total_shots == 0:
        raise ValueError("raw_counts is empty (zero total shots)")

    reduced_matrix = np.zeros((n, n))
    for i, measured in enumerate(bitstrings):
        for j, prepared in enumerate(bitstrings):
            prob = 1.0
            for q in range(num_qubits):
                measured_bit = int(measured[::-1][q])
                prepared_bit = int(prepared[::-1][q])
                prob *= single_qubit_cals[q][measured_bit, prepared_bit]
            reduced_matrix[i, j] = prob

    p_noisy = np.array([raw_counts[b] / total_shots for b in bitstrings])

    if method == "nnls":
        from scipy.optimize import nnls

        x, _residual = nnls(reduced_matrix, p_noisy)
    elif method == "pinv":
        x = np.linalg.pinv(reduced_matrix) @ p_noisy
        x = np.clip(x, 0, None)
    else:
        raise ValueError(f"Unknown correction method '{method}'")

    total = x.sum()
    if total <= 0:
        x = p_noisy
        total = x.sum()
        if total <= 0:
            x = np.ones(n) / n
            total = 1.0
    x = x / total

    return {b: float(p) for b, p in zip(bitstrings, x) if p > 0}


def correct_counts_m3(
    raw_counts: Dict[str, int],
    single_qubit_cals: Sequence[np.ndarray],
    physical_qubits: Optional[Sequence[int]] = None,
    method: str = "nnls",
) -> Dict[str, float]:
    """Corrects `raw_counts` using per-qubit M3 calibration matrices.

    Tries the `mthree` package first (M3Mitigation(system=None) +
    cals_from_matrices(), confirmed to work without any live backend object);
    falls back to _correct_counts_m3_reduced() -- a reduced-support
    approximation that, like real M3, never materializes the full
    2**num_qubits x 2**num_qubits matrix -- if mthree is not installed.
    """
    num_qubits = len(single_qubit_cals)
    if physical_qubits is None:
        physical_qubits = list(range(num_qubits))

    try:
        import mthree

        mit = mthree.M3Mitigation(system=None)
        mit.cals_from_matrices(list(single_qubit_cals))
        quasi = mit.apply_correction(dict(raw_counts), qubits=list(physical_qubits))
        return dict(quasi)
    except ImportError:
        return _correct_counts_m3_reduced(raw_counts, single_qubit_cals, num_qubits, method=method)


# ---------------------------------------------------------------------------
# TREx (measurement twirling)
# ---------------------------------------------------------------------------
#
# Minimum viable version: apply a random X gate to each qubit immediately
# before measurement, then classically XOR-undo that known mask on the
# resulting bitstring labels. Repeated over several independently-random
# twirl instances and aggregated, this converts coherent/biased readout
# error into unbiased stochastic noise without needing any calibration data.
#
# Full Pauli twirling of the *circuit* (inserting randomized Pauli
# conjugations around every entangling gate, not just before measurement) is
# a materially larger undertaking -- it requires a per-gate twirling set to
# preserve the ideal unitary -- and is not implemented here; only the
# measurement-twirling MVP described in the task is.

# Default number of independent twirl instances (each a distinct hardware
# execution) averaged together per circuit. 32 matches the commonly used
# default for measurement-twirling randomizations in Qiskit Runtime's
# twirled-readout implementations -- a reasonable balance between averaging
# out readout bias and added hardware time. Configurable per study via
# `trex_twirl_instances` in configs/execution_plan.yaml (see
# load_trex_twirl_instances()) rather than hardcoded at every call site.
DEFAULT_TREX_TWIRL_INSTANCES = 32


def load_trex_twirl_instances(config_path: Optional[str] = None) -> int:
    """Reads `trex_twirl_instances` from configs/execution_plan.yaml -- the
    single place this is configured, shared by budget_planner.py's cost
    model and every execution script that actually runs the TREx
    twirl-instance loop, so they can never disagree about how many twirl
    instances a TREx pass costs vs. how many it actually runs.

    Falls back to DEFAULT_TREX_TWIRL_INSTANCES if the file or key is absent.
    """
    import yaml

    if config_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, "..", "..", "configs", "execution_plan.yaml")

    if not os.path.exists(config_path):
        return DEFAULT_TREX_TWIRL_INSTANCES

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return int(raw.get("trex_twirl_instances", DEFAULT_TREX_TWIRL_INSTANCES))


def generate_random_twirl_mask(num_qubits: int, rng: Optional[random.Random] = None) -> List[int]:
    rng = rng or random
    return [rng.randint(0, 1) for _ in range(num_qubits)]


def build_trex_twirled_circuit(base_circuit, twirl_mask: Sequence[int]):
    """Returns a copy of `base_circuit` with its trailing measurement/barrier
    operations stripped and replaced by: X gates on qubits where
    `twirl_mask[q] == 1`, then a fresh full-width measurement.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(base_circuit.num_qubits, base_circuit.num_clbits)
    for instr in base_circuit.data:
        if instr.operation.name in ("measure", "barrier"):
            continue
        qubit_indices = [base_circuit.find_bit(q).index for q in instr.qubits]
        clbit_indices = [base_circuit.find_bit(c).index for c in instr.clbits]
        qc.append(instr.operation, [qc.qubits[i] for i in qubit_indices], [qc.clbits[i] for i in clbit_indices])

    for q, flip in enumerate(twirl_mask):
        if flip:
            qc.x(q)

    qc.measure(range(base_circuit.num_qubits), range(base_circuit.num_clbits))
    return qc


def undo_trex_twirl(raw_counts: Dict[str, int], twirl_mask: Sequence[int]) -> Dict[str, int]:
    """XOR-corrects every measured bitstring in `raw_counts` by the known
    `twirl_mask` used to build the circuit that produced them."""
    num_qubits = len(twirl_mask)
    corrected: Counter = Counter()
    for bitstring, count in raw_counts.items():
        bits = list(bitstring[::-1])  # index 0 = qubit 0
        for q in range(num_qubits):
            if twirl_mask[q]:
                bits[q] = "1" if bits[q] == "0" else "0"
        corrected_bitstring = "".join(reversed(bits))
        corrected[corrected_bitstring] += count
    return dict(corrected)


def aggregate_trex_instances(instances: Sequence[Tuple[Sequence[int], Dict[str, int]]]) -> Dict[str, int]:
    """Combines several (twirl_mask, raw_counts) instances -- each produced by
    a fresh circuit execution, per instance, distinct from the raw baseline --
    into one aggregated, twirl-undone counts dict."""
    total: Counter = Counter()
    for twirl_mask, raw_counts in instances:
        for bitstring, count in undo_trex_twirl(raw_counts, twirl_mask).items():
            total[bitstring] += count
    return dict(total)


def aggregate_trex_records(records: Sequence[Any]) -> Dict[Tuple[Any, Any, Any], Dict[str, int]]:
    """Groups TREx raw-counts records (each a RawCountsRecord with its own
    `twirl_mask` set, as written to `*-trex-raw_counts.jsonl` by the
    execution scripts) by `(cluster_id, iteration_id, subproblem_id)` and
    aggregates each group's twirl instances into one twirl-undone counts dict
    per circuit -- the shape
    `piastq_execution.evaluation.evaluate_single_objective_combo(method="trex",
    cluster_raw_counts=..., ...)` expects.

    The 3-part key (rather than just cluster/iteration) is what keeps
    QAOA-TCS (identifies a circuit by cluster_id + iteration_id,
    subproblem_id always None) and IGDec-QAOA (identifies a circuit by
    iteration_id + subproblem_id, cluster_id always None) from colliding:
    both algorithms leave the field they don't use at its default None, so
    grouping on all three together is unambiguous for either.

    `records` items only need `.twirl_mask`, `.aggregated_counts`,
    `.cluster_id`, `.iteration_id`, and `.subproblem_id` attributes
    (duck-typed, so this works directly on RawCountsRecord instances without
    importing that class here).
    """
    by_key: Dict[Tuple[Any, Any, Any], List[Tuple[Sequence[int], Dict[str, int]]]] = defaultdict(list)
    for record in records:
        if record.twirl_mask is None:
            raise ValueError(f"Raw-counts record {record.circuit_id!r} has no twirl_mask -- not a TREx record")
        key = (record.cluster_id, record.iteration_id, record.subproblem_id)
        by_key[key].append((record.twirl_mask, record.aggregated_counts))
    return {key: aggregate_trex_instances(instances) for key, instances in by_key.items()}


# ---------------------------------------------------------------------------
# Calibration storage (shared by MEM and M3)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    method: str  # "mem" | "m3"
    num_qubits: int
    physical_qubits: List[int]
    backend_name: str
    backend_version: str
    calibration_timestamp: str
    shots_per_calibration_circuit: int
    calibration_wall_clock_seconds: float  # measured hardware time for every calibration circuit combined
    data: dict  # method-specific: MEM -> {"confusion_matrix": [[...]]}; M3 -> {"single_qubit_cals": [[[...]], ...]}

    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict) -> "CalibrationRecord":
        return cls(**d)

    def confusion_matrix(self) -> np.ndarray:
        return np.array(self.data["confusion_matrix"])

    def single_qubit_cals(self) -> List[np.ndarray]:
        return [np.array(m) for m in self.data["single_qubit_cals"]]


def make_mem_calibration_record(
    confusion_matrix: np.ndarray,
    physical_qubits: Sequence[int],
    backend_name: str,
    backend_version: str,
    shots_per_calibration_circuit: int,
    calibration_wall_clock_seconds: float,
) -> CalibrationRecord:
    return CalibrationRecord(
        method="mem",
        num_qubits=len(physical_qubits),
        physical_qubits=list(physical_qubits),
        backend_name=backend_name,
        backend_version=backend_version,
        calibration_timestamp=datetime.now(timezone.utc).isoformat(),
        shots_per_calibration_circuit=shots_per_calibration_circuit,
        calibration_wall_clock_seconds=calibration_wall_clock_seconds,
        data={"confusion_matrix": confusion_matrix.tolist()},
    )


def make_m3_calibration_record(
    single_qubit_cals: Sequence[np.ndarray],
    physical_qubits: Sequence[int],
    backend_name: str,
    backend_version: str,
    shots_per_calibration_circuit: int,
    calibration_wall_clock_seconds: float,
) -> CalibrationRecord:
    return CalibrationRecord(
        method="m3",
        num_qubits=len(physical_qubits),
        physical_qubits=list(physical_qubits),
        backend_name=backend_name,
        backend_version=backend_version,
        calibration_timestamp=datetime.now(timezone.utc).isoformat(),
        shots_per_calibration_circuit=shots_per_calibration_circuit,
        calibration_wall_clock_seconds=calibration_wall_clock_seconds,
        data={"single_qubit_cals": [m.tolist() for m in single_qubit_cals]},
    )


class CalibrationStore:
    """Persists calibration data separately from any QAOA raw-counts file, so
    it can be reapplied to any raw-counts file sharing the same (method,
    physical qubit mapping) without rerunning hardware.
    """

    def __init__(self, base_dir: Optional[str] = None):
        # Default resolves to <repo_root>/calibration regardless of the
        # caller's current working directory (this file lives at
        # <repo_root>/src/piastq_execution/mitigation.py).
        if base_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(here, "..", "..", "calibration")
        self.base_dir = base_dir

    def _path(self, method: str, physical_qubits: Sequence[int]) -> str:
        signature = "-".join(str(q) for q in sorted(physical_qubits))
        return os.path.join(self.base_dir, f"{method}_q{signature}.json")

    def save(self, record: CalibrationRecord) -> str:
        os.makedirs(self.base_dir, exist_ok=True)
        path = self._path(record.method, record.physical_qubits)
        with open(path, "w") as f:
            json.dump(record.to_json_dict(), f, indent=2)
        return path

    def load(self, method: str, physical_qubits: Sequence[int]) -> Optional[CalibrationRecord]:
        path = self._path(method, physical_qubits)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return CalibrationRecord.from_json_dict(json.load(f))


# ---------------------------------------------------------------------------
# Unified dispatch (used by the evaluation module)
# ---------------------------------------------------------------------------

def correct_counts(
    method: str,
    raw_counts: Dict[str, int],
    calibration_record: Optional[CalibrationRecord],
    num_qubits: int,
    physical_qubits: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Single entry point used identically regardless of algorithm/dataset:
    `method` is "raw", "mem", "m3", or "trex".

    For "trex", `raw_counts` must already be twirl-undone and aggregated
    across every twirl instance (via aggregate_trex_records() /
    aggregate_trex_instances()) -- undoing the twirl needs each instance's
    mask, which this function's plain counts-dict signature has no room for,
    so that step happens before calling this. Once aggregated, a TREx counts
    dict is normalized exactly like the raw baseline.
    """
    if method in ("raw", "trex"):
        total = sum(raw_counts.values())
        return {k: v / total for k, v in raw_counts.items()} if total else {}
    if method == "mem":
        if calibration_record is None:
            raise ValueError("MEM correction requested but no calibration record was supplied")
        return correct_counts_with_matrix(raw_counts, calibration_record.confusion_matrix(), num_qubits)
    if method == "m3":
        if calibration_record is None:
            raise ValueError("M3 correction requested but no calibration record was supplied")
        return correct_counts_m3(raw_counts, calibration_record.single_qubit_cals(), physical_qubits)
    raise ValueError(f"Unknown mitigation method '{method}'")
