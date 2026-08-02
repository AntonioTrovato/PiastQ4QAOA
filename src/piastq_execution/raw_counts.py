"""Shared raw-measurement persistence for every PIAST-Q execution tail.

qiskit primitives samplers (including AQTSampler) hand back a quasi-distribution
of *probabilities*, not the underlying integer shot counts. Every execution tail
in this repo (qaoa_tcs/single_obj.py, qaoa_tcs/multi_obj.py,
igdec_qaoa/loch_qaoa_tcm_extract_circuits.py and the elevator_o2/elevator_o3
IGDec-QAOA scripts) used to reduce that quasi-distribution down to a single
argmax bitstring and throw the rest away. That makes any post-hoc mitigation
(MEM/M3/TREx) or re-analysis impossible, since the full outcome distribution is
gone.

`run_circuit_with_batching_recorded` is the single, algorithm-agnostic
replacement for each script's local `run_circuit_with_batching`: it performs the
same shot-batched execution, but returns both the aggregated Counter (so the
existing argmax-based selection logic keeps working unchanged) and a
`RawCountsRecord` with everything needed to reproduce or re-mitigate the result
later without touching hardware again.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class BatchRecord:
    batch_index: int
    shots_requested: int
    shots_returned: int
    raw_counts: Dict[str, int]
    timestamp: str
    wall_clock_seconds: float


@dataclass
class RawCountsRecord:
    algorithm: str  # "qaoa_tcs" | "igdec_qaoa"
    objective_mode: str  # "single_objective" | "multi_objective"
    dataset: str
    circuit_id: str
    cluster_id: Optional[Any]
    iteration_id: Optional[Any]
    subproblem_id: Optional[Any]
    backend_name: str
    backend_version: str
    physical_qubit_mapping: Dict[int, int]
    batches: List[BatchRecord]
    total_shots_requested: int
    total_shots_returned: int
    total_wall_clock_seconds: float
    aggregated_counts: Dict[str, int]
    twirl_mask: Optional[List[int]] = None  # set only for TREx twirl-instance executions

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_backend_identity(backend) -> Dict[str, str]:
    """Best-effort extraction of a backend's name/version across provider APIs."""
    name = getattr(backend, "name", None)
    if callable(name):
        try:
            name = name()
        except TypeError:
            name = None
    version = getattr(backend, "backend_version", None) or getattr(backend, "version", None)
    if callable(version):
        try:
            version = version()
        except TypeError:
            version = None
    return {
        "backend_name": str(name) if name is not None else backend.__class__.__name__,
        "backend_version": str(version) if version is not None else "unknown",
    }


def get_physical_qubit_mapping(circuit, backend, optimization_level: int = 3) -> Dict[int, int]:
    """Return {logical_qubit_index: physical_qubit_index} for `circuit` on `backend`.

    Transpiles `circuit` for `backend` at the given optimization level (matching
    the optimization_level every script already passes to
    `sampler.set_transpile_options`) purely to recover the layout for
    record-keeping; the circuit actually submitted through the sampler is
    untouched by this call.
    """
    from qiskit import transpile

    transpiled = transpile(circuit, backend=backend, optimization_level=optimization_level)
    layout = getattr(transpiled, "layout", None)

    if layout is None or layout.initial_layout is None:
        return {i: i for i in range(circuit.num_qubits)}

    mapping: Dict[int, int] = {}
    for qubit, physical_index in layout.initial_layout.get_virtual_bits().items():
        try:
            logical_index = circuit.find_bit(qubit).index
        except Exception:
            continue
        mapping[logical_index] = physical_index

    if not mapping:
        return {i: i for i in range(circuit.num_qubits)}
    return mapping


def counts_from_quasi_probabilities(probabilities: Dict[str, float], shots: int) -> Dict[str, int]:
    """Reconstruct integer per-batch shot counts from a quasi-distribution.

    Each batch is run with a fixed shot count, so its probabilities are exact
    fractions count/shots; multiplying back by shots and rounding recovers the
    original integer counts.
    """
    return {bitstring: int(round(prob * shots)) for bitstring, prob in probabilities.items()}


def _run_one_batch(circuit, sampler, shots: int, batch_index: int) -> BatchRecord:
    sampler.options.shots = shots
    start = time.time()
    result = sampler.run([circuit]).result()
    elapsed = time.time() - start

    probabilities = result.quasi_dists[0].binary_probabilities()
    raw_counts = counts_from_quasi_probabilities(probabilities, shots)

    return BatchRecord(
        batch_index=batch_index,
        shots_requested=shots,
        shots_returned=sum(raw_counts.values()),
        raw_counts=raw_counts,
        timestamp=datetime.now(timezone.utc).isoformat(),
        wall_clock_seconds=elapsed,
    )


def run_circuit_with_batching_recorded(
    circuit,
    sampler,
    *,
    algorithm: str,
    objective_mode: str,
    dataset: str,
    circuit_id: str,
    backend,
    cluster_id: Optional[Any] = None,
    iteration_id: Optional[Any] = None,
    subproblem_id: Optional[Any] = None,
    shots_per_batch: int = 80,
    num_batches: int = 1,
    remainder_shots: int = 0,
    optimization_level: int = 3,
    record_qubit_mapping: bool = True,
    twirl_mask: Optional[List[int]] = None,
):
    """Shot-batched circuit execution with full raw-data capture.

    Runs `num_batches` batches of `shots_per_batch` shots each, plus one final
    batch of `remainder_shots` shots if given (mirrors the "N x 200 + remainder"
    pattern used for the full hardware shot budget). Returns
    `(aggregated_counts, record)`:
      - `aggregated_counts`: a Counter[bitstring] -> int, summed across all
        batches, for drop-in use by existing argmax-based selection code.
      - `record`: a RawCountsRecord with every batch's raw counts plus metadata,
        for the raw-counts persistence file.

    `twirl_mask`: pass the mask used to build `circuit` (via
    piastq_execution.mitigation.build_trex_twirled_circuit) when this call is
    one TREx twirl instance, so the mask travels with its counts in the
    persisted record and can be undone later with
    piastq_execution.mitigation.aggregate_trex_records(). Leave as None for
    every non-TREx (raw/MEM/M3) execution.
    """
    batch_records: List[BatchRecord] = []

    for batch_index in range(num_batches):
        batch_records.append(_run_one_batch(circuit, sampler, shots_per_batch, batch_index))

    if remainder_shots > 0:
        batch_records.append(_run_one_batch(circuit, sampler, remainder_shots, num_batches))

    total_counts: Counter = Counter()
    for batch in batch_records:
        for bitstring, count in batch.raw_counts.items():
            total_counts[bitstring] += count

    backend_identity = get_backend_identity(backend)
    qubit_mapping = (
        get_physical_qubit_mapping(circuit, backend, optimization_level=optimization_level)
        if record_qubit_mapping
        else {i: i for i in range(circuit.num_qubits)}
    )

    record = RawCountsRecord(
        algorithm=algorithm,
        objective_mode=objective_mode,
        dataset=dataset,
        circuit_id=circuit_id,
        cluster_id=cluster_id,
        iteration_id=iteration_id,
        subproblem_id=subproblem_id,
        backend_name=backend_identity["backend_name"],
        backend_version=backend_identity["backend_version"],
        physical_qubit_mapping=qubit_mapping,
        batches=batch_records,
        total_shots_requested=sum(b.shots_requested for b in batch_records),
        total_shots_returned=sum(b.shots_returned for b in batch_records),
        total_wall_clock_seconds=sum(b.wall_clock_seconds for b in batch_records),
        aggregated_counts=dict(total_counts),
        twirl_mask=list(twirl_mask) if twirl_mask is not None else None,
    )

    return total_counts, record


class RawCountsWriter:
    """Appends one JSON record per line (JSONL) so a crashed/interrupted hardware
    run keeps every circuit executed so far, and records can be streamed rather
    than held in memory for the whole run.
    """

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "a")

    def write(self, record: RawCountsRecord) -> None:
        self._fh.write(json.dumps(record.to_json_dict()) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RawCountsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
