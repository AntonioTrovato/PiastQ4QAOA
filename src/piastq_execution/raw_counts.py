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
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BatchRecord:
    batch_index: int
    shots_requested: int
    shots_returned: int
    raw_counts: Dict[str, int]
    timestamp: str
    wall_clock_seconds: float
    # Best-effort job metadata from the provider (None if it exposes none):
    # the job id, plus whatever the primitive result's `metadata` holds
    # (e.g. queue/execution timestamps, if the backend reports them).
    job_id: Optional[str] = None
    job_metadata: Optional[Dict[str, Any]] = None
    # QPU seconds billed for this batch (modeled from the measured PIAST-Q
    # billing, see qpu_usage.modeled_billed_seconds); None on simulators.
    billed_seconds: Optional[float] = None


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
    # Depth/size/gate counts of the circuit after transpilation for the
    # backend (see transpiled_circuit_stats) -- None if it couldn't be computed.
    transpiled_stats: Optional[Dict[str, Any]] = None
    # Sum of the batches' billed QPU seconds; None on simulators. This, not
    # the wall-clock (which includes minutes of queue wait per job), is the
    # execution time used by the evaluation on the real machine.
    total_billed_seconds: Optional[float] = None

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


def physical_qubit_mapping_from_layout(circuit, initial_layout: List[int]) -> Dict[int, int]:
    """Return {logical_qubit_index: physical_qubit_index} directly from a
    pinned `initial_layout` (see piastq_execution.qubit_layout), with no
    transpile call needed: since the layout was pinned before submission (see
    `run_circuit_with_batching_recorded`), it's already known by construction,
    not something to reverse-engineer afterwards.
    """
    if len(initial_layout) != circuit.num_qubits:
        raise ValueError(
            f"initial_layout has {len(initial_layout)} qubits but circuit has "
            f"{circuit.num_qubits} -- they must match so calibration and "
            f"execution provably share the same physical qubits."
        )
    return {i: int(initial_layout[i]) for i in range(circuit.num_qubits)}


def get_physical_qubit_mapping(circuit, backend, optimization_level: int = 3) -> Dict[int, int]:
    """Return {logical_qubit_index: physical_qubit_index} for `circuit` on `backend`.

    Fallback used only when no `initial_layout` was pinned for this
    execution (see `run_circuit_with_batching_recorded`): transpiles
    `circuit` for `backend` at the given optimization level purely to
    *guess* the layout the transpiler would pick, for record-keeping. This is
    an independent, best-effort transpile call, separate from whatever
    transpilation the sampler itself performs for the actual submitted
    circuit -- it is not guaranteed to agree with it. Prefer pinning
    `initial_layout` (piastq_execution.qubit_layout.physical_layout_for_width)
    and using `physical_qubit_mapping_from_layout` instead, which has no such
    gap.
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


def _json_safe(value: Any) -> Any:
    """Round-trips `value` through JSON (str() for anything unserializable)
    so provider-specific metadata can never break a raw-counts write."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return None


def _job_info(job, result) -> Dict[str, Any]:
    """Best-effort job id + result metadata; never raises."""
    job_id = None
    try:
        job_id = str(job.job_id())
    except Exception:
        pass
    metadata = None
    raw_metadata = getattr(result, "metadata", None)
    if raw_metadata:
        metadata = _json_safe(raw_metadata)
    return {"job_id": job_id, "job_metadata": metadata}


def transpiled_circuit_stats(
    circuit, backend, optimization_level: int = 3, initial_layout: Optional[List[int]] = None
) -> Optional[Dict[str, Any]]:
    """Depth, size, two-qubit-gate count and gate histogram of `circuit`
    transpiled for `backend` with the same optimization level / pinned layout
    the execution uses. Like `get_physical_qubit_mapping`, this is an
    independent transpile call, not the sampler's own -- it describes what
    the transpiler produces for these settings, and is not guaranteed to be
    gate-for-gate what the sampler submitted. Returns None on any failure
    (record-keeping must never break an execution).
    """
    try:
        from qiskit import transpile

        kwargs = {"backend": backend, "optimization_level": optimization_level}
        if initial_layout is not None:
            kwargs["initial_layout"] = list(initial_layout)
        transpiled = transpile(circuit, **kwargs)
        return {
            "depth": int(transpiled.depth()),
            "size": int(transpiled.size()),
            "num_nonlocal_gates": int(transpiled.num_nonlocal_gates()),
            "count_ops": {str(k): int(v) for k, v in transpiled.count_ops().items()},
        }
    except Exception:
        return None


def resume_enabled() -> bool:
    """True when the run was started with `python run_all.py --resume`
    (PIASTQ_RESUME=1): circuits already saved in the raw-counts files are
    reused instead of being sent to the QPU again."""
    return os.environ.get("PIASTQ_RESUME") == "1"


def load_completed_records(path: str) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """{(circuit_id, iteration_id): record} for every circuit already written
    to the raw-counts JSONL at `path`. Empty unless resume is enabled and the
    file exists.

    A crash mid-write leaves a half-written last line. It is dropped, and the
    file is rewritten with only the complete lines (newline-terminated) BEFORE
    the writer reopens it in append mode -- otherwise the next record would be
    glued onto the fragment and corrupt both."""
    completed: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    if not resume_enabled() or not os.path.exists(path):
        return completed

    valid_lines: List[str] = []
    dirty = False
    with open(path) as f:
        text = f.read()
    if text and not text.endswith("\n"):
        dirty = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            key = (record["circuit_id"], record.get("iteration_id"))
        except (json.JSONDecodeError, KeyError):
            dirty = True
            continue
        completed[key] = record
        valid_lines.append(line)

    if dirty:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write("".join(line + "\n" for line in valid_lines))
        os.replace(tmp_path, path)
    return completed


MAX_JOB_ATTEMPTS = 3


def _submit_and_wait(sampler, circuit):
    """sampler.run(...).result(), retried when the service ends a job as
    `failed`/`canceled` (seen once on PIAST-Q; a failed job is billed 0 s).
    Any other error is not retried."""
    last_error = None
    for attempt in range(1, MAX_JOB_ATTEMPTS + 1):
        try:
            job = sampler.run([circuit])
            return job, job.result()
        except RuntimeError as exc:
            if "status:" not in str(exc):
                raise
            last_error = exc
            print(f"  [retry] {exc} (attempt {attempt}/{MAX_JOB_ATTEMPTS})")
            time.sleep(5)
    raise last_error


def _run_one_batch(circuit, sampler, shots: int, batch_index: int) -> BatchRecord:
    from piastq_execution.qpu_usage import check_budget, modeled_billed_seconds

    check_budget()
    sampler.options.shots = shots
    start = time.time()
    job, result = _submit_and_wait(sampler, circuit)
    elapsed = time.time() - start
    job_info = _job_info(job, result)

    probabilities = result.quasi_dists[0].binary_probabilities()
    raw_counts = counts_from_quasi_probabilities(probabilities, shots)

    return BatchRecord(
        batch_index=batch_index,
        shots_requested=shots,
        shots_returned=sum(raw_counts.values()),
        raw_counts=raw_counts,
        timestamp=datetime.now(timezone.utc).isoformat(),
        wall_clock_seconds=elapsed,
        job_id=job_info["job_id"],
        job_metadata=job_info["job_metadata"],
        billed_seconds=modeled_billed_seconds(shots),
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
    initial_layout: Optional[List[int]] = None,
    completed: Optional[Dict[Tuple[str, Any], Dict[str, Any]]] = None,
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

    `initial_layout`: the pinned physical qubits this circuit must run on
    (see piastq_execution.qubit_layout.physical_layout_for_width) -- pass this
    on every call so the *same* physical qubits are used for the real QAOA/
    TREx circuit and for the MEM/M3 calibration circuits of the same width
    (run_calibration.py). When given, it is applied to `sampler` via
    `set_transpile_options` before this circuit is submitted, and the
    recorded `physical_qubit_mapping` is derived directly from it (no guess
    -- see `physical_qubit_mapping_from_layout`). When left as None, the
    transpiler picks a layout automatically and the recorded mapping is only
    a best-effort guess (`get_physical_qubit_mapping`) -- kept for backward
    compatibility, not recommended for new call sites.

    `completed`: the dict from `load_completed_records(path)`. If this
    (circuit_id, iteration_id) is already in it, nothing is sent to the QPU:
    the saved counts are returned with `record=None` (the caller must then NOT
    write a record -- it is already in the file). Used by `--resume`.
    """
    if completed:
        prior = completed.get((circuit_id, iteration_id))
        if prior is not None:
            print(f"  [resume] {circuit_id} (iteration {iteration_id}) already executed, reusing saved counts")
            return Counter(prior["aggregated_counts"]), None

    if initial_layout is not None:
        sampler.set_transpile_options(optimization_level=optimization_level, initial_layout=list(initial_layout))

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
    if not record_qubit_mapping:
        qubit_mapping = {i: i for i in range(circuit.num_qubits)}
    elif initial_layout is not None:
        qubit_mapping = physical_qubit_mapping_from_layout(circuit, initial_layout)
    else:
        qubit_mapping = get_physical_qubit_mapping(circuit, backend, optimization_level=optimization_level)

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
        total_billed_seconds=(
            sum(b.billed_seconds for b in batch_records)
            if all(b.billed_seconds is not None for b in batch_records) else None
        ),
        aggregated_counts=dict(total_counts),
        twirl_mask=list(twirl_mask) if twirl_mask is not None else None,
        transpiled_stats=transpiled_circuit_stats(circuit, backend, optimization_level, initial_layout),
    )

    return total_counts, record


MAX_IN_FLIGHT_ENV = "PIASTQ_MAX_IN_FLIGHT"


def max_jobs_in_flight() -> int:
    """How many circuits may be submitted to the service at once (env var set
    by run_all.py for the real machine; 1 = strictly sequential, the default
    and what simulators use). On PIAST-Q every job waits minutes in the shared
    queue, so sequential submission is limited by that wait, not by the 18.5 s
    of QPU time -- queueing many jobs together overlaps the waits."""
    try:
        return max(1, int(os.environ.get(MAX_IN_FLIGHT_ENV, "1")))
    except ValueError:
        return 1


def run_circuits_concurrently(specs, make_sampler, on_record=None, max_workers: Optional[int] = None):
    """Runs several independent circuits at once.

    `specs`: list of dicts, each holding `circuit` plus every other keyword of
    `run_circuit_with_batching_recorded` (circuit_id, iteration_id, ...).
    `make_sampler`: builds a fresh sampler per circuit -- samplers carry
    mutable state (shots, transpile layout), so threads must not share one.
    `on_record(record)`: called in the CALLING thread as each circuit finishes
    (not for circuits reused by --resume), so results reach disk immediately.

    Returns [(counts, record)] in `specs` order. If any circuit fails (e.g.
    BudgetExceeded), circuits not yet started are cancelled, those already in
    flight still finish and are recorded, and the first error is re-raised.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = max_workers or max_jobs_in_flight()
    results = [None] * len(specs)

    def run_one(index):
        spec = dict(specs[index])
        circuit = spec.pop("circuit")
        return run_circuit_with_batching_recorded(circuit, make_sampler(), **spec)

    if workers <= 1 or len(specs) <= 1:
        for index in range(len(specs)):
            results[index] = run_one(index)
            if on_record is not None and results[index][1] is not None:
                on_record(results[index][1])
        return results

    first_error = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, index): index for index in range(len(specs))}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                results[futures[future]] = future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if first_error is None:
                    first_error = exc
                    for other in futures:
                        other.cancel()
                continue
            record = results[futures[future]][1]
            if on_record is not None and record is not None:
                on_record(record)
    if first_error is not None:
        raise first_error
    return results


def _member_batch(joint_batch: BatchRecord, offset: int, width: int, total_width: int, pack_info: Dict[str, Any]) -> BatchRecord:
    from dataclasses import replace

    from piastq_execution.packing import marginalize_counts

    marginal = marginalize_counts(joint_batch.raw_counts, offset, width)
    metadata = dict(joint_batch.job_metadata) if isinstance(joint_batch.job_metadata, dict) else {}
    metadata["pack"] = pack_info
    return replace(
        joint_batch,
        raw_counts=marginal,
        shots_returned=sum(marginal.values()),
        job_metadata=metadata,
        # the job's billed time is shared between its circuits by qubit share
        billed_seconds=(joint_batch.billed_seconds * width / total_width
                        if joint_batch.billed_seconds is not None else None),
    )


def member_record_from_joint(joint: RawCountsRecord, spec: Dict[str, Any], placement, total_width: int,
                             pack_info: Dict[str, Any]) -> RawCountsRecord:
    """The ordinary one-circuit RawCountsRecord of one member of a packed job:
    its own marginal counts, its physical qubits, and its share of the job's
    time, so everything downstream (evaluation, mitigation, statistics) treats
    it exactly like a circuit that had been run alone."""
    batches = [_member_batch(b, placement.offset, placement.width, total_width, pack_info)
               for b in joint.batches]
    counts: Counter = Counter()
    for batch in batches:
        for bitstring, count in batch.raw_counts.items():
            counts[bitstring] += count
    stats = dict(joint.transpiled_stats) if isinstance(joint.transpiled_stats, dict) else None
    if stats is not None:
        stats["note"] = "statistics of the whole packed job, not of this circuit alone"
    return RawCountsRecord(
        algorithm=spec["algorithm"],
        objective_mode=spec["objective_mode"],
        dataset=spec["dataset"],
        circuit_id=spec["circuit_id"],
        cluster_id=spec.get("cluster_id"),
        iteration_id=spec.get("iteration_id"),
        subproblem_id=spec.get("subproblem_id"),
        backend_name=joint.backend_name,
        backend_version=joint.backend_version,
        physical_qubit_mapping={i: placement.start + i for i in range(placement.width)},
        batches=batches,
        total_shots_requested=sum(b.shots_requested for b in batches),
        total_shots_returned=sum(b.shots_returned for b in batches),
        total_wall_clock_seconds=sum(b.wall_clock_seconds for b in batches),
        aggregated_counts=dict(counts),
        twirl_mask=list(spec["twirl_mask"]) if spec.get("twirl_mask") is not None else None,
        transpiled_stats=stats,
        total_billed_seconds=(sum(b.billed_seconds for b in batches)
                              if all(b.billed_seconds is not None for b in batches) else None),
    )


def run_packed_circuits(specs, make_sampler, on_record=None, max_workers: Optional[int] = None, slots=None,
                        label: str = "pack"):
    """Runs many independent circuits as few hardware jobs as possible.

    `specs` as for `run_circuits_concurrently` (each with its `circuit`,
    `circuit_id`, `iteration_id`, optional `twirl_mask`, `completed`, shot
    settings; `initial_layout` is ignored -- placement is decided here, see
    piastq_execution.packing). Circuits are packed into jobs of up to 20
    qubits, the jobs run concurrently, and each circuit gets back its own
    marginal counts and its own RawCountsRecord, exactly like an unpacked run.

    Returns [(counts, record)] in `specs` order; `record` is None for circuits
    reused by --resume (already in the raw-counts file, so not written again).
    A job is skipped entirely when all its circuits were already completed.
    """
    from piastq_execution.packing import build_packed_circuit, load_pack_slots, plan_packs

    slots = list(slots) if slots is not None else load_pack_slots()
    widths = [spec["circuit"].remove_final_measurements(inplace=False).num_qubits for spec in specs]
    packs = plan_packs(widths, slots)
    results = [None] * len(specs)

    def key(spec):
        return (spec["circuit_id"], spec.get("iteration_id"))

    def prior_of(spec):
        completed = spec.get("completed") or {}
        return completed.get(key(spec))

    pack_specs, pack_by_id = [], {}
    for number, pack in enumerate(packs):
        priors = [prior_of(specs[p.member_index]) for p in pack]
        if all(prior is not None for prior in priors):
            for placement, prior in zip(pack, priors):
                results[placement.member_index] = (Counter(prior["aggregated_counts"]), None)
            print(f"  [resume] {label}{number}: all {len(pack)} circuits already executed")
            continue
        combined, layout = build_packed_circuit([spec["circuit"] for spec in specs], pack)
        base = specs[pack[0].member_index]
        pack_id = f"{label}{number}"
        pack_by_id[pack_id] = pack
        pack_specs.append(dict(
            circuit=combined, algorithm=base["algorithm"], objective_mode=base["objective_mode"],
            dataset=base["dataset"], circuit_id=pack_id, backend=base["backend"],
            iteration_id=base.get("iteration_id"), shots_per_batch=base["shots_per_batch"],
            num_batches=base["num_batches"], remainder_shots=base["remainder_shots"],
            initial_layout=layout, record_qubit_mapping=False))
        print(f"  [pack] {pack_id}: {len(pack)} circuits, {combined.num_qubits} qubits "
              f"(widths {[p.width for p in pack]})")

    def on_joint(joint):
        pack = pack_by_id[joint.circuit_id]
        total_width = sum(p.width for p in pack)
        info = {"pack_id": joint.circuit_id, "job_qubits": total_width, "members": len(pack)}
        for placement in pack:
            spec = specs[placement.member_index]
            prior = prior_of(spec)
            if prior is not None:               # --resume: keep the saved result, write nothing
                results[placement.member_index] = (Counter(prior["aggregated_counts"]), None)
                continue
            record = member_record_from_joint(joint, spec, placement, total_width, info)
            results[placement.member_index] = (Counter(record.aggregated_counts), record)
            if on_record is not None:
                on_record(record)

    run_circuits_concurrently(pack_specs, make_sampler, on_record=on_joint, max_workers=max_workers)
    return results


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
