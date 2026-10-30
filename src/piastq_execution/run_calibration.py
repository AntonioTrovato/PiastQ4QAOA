"""Runs MEM and M3 readout calibration on PIAST-Q and stores it via
CalibrationStore, so every other script's `correct_counts(...)` call can reuse
it without re-touching hardware.

Calibration is done on the same fixed qubit SLOTS the packed QAOA jobs use
(piastq_execution.packing, configs/backend.yaml `pack_slots`, default
[0..6], [7..13], [14..19]) and under the same conditions (all slots active at
once), in few jobs:

  * M3: 2 jobs -- every qubit prepared in |0>, then every qubit in |1>. That
    gives each of the 20 qubits its 2x2 assignment matrix.
  * MEM: 2**(widest slot) = 128 jobs. In job j every slot prepares basis
    state (j mod 2**slot_width), so after 128 jobs each slot has seen all of
    its own basis states -> the slot's full confusion matrix. The matrix of
    any smaller window inside a slot (a circuit placed there) is derived from
    it by marginalisation (mitigation.CalibrationStore.load_window).

    cd src/piastq_execution
    python run_calibration.py            # start of the run: M3 + MEM
    python run_calibration.py drift      # END of the run: re-measure M3 (2 jobs)
                                         # -> calibration/drift/m3_drift_report.json

Also saves a timestamped snapshot of the backend's own reported qubit
calibration (T1/T2/etc., backend_calibration_snapshot.py) before running
anything.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qiskit import QuantumCircuit
from qiskit_aqt_provider.primitives import AQTSampler

from piastq_execution.backend_calibration_snapshot import save_backend_calibration_snapshot
from piastq_execution.backend_config import get_backend
from piastq_execution.mitigation import (
    CalibrationStore,
    build_confusion_matrix,
    build_m3_single_qubit_cals,
    make_m3_calibration_record,
    make_mem_calibration_record,
)
from piastq_execution.packing import load_pack_slots, marginalize_counts, slot_qubits
from piastq_execution.raw_counts import get_backend_identity, max_jobs_in_flight, resume_enabled

CALIBRATION_SHOTS = 200  # one 200-shot batch per calibration job


def calibration_job_count(slots=None):
    """Jobs the whole calibration takes: 2 (M3) + 2**(widest slot) (MEM)."""
    slots = slots or load_pack_slots()
    return 2 + 2 ** max(width for _, width in slots)


def _run_and_count(circuit, sampler, initial_layout, shots=CALIBRATION_SHOTS):
    """Returns (counts, elapsed_seconds) of one calibration job, pinned to
    `initial_layout` (the physical qubits it must run on)."""
    from piastq_execution.qpu_usage import check_budget
    from piastq_execution.raw_counts import _submit_and_wait

    check_budget()
    sampler.set_transpile_options(optimization_level=3, initial_layout=list(initial_layout))
    sampler.options.shots = shots
    start = time.time()
    _job, result = _submit_and_wait(sampler, circuit)
    elapsed = time.time() - start
    probabilities = result.quasi_dists[0].binary_probabilities()
    counts = {bitstring: int(round(prob * shots)) for bitstring, prob in probabilities.items()}
    return counts, elapsed


def _run_many(items, sampler, make_sampler, initial_layout):
    """Runs the calibration jobs `items` = [(key, circuit), ...] and returns
    ({key: counts}, summed per-job seconds). With more than one job allowed in
    flight (real machine, raw_counts.max_jobs_in_flight) and a sampler
    factory they are submitted together so the queue waits overlap; otherwise
    strictly one after the other with `sampler`."""
    workers = max_jobs_in_flight()
    counts_by_key, total_seconds = {}, 0.0
    if workers <= 1 or make_sampler is None:
        for key, circuit in items:
            counts, elapsed = _run_and_count(circuit, sampler, initial_layout)
            counts_by_key[key] = counts
            total_seconds += elapsed
        return counts_by_key, total_seconds

    from concurrent.futures import ThreadPoolExecutor, as_completed

    first_error = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_and_count, circuit, make_sampler(), initial_layout): key
                   for key, circuit in items}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                counts, elapsed = future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if first_error is None:
                    first_error = exc
                    for other in futures:
                        other.cancel()
                continue
            counts_by_key[futures[future]] = counts
            total_seconds += elapsed
    if first_error is not None:
        raise first_error
    return counts_by_key, total_seconds


def _billed_total(num_jobs):
    """Billed QPU seconds of `num_jobs` calibration jobs on the real machine
    (None on simulators): queue wait makes wall-clock useless as a cost."""
    from piastq_execution.qpu_usage import modeled_billed_seconds

    per_job = modeled_billed_seconds(CALIBRATION_SHOTS)
    return None if per_job is None else per_job * num_jobs


def _prepared_circuit(num_qubits, ones):
    """A circuit preparing |1> on the qubits listed in `ones` (|0> elsewhere)
    and measuring every qubit once (qubit k into clbit k)."""
    circuit = QuantumCircuit(num_qubits)
    for qubit in ones:
        circuit.x(qubit)
    circuit.measure_all()
    return circuit


def calibrate_m3_all(slots, sampler, backend, store, make_sampler=None):
    """M3 for every slot qubit in 2 jobs (all |0>, all |1>); one record."""
    qubits = slot_qubits(slots)
    n = len(qubits)
    by_state, seconds = _run_many(
        [(0, _prepared_circuit(n, [])), (1, _prepared_circuit(n, range(n)))], sampler, make_sampler, qubits)
    counts = {(q, prepared): by_state[prepared] for q in range(n) for prepared in (0, 1)}
    cals = build_m3_single_qubit_cals(counts, n)
    identity = get_backend_identity(backend)
    record = make_m3_calibration_record(
        cals, qubits, identity["backend_name"], identity["backend_version"], CALIBRATION_SHOTS, seconds,
        calibration_billed_seconds=_billed_total(2))
    record.data["calibration_jobs"] = 2
    path = store.save(record)
    print(f"  M3 for qubits {qubits[0]}..{qubits[-1]}: saved calibration to {path} ({seconds:.2f}s)")
    return record


def calibrate_mem_slots(slots, sampler, backend, store, make_sampler=None):
    """MEM for every slot in 2**(widest slot) jobs shared by all slots; one
    record per slot."""
    qubits = slot_qubits(slots)
    n = len(qubits)
    jobs = 2 ** max(width for _, width in slots)
    first_index, index = [], 0
    for _, width in slots:
        first_index.append(index)
        index += width

    items = []
    for j in range(jobs):
        ones = []
        for (_, width), base in zip(slots, first_index):
            state = j % (2 ** width)
            ones += [base + k for k in range(width) if (state >> k) & 1]
        items.append((j, _prepared_circuit(n, ones)))
    by_job, seconds = _run_many(items, sampler, make_sampler, qubits)

    identity = get_backend_identity(backend)
    records = []
    for (start, width), base in zip(slots, first_index):
        calibration_counts = {}
        for j in range(jobs):
            prepared = format(j % (2 ** width), f"0{width}b")
            slot_counts = marginalize_counts(by_job[j], base, width)
            bucket = calibration_counts.setdefault(prepared, {})
            for measured, count in slot_counts.items():
                bucket[measured] = bucket.get(measured, 0) + count
        physical = list(range(start, start + width))
        record = make_mem_calibration_record(
            build_confusion_matrix(calibration_counts, width), physical,
            identity["backend_name"], identity["backend_version"], CALIBRATION_SHOTS, seconds,
            calibration_billed_seconds=_billed_total(jobs))
        record.data["calibration_jobs"] = jobs
        path = store.save(record)
        print(f"  MEM slot {physical[0]}..{physical[-1]}: saved calibration to {path} ({seconds:.2f}s)")
        records.append(record)
    return records


def _already_calibrated(store, slots):
    qubits = slot_qubits(slots)
    return (store.load("m3", qubits) is not None
            and all(store.load("mem", list(range(start, start + width))) is not None for start, width in slots))


def run_calibration():
    backend = get_backend()
    snapshot_path = save_backend_calibration_snapshot(backend)
    print(f"Saved backend calibration snapshot: {snapshot_path}")

    slots = load_pack_slots()
    store = CalibrationStore()
    if resume_enabled() and _already_calibrated(store, slots):
        print("Calibration already saved, skipping (resume).")
        return

    def make_sampler():
        sampler = AQTSampler(backend)
        sampler.set_transpile_options(optimization_level=3)
        return sampler

    sampler = make_sampler()
    print(f"\n=== Calibrating slots {slots}: {calibration_job_count(slots)} jobs ===")
    calibrate_m3_all(slots, sampler, backend, store, make_sampler)
    calibrate_mem_slots(slots, sampler, backend, store, make_sampler)


def compare_m3_calibrations(start_record, end_record):
    """Readout-drift summary between two M3 calibrations of the same qubits:
    the largest absolute change in any single-qubit assignment-matrix entry
    (overall and per qubit), plus the timestamps they were measured at.
    """
    import numpy as np

    start_cals = start_record.single_qubit_cals()
    end_cals = end_record.single_qubit_cals()
    per_qubit = [float(np.max(np.abs(a - b))) for a, b in zip(start_cals, end_cals)]
    return {
        "width": start_record.num_qubits,
        "start_timestamp": start_record.calibration_timestamp,
        "end_timestamp": end_record.calibration_timestamp,
        "max_abs_change_per_qubit": per_qubit,
        "max_abs_change": max(per_qubit) if per_qubit else 0.0,
    }


def run_drift_check():
    """Re-measures M3 (2 jobs) and compares it to the calibration saved at the
    start of the session, to quantify readout drift over the run. The fresh
    calibration goes to calibration/drift/ and never overwrites the primary
    one used for correction. Run once at the END of a hardware session; report:
    calibration/drift/m3_drift_report.json.
    """
    backend = get_backend()
    slots = load_pack_slots()

    def make_sampler():
        sampler = AQTSampler(backend)
        sampler.set_transpile_options(optimization_level=3)
        return sampler

    primary_store = CalibrationStore()
    drift_dir = os.path.join(primary_store.base_dir, "drift")
    drift_store = CalibrationStore(base_dir=drift_dir)

    start_record = primary_store.load("m3", slot_qubits(slots))
    if start_record is None:
        print("no start-of-session M3 calibration found, nothing to compare")
        return
    end_record = calibrate_m3_all(slots, make_sampler(), backend, drift_store, make_sampler)
    reports = [compare_m3_calibrations(start_record, end_record)]

    os.makedirs(drift_dir, exist_ok=True)
    path = os.path.join(drift_dir, "m3_drift_report.json")
    with open(path, "w") as f:
        json.dump({"generated": datetime.now(timezone.utc).isoformat(), "widths": reports}, f, indent=2)
    print(f"Saved drift report: {path}")


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "drift":
        run_drift_check()
    else:
        run_calibration()
