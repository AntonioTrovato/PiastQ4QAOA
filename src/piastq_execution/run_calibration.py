"""Runs full-MEM and M3 calibration on real PIAST-Q hardware and stores the
result via CalibrationStore, so every other script's `correct_counts(...)`
call can reuse it without re-touching hardware.

This is the concrete "how mitigation execution is done" entry point referred
to in the README: it is a normal execution tail, same shape as
qaoa_tcs/single_obj.py or igdec_qaoa/loch_qaoa_tcm_extract_circuits.py --
same provider/backend/sampler setup, same shot-batching helper, same
raw-counts persistence. It just runs calibration circuits instead of QAOA
circuits.

Run this on the machine with PIAST-Q/AQT access, not in an authoring/review
session:

    cd src/piastq_execution
    python run_calibration.py               # calibrates widths 1..7
    python run_calibration.py 2 7            # calibrates only widths 2 and 7
    python run_calibration.py drift          # END of session: re-measure M3 and
                                             # write calibration/drift/m3_drift_report.json

Calibrates by circuit width, pinned to the *same* physical qubits every real
QAOA/TREx execution uses for that width --
piastq_execution.qubit_layout.physical_layout_for_width(width), configured
once in configs/backend.yaml's `physical_qubits` list. This is what makes a
calibration built here safe to apply to counts collected by
raw_counts.run_circuit_with_batching_recorded() elsewhere: both pin the same
explicit `initial_layout` rather than each independently letting the
transpiler pick one, which offered no such guarantee.

Also saves a timestamped snapshot of the backend's own reported qubit
calibration (T1/T2/etc., see backend_calibration_snapshot.py) before running
anything -- distinct from the MEM/M3 calibration this script measures itself.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qiskit_aqt_provider.primitives import AQTSampler

from piastq_execution.backend_calibration_snapshot import save_backend_calibration_snapshot
from piastq_execution.backend_config import get_backend
from piastq_execution.mitigation import (
    CalibrationStore,
    build_confusion_matrix,
    build_m3_calibration_circuits,
    build_m3_single_qubit_cals,
    build_mem_calibration_circuits,
    make_m3_calibration_record,
    make_mem_calibration_record,
)
from piastq_execution.qubit_layout import physical_layout_for_width
from piastq_execution.raw_counts import get_backend_identity, resume_enabled

CALIBRATION_SHOTS = 200  # one 200-shot batch per calibration circuit


def _run_and_count(circuit, sampler, initial_layout, shots=CALIBRATION_SHOTS):
    """Returns (counts, elapsed_seconds) -- elapsed_seconds is the measured
    wall-clock time of this one circuit's sampler.run() call, summed by the
    caller across every calibration circuit to get
    CalibrationRecord.calibration_wall_clock_seconds.

    `initial_layout` pins this calibration circuit to the same physical
    qubits a real QAOA circuit of this width would use (see module
    docstring) -- applied before every run() call since a differently-sized
    circuit may have been submitted through the same sampler since the last
    call.
    """
    from piastq_execution.qpu_usage import check_budget

    check_budget()
    sampler.set_transpile_options(optimization_level=3, initial_layout=list(initial_layout))
    sampler.options.shots = shots
    start = time.time()
    result = sampler.run([circuit]).result()
    elapsed = time.time() - start
    probabilities = result.quasi_dists[0].binary_probabilities()
    counts = {bitstring: int(round(prob * shots)) for bitstring, prob in probabilities.items()}
    return counts, elapsed


def calibrate_mem(width, sampler, backend, store):
    physical_qubits = physical_layout_for_width(width)
    calibration_counts = {}
    total_wall_clock_seconds = 0.0
    for bitstring, circuit in build_mem_calibration_circuits(width):
        counts, elapsed = _run_and_count(circuit, sampler, physical_qubits)
        calibration_counts[bitstring] = counts
        total_wall_clock_seconds += elapsed

    matrix = build_confusion_matrix(calibration_counts, width)
    backend_identity = get_backend_identity(backend)
    record = make_mem_calibration_record(
        matrix, physical_qubits,
        backend_identity["backend_name"], backend_identity["backend_version"],
        CALIBRATION_SHOTS,
        total_wall_clock_seconds,
    )
    path = store.save(record)
    print(f"  MEM width={width}: saved calibration to {path} ({total_wall_clock_seconds:.2f}s)")


def calibrate_m3(width, sampler, backend, store):
    physical_qubits = physical_layout_for_width(width)
    calibration_counts = {}
    total_wall_clock_seconds = 0.0
    for qubit, prepared_bit, circuit in build_m3_calibration_circuits(width):
        counts, elapsed = _run_and_count(circuit, sampler, physical_qubits)
        calibration_counts[(qubit, prepared_bit)] = counts
        total_wall_clock_seconds += elapsed

    single_qubit_cals = build_m3_single_qubit_cals(calibration_counts, width)
    backend_identity = get_backend_identity(backend)
    record = make_m3_calibration_record(
        single_qubit_cals, physical_qubits,
        backend_identity["backend_name"], backend_identity["backend_version"],
        CALIBRATION_SHOTS,
        total_wall_clock_seconds,
    )
    path = store.save(record)
    print(f"  M3 width={width}: saved calibration to {path} ({total_wall_clock_seconds:.2f}s)")


def run_calibration(widths):
    backend = get_backend()
    snapshot_path = save_backend_calibration_snapshot(backend)
    print(f"Saved backend calibration snapshot: {snapshot_path}")

    sampler = AQTSampler(backend)
    sampler.set_transpile_options(optimization_level=3)

    store = CalibrationStore()

    for width in widths:
        layout = physical_layout_for_width(width)
        if resume_enabled() and store.load("mem", layout) is not None and store.load("m3", layout) is not None:
            print(f"\n=== width={width}: calibration already saved, skipping (resume) ===")
            continue
        print(f"\n=== Calibrating width={width} ===")
        calibrate_mem(width, sampler, backend, store)
        calibrate_m3(width, sampler, backend, store)


def compare_m3_calibrations(start_record, end_record):
    """Readout-drift summary between two M3 calibrations of the same width:
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


def run_drift_check(widths):
    """Re-measures M3 (2*width cheap circuits per width) and compares it to
    the calibration saved at the start of the session, to quantify readout
    drift over the run. The fresh calibration goes to calibration/drift/ and
    never overwrites the primary one used for correction. Run once at the END
    of a hardware session; report: calibration/drift/m3_drift_report.json.
    """
    backend = get_backend()
    sampler = AQTSampler(backend)
    sampler.set_transpile_options(optimization_level=3)

    primary_store = CalibrationStore()
    drift_dir = os.path.join(primary_store.base_dir, "drift")
    drift_store = CalibrationStore(base_dir=drift_dir)

    reports = []
    for width in widths:
        start_record = primary_store.load("m3", physical_layout_for_width(width))
        if start_record is None:
            print(f"  width={width}: no start-of-session M3 calibration found, skipping")
            continue
        calibrate_m3(width, sampler, backend, drift_store)
        end_record = drift_store.load("m3", physical_layout_for_width(width))
        reports.append(compare_m3_calibrations(start_record, end_record))

    os.makedirs(drift_dir, exist_ok=True)
    path = os.path.join(drift_dir, "m3_drift_report.json")
    with open(path, "w") as f:
        json.dump({"generated": datetime.now(timezone.utc).isoformat(), "widths": reports}, f, indent=2)
    print(f"Saved drift report: {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "drift":
        run_drift_check([int(w) for w in args[1:]] or list(range(1, 8)))
    else:
        run_calibration([int(w) for w in args] or list(range(1, 8)))
