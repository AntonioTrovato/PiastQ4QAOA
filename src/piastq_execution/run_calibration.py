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

import os
import sys
import time

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
from piastq_execution.raw_counts import get_backend_identity

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
        print(f"\n=== Calibrating width={width} ===")
        calibrate_mem(width, sampler, backend, store)
        calibrate_m3(width, sampler, backend, store)


if __name__ == "__main__":
    requested_widths = [int(w) for w in sys.argv[1:]] or list(range(1, 8))
    run_calibration(requested_widths)
