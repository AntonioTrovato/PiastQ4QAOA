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

Calibrates by circuit width using logical qubit indices 0..width-1 as the
physical qubit list -- the simplification a standalone calibration pass
needs to make. If you need calibration keyed to a *specific* physical qubit
mapping (e.g. exactly the qubits a given cluster's circuit actually gets
transpiled onto -- see piastq_execution.raw_counts.get_physical_qubit_mapping),
call build_mem_calibration_circuits()/build_m3_calibration_circuits() and
CalibrationStore.save() directly with that mapping instead of running this
script as-is.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler

from piastq_execution.mitigation import (
    CalibrationStore,
    build_confusion_matrix,
    build_m3_calibration_circuits,
    build_m3_single_qubit_cals,
    build_mem_calibration_circuits,
    make_m3_calibration_record,
    make_mem_calibration_record,
)
from piastq_execution.raw_counts import get_backend_identity

CALIBRATION_SHOTS = 200  # one 200-shot batch per calibration circuit


def _run_and_count(circuit, sampler, shots=CALIBRATION_SHOTS):
    sampler.options.shots = shots
    result = sampler.run([circuit]).result()
    probabilities = result.quasi_dists[0].binary_probabilities()
    return {bitstring: int(round(prob * shots)) for bitstring, prob in probabilities.items()}


def calibrate_mem(width, sampler, backend, store):
    calibration_counts = {}
    for bitstring, circuit in build_mem_calibration_circuits(width):
        calibration_counts[bitstring] = _run_and_count(circuit, sampler)

    matrix = build_confusion_matrix(calibration_counts, width)
    physical_qubits = list(range(width))
    backend_identity = get_backend_identity(backend)
    record = make_mem_calibration_record(
        matrix, physical_qubits,
        backend_identity["backend_name"], backend_identity["backend_version"],
        CALIBRATION_SHOTS,
    )
    path = store.save(record)
    print(f"  MEM width={width}: saved calibration to {path}")


def calibrate_m3(width, sampler, backend, store):
    calibration_counts = {}
    for qubit, prepared_bit, circuit in build_m3_calibration_circuits(width):
        calibration_counts[(qubit, prepared_bit)] = _run_and_count(circuit, sampler)

    single_qubit_cals = build_m3_single_qubit_cals(calibration_counts, width)
    physical_qubits = list(range(width))
    backend_identity = get_backend_identity(backend)
    record = make_m3_calibration_record(
        single_qubit_cals, physical_qubits,
        backend_identity["backend_name"], backend_identity["backend_version"],
        CALIBRATION_SHOTS,
    )
    path = store.save(record)
    print(f"  M3 width={width}: saved calibration to {path}")


def run_calibration(widths):
    provider = AQTProvider("ACCESS_TOKEN")
    backend = provider.get_backend("offline_simulator_no_noise")
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
