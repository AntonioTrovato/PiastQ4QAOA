"""Tests for piastq_execution.run_calibration (packed-slot MEM/M3 calibration).

A fake machine with KNOWN independent per-qubit readout errors answers the
calibration jobs, so the recovered matrices can be checked against the truth,
including the window matrices derived for circuits placed inside a slot."""

import itertools
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hermetic_env  # noqa: F401,E402  (scrubs PIASTQ_* variables)

import numpy as np

from piastq_execution.mitigation import CalibrationStore, correct_counts, make_m3_calibration_record
from piastq_execution.run_calibration import (
    calibrate_m3_all,
    calibrate_mem_slots,
    calibration_job_count,
    compare_m3_calibrations,
)

SLOTS = [(0, 2), (2, 2), (4, 1)]          # 5 qubits: small enough for exact outcome tables
# true per-qubit readout error: P(read 1 | prepared 0), P(read 0 | prepared 1)
E01 = [0.02, 0.05, 0.03, 0.08, 0.04]
E10 = [0.06, 0.03, 0.10, 0.02, 0.07]


def true_matrix(qubit):
    """[measured, prepared] for one qubit."""
    return np.array([[1 - E01[qubit], E10[qubit]], [E01[qubit], 1 - E10[qubit]]])


def true_tensor(qubits):
    out = np.array([[1.0]])
    for q in reversed(list(qubits)):      # qubit 0 = least significant bit
        out = np.kron(out, true_matrix(q))
    return out


class _Dist:
    def __init__(self, p):
        self.p = p

    def binary_probabilities(self):
        return self.p


class _Result:
    def __init__(self, p):
        self.quasi_dists = [_Dist(p)]


class _Handle:
    def __init__(self, p):
        self.p = p

    def result(self):
        return _Result(self.p)


class _Options:
    shots = None


class NoisyMachine:
    """Prepares the basis state given by the circuit's X gates and reads every
    qubit through its own known readout error (exact probabilities)."""

    def __init__(self):
        self.options = _Options()
        self.layouts = []

    def set_transpile_options(self, **kw):
        if "initial_layout" in kw:
            self.layouts.append(list(kw["initial_layout"]))

    def run(self, circuits):
        circuit = circuits[0]
        n = circuit.num_qubits
        prepared = [0] * n
        for inst in circuit.data:
            if inst.operation.name == "x":
                prepared[circuit.find_bit(inst.qubits[0]).index] ^= 1
        probabilities = {}
        for measured in itertools.product((0, 1), repeat=n):
            p = 1.0
            for q in range(n):
                p *= (1 - (E10[q] if prepared[q] else E01[q])) if measured[q] == prepared[q] \
                    else (E10[q] if prepared[q] else E01[q])
            if p > 0:
                probabilities["".join(str(b) for b in reversed(measured))] = p
        return _Handle(probabilities)


class FakeBackend:
    name = "fake"
    backend_version = "0"


class TestCalibration(unittest.TestCase):
    def test_job_count(self):
        self.assertEqual(calibration_job_count(SLOTS), 2 + 2 ** 2)
        self.assertEqual(calibration_job_count([(0, 7), (7, 7), (14, 6)]), 130)

    def test_m3_recovers_every_qubits_matrix_from_two_jobs(self):
        machine = NoisyMachine()
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            record = calibrate_m3_all(SLOTS, machine, FakeBackend(), store)
            self.assertEqual(record.physical_qubits, [0, 1, 2, 3, 4])
            self.assertEqual(record.data["calibration_jobs"], 2)
            for q, cal in enumerate(record.single_qubit_cals()):
                np.testing.assert_allclose(cal, true_matrix(q), atol=0.006)
            self.assertIsNotNone(store.load("m3", [0, 1, 2, 3, 4]))
        self.assertTrue(all(layout == [0, 1, 2, 3, 4] for layout in machine.layouts))   # pinned to the slot qubits

    def test_mem_recovers_each_slots_full_matrix_from_shared_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            records = calibrate_mem_slots(SLOTS, NoisyMachine(), FakeBackend(), store)
            self.assertEqual([r.physical_qubits for r in records], [[0, 1], [2, 3], [4]])
            for record in records:
                np.testing.assert_allclose(record.confusion_matrix(), true_tensor(record.physical_qubits), atol=0.01)
                self.assertEqual(record.data["calibration_jobs"], 4)

    def test_derived_window_matrices_match_the_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            calibrate_m3_all(SLOTS, NoisyMachine(), FakeBackend(), store)
            calibrate_mem_slots(SLOTS, NoisyMachine(), FakeBackend(), store)
            # a 1-qubit circuit placed on qubit 3 (second half of slot [2,3]) and a 2-qubit one on [0,1]
            mem3 = store.load_window("mem", [3])
            np.testing.assert_allclose(mem3.confusion_matrix(), true_tensor([3]), atol=0.01)
            mem01 = store.load_window("mem", [0, 1])
            np.testing.assert_allclose(mem01.confusion_matrix(), true_tensor([0, 1]), atol=0.01)
            m3 = store.load_window("m3", [2, 3, 4])
            for cal, q in zip(m3.single_qubit_cals(), [2, 3, 4]):
                np.testing.assert_allclose(cal, true_matrix(q), atol=0.006)
            self.assertEqual(m3.num_qubits, 3)
            self.assertIsNone(store.load_window("mem", [0, 1, 2]))          # spans two slots: not available

    def test_correction_with_a_derived_window_recovers_the_true_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            calibrate_m3_all(SLOTS, NoisyMachine(), FakeBackend(), store)
            calibrate_mem_slots(SLOTS, NoisyMachine(), FakeBackend(), store)
            window = [2, 3]                                                     # a circuit placed in the second slot
            ideal = np.array([0.0, 0.7, 0.0, 0.3])                            # P(bitstring) over '00','01','10','11'
            observed = true_tensor(window) @ ideal
            raw_counts = {format(i, "02b"): int(round(1e6 * observed[i])) for i in range(4)}
            for method in ("mem", "m3"):
                record = store.load_window(method, window)
                corrected = correct_counts(method, raw_counts, record, 2)
                total = sum(corrected.values())
                got = [corrected.get(format(i, "02b"), 0.0) / total for i in range(4)]
                np.testing.assert_allclose(got, ideal, atol=0.02, err_msg=method)


class TestCompareM3(unittest.TestCase):
    def test_reports_max_change_per_qubit(self):
        a = make_m3_calibration_record([np.array([[0.99, 0.02], [0.01, 0.98]])], [0], "b", "v", 200, 1.0)
        b = make_m3_calibration_record([np.array([[0.97, 0.02], [0.03, 0.98]])], [0], "b", "v", 200, 1.0)
        report = compare_m3_calibrations(a, b)
        self.assertAlmostEqual(report["max_abs_change"], 0.02)
        self.assertEqual(len(report["max_abs_change_per_qubit"]), 1)
        self.assertEqual(report["width"], 1)


if __name__ == "__main__":
    unittest.main()
