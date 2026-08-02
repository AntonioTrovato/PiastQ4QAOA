"""Synthetic unit tests for piastq_execution.mitigation.

All calibration data here is tiny and hand-built (1-3 qubits) so expected
results can be verified by hand -- no real calibration data, no AQT/backend
calls. Circuit-building tests use small local QuantumCircuit objects only.
"""

import os
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.mitigation import (
    CalibrationStore,
    aggregate_trex_instances,
    build_confusion_matrix,
    build_m3_calibration_circuits,
    build_m3_single_qubit_cals,
    build_mem_calibration_circuits,
    build_trex_twirled_circuit,
    correct_counts,
    correct_counts_m3,
    correct_counts_with_matrix,
    generate_random_twirl_mask,
    make_m3_calibration_record,
    make_mem_calibration_record,
    undo_trex_twirl,
)


class TestMemCalibrationCircuits(unittest.TestCase):
    def test_builds_one_circuit_per_basis_state_with_correct_x_gates(self):
        circuits = build_mem_calibration_circuits(2)
        self.assertEqual(len(circuits), 4)
        labels = sorted(label for label, _ in circuits)
        self.assertEqual(labels, ["00", "01", "10", "11"])

        by_label = dict(circuits)
        # "10" -> qubit1=1, qubit0=0 (little-endian): X only on qubit 1
        ops = [instr.operation.name for instr in by_label["10"].data]
        x_qubits = [
            by_label["10"].find_bit(q).index
            for instr in by_label["10"].data if instr.operation.name == "x"
            for q in instr.qubits
        ]
        self.assertEqual(x_qubits, [1])
        self.assertIn("measure", ops)

        # "00" -> no X gates at all
        ops_00 = [instr.operation.name for instr in by_label["00"].data]
        self.assertNotIn("x", ops_00)


class TestBuildConfusionMatrix(unittest.TestCase):
    def test_matches_hand_computed_probabilities(self):
        calibration_counts = {
            "0": {"0": 900, "1": 100},   # prepared 0: measured 0 90%, 1 10%
            "1": {"0": 200, "1": 800},   # prepared 1: measured 0 20%, 1 80%
        }
        matrix = build_confusion_matrix(calibration_counts, num_qubits=1)
        expected = np.array([[0.9, 0.2], [0.1, 0.8]])
        np.testing.assert_allclose(matrix, expected)


class TestCorrectCountsWithMatrix(unittest.TestCase):
    def test_exact_recovery_for_invertible_matrix(self):
        # A = [[0.9,0.2],[0.1,0.8]]; ideal state |0> -> noisy p=[0.9,0.1] exactly.
        matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
        raw_counts = {"0": 900, "1": 100}
        corrected = correct_counts_with_matrix(raw_counts, matrix, num_qubits=1, method="nnls")
        self.assertAlmostEqual(corrected.get("0", 0.0), 1.0, places=6)
        self.assertAlmostEqual(sum(corrected.values()), 1.0, places=6)
        for p in corrected.values():
            self.assertGreaterEqual(p, 0.0)

    def test_nnls_stays_nonnegative_and_normalized_when_pinv_would_go_negative(self):
        # Exact (unconstrained) solve for this A, p is x = [1.125, -0.125]:
        # confirms this is a case where naive inversion produces a negative
        # "probability" and mitigation must not just pass that through.
        matrix = np.array([[0.9, 0.5], [0.1, 0.5]])
        p_noisy = np.array([0.95, 0.05])
        exact_unconstrained = np.linalg.inv(matrix) @ p_noisy
        self.assertLess(exact_unconstrained[1], 0.0)  # sanity-check the premise

        raw_counts = {"0": 950, "1": 50}
        for method in ("nnls", "pinv"):
            corrected = correct_counts_with_matrix(raw_counts, matrix, num_qubits=1, method=method)
            self.assertAlmostEqual(sum(corrected.values()), 1.0, places=6)
            for p in corrected.values():
                self.assertGreaterEqual(p, 0.0, f"method={method} produced a negative probability")

    def test_rejects_empty_counts(self):
        matrix = np.eye(2)
        with self.assertRaises(ValueError):
            correct_counts_with_matrix({}, matrix, num_qubits=1)


class TestM3Calibration(unittest.TestCase):
    def test_builds_two_circuits_per_qubit(self):
        circuits = build_m3_calibration_circuits(2)
        self.assertEqual(len(circuits), 4)
        keys = sorted((q, bit) for q, bit, _ in circuits)
        self.assertEqual(keys, [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_single_qubit_cals_match_hand_computed_marginals(self):
        # Qubit 0 calibration: prepared 0 -> always measures "00"; prepared 1 -> "01" 90%, "00" 10%.
        # Qubit 1 unused here (held at its own separate calibration).
        calibration_counts = {
            (0, 0): {"00": 1000},
            (0, 1): {"01": 900, "00": 100},
            (1, 0): {"00": 1000},
            (1, 1): {"10": 1000},
        }
        cals = build_m3_single_qubit_cals(calibration_counts, num_qubits=2)
        self.assertEqual(len(cals), 2)
        # qubit 0: P(measured=0|prepared=0)=1.0, P(measured=1|prepared=1)=0.9
        np.testing.assert_allclose(cals[0], np.array([[1.0, 0.1], [0.0, 0.9]]))
        # qubit 1: perfect readout
        np.testing.assert_allclose(cals[1], np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_correct_counts_m3_matches_kron_matrix_correction(self):
        cal0 = np.array([[0.9, 0.1], [0.1, 0.9]])
        cal1 = np.array([[0.8, 0.2], [0.2, 0.8]])
        raw_counts = {"00": 500, "01": 200, "10": 200, "11": 100}

        m3_result = correct_counts_m3(raw_counts, [cal0, cal1], physical_qubits=[0, 1])

        full_matrix = np.kron(cal1, cal0)
        matrix_result = correct_counts_with_matrix(raw_counts, full_matrix, num_qubits=2)

        self.assertEqual(set(m3_result.keys()) | {"__pad__"}, set(matrix_result.keys()) | {"__pad__"})
        for key in matrix_result:
            self.assertAlmostEqual(m3_result.get(key, 0.0), matrix_result[key], places=3)


class TestTrex(unittest.TestCase):
    def test_build_trex_twirled_circuit_strips_old_measure_and_applies_mask(self):
        from qiskit import QuantumCircuit

        base = QuantumCircuit(2, 2)
        base.h(0)
        base.cx(0, 1)
        base.barrier()
        base.measure([0, 1], [0, 1])

        twirled = build_trex_twirled_circuit(base, twirl_mask=[1, 0])

        names = [instr.operation.name for instr in twirled.data]
        self.assertEqual(names.count("measure"), 2)
        self.assertEqual(names.count("barrier"), 0)
        x_qubits = [
            twirled.find_bit(q).index
            for instr in twirled.data if instr.operation.name == "x"
            for q in instr.qubits
        ]
        self.assertEqual(x_qubits, [0])
        self.assertIn("h", names)
        self.assertIn("cx", names)

    def test_undo_trex_twirl_xors_known_mask(self):
        raw_counts = {"00": 40, "01": 30, "10": 20, "11": 10}
        # flip qubit 0 only: "00"->"01", "01"->"00", "10"->"11", "11"->"10"
        undone = undo_trex_twirl(raw_counts, twirl_mask=[1, 0])
        self.assertEqual(undone, {"01": 40, "00": 30, "11": 20, "10": 10})

    def test_aggregate_trex_instances_sums_undone_counts(self):
        instances = [
            ([1, 0], {"00": 10, "11": 5}),   # undoes to {"01":10, "10":5}
            ([0, 1], {"01": 8, "10": 2}),    # undoes to {"11":8, "00":2}
        ]
        aggregated = aggregate_trex_instances(instances)
        self.assertEqual(aggregated, {"01": 10, "10": 5, "11": 8, "00": 2})

    def test_generate_random_twirl_mask_length_and_values(self):
        rng = random.Random(42)
        mask = generate_random_twirl_mask(5, rng=rng)
        self.assertEqual(len(mask), 5)
        self.assertTrue(all(b in (0, 1) for b in mask))


class TestCalibrationStore(unittest.TestCase):
    def test_save_and_load_round_trip_mem(self):
        matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
        record = make_mem_calibration_record(
            matrix, physical_qubits=[3], backend_name="offline_simulator_no_noise",
            backend_version="1.0", shots_per_calibration_circuit=200,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            path = store.save(record)
            self.assertTrue(os.path.exists(path))

            loaded = store.load("mem", [3])
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(loaded.confusion_matrix(), matrix)
            self.assertEqual(loaded.backend_name, "offline_simulator_no_noise")

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            self.assertIsNone(store.load("mem", [0, 1]))

    def test_save_and_load_round_trip_m3(self):
        cals = [np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.8, 0.2], [0.2, 0.8]])]
        record = make_m3_calibration_record(
            cals, physical_qubits=[0, 1], backend_name="offline_simulator_no_noise",
            backend_version="1.0", shots_per_calibration_circuit=200,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            store.save(record)
            loaded = store.load("m3", [0, 1])
            self.assertIsNotNone(loaded)
            loaded_cals = loaded.single_qubit_cals()
            self.assertEqual(len(loaded_cals), 2)
            np.testing.assert_allclose(loaded_cals[0], cals[0])


class TestCorrectCountsDispatch(unittest.TestCase):
    def test_raw_method_just_normalizes(self):
        result = correct_counts("raw", {"0": 3, "1": 1}, calibration_record=None, num_qubits=1)
        self.assertAlmostEqual(result["0"], 0.75)
        self.assertAlmostEqual(result["1"], 0.25)

    def test_mem_method_requires_calibration_record(self):
        with self.assertRaises(ValueError):
            correct_counts("mem", {"0": 3, "1": 1}, calibration_record=None, num_qubits=1)

    def test_mem_method_uses_stored_record(self):
        matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
        record = make_mem_calibration_record(
            matrix, physical_qubits=[0], backend_name="b", backend_version="v",
            shots_per_calibration_circuit=200,
        )
        result = correct_counts("mem", {"0": 900, "1": 100}, calibration_record=record, num_qubits=1)
        self.assertAlmostEqual(result.get("0", 0.0), 1.0, places=6)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            correct_counts("bogus", {"0": 1}, calibration_record=None, num_qubits=1)


if __name__ == "__main__":
    unittest.main()
