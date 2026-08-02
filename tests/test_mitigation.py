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
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.mitigation import (
    DEFAULT_TREX_TWIRL_INSTANCES,
    CalibrationStore,
    _correct_counts_m3_reduced,
    aggregate_trex_instances,
    aggregate_trex_records,
    build_confusion_matrix,
    build_m3_calibration_circuits,
    build_m3_single_qubit_cals,
    build_mem_calibration_circuits,
    build_trex_twirled_circuit,
    correct_counts,
    correct_counts_m3,
    correct_counts_with_matrix,
    generate_random_twirl_mask,
    load_trex_twirl_instances,
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


class TestM3ReducedFallback(unittest.TestCase):
    """_correct_counts_m3_reduced() is the fallback correct_counts_m3() uses
    when `mthree` isn't installed -- unlike the old full-2**n-matrix
    fallback it replaced, it must never build a matrix larger than the
    number of distinct bitstrings actually observed."""

    def test_matches_full_matrix_when_support_is_complete(self):
        # With all 4 states observed, the reduced matrix IS the full matrix,
        # so this must agree with correct_counts_with_matrix() exactly --
        # same case the old full-matrix fallback was tested against.
        cal0 = np.array([[0.9, 0.1], [0.1, 0.9]])
        cal1 = np.array([[0.8, 0.2], [0.2, 0.8]])
        raw_counts = {"00": 500, "01": 200, "10": 200, "11": 100}

        reduced_result = _correct_counts_m3_reduced(raw_counts, [cal0, cal1], num_qubits=2)

        full_matrix = np.kron(cal1, cal0)
        matrix_result = correct_counts_with_matrix(raw_counts, full_matrix, num_qubits=2)

        self.assertEqual(set(reduced_result.keys()) | {"__pad__"}, set(matrix_result.keys()) | {"__pad__"})
        for key in matrix_result:
            self.assertAlmostEqual(reduced_result.get(key, 0.0), matrix_result[key], places=3)

    def test_restricts_to_observed_support_not_full_hilbert_space(self):
        # 3 qubits = 8 possible bitstrings, but only 2 are observed -- the
        # matrix built internally must be 2x2, not 8x8 (the whole point of
        # this fallback vs. the full-matrix approach MEM uses).
        cal = np.array([[0.95, 0.05], [0.05, 0.95]])
        raw_counts = {"000": 800, "111": 200}

        result = _correct_counts_m3_reduced(raw_counts, [cal, cal, cal], num_qubits=3)

        self.assertLessEqual(set(result.keys()), {"000", "111"})
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)

    def test_empty_counts_raises(self):
        cal = np.array([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            _correct_counts_m3_reduced({}, [cal], num_qubits=1)

    def test_correct_counts_m3_falls_back_when_mthree_unavailable(self):
        # Force the ImportError branch inside correct_counts_m3() without
        # touching the real installed mthree package.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mthree":
                raise ImportError("forced for test")
            return real_import(name, *args, **kwargs)

        cal = np.array([[0.9, 0.1], [0.1, 0.9]])
        raw_counts = {"0": 900, "1": 100}

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = correct_counts_m3(raw_counts, [cal], physical_qubits=[0])

        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)


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


class _FakeTrexRecord:
    """Minimal stand-in for RawCountsRecord -- aggregate_trex_records() is
    duck-typed on exactly these five attributes."""

    def __init__(self, circuit_id, cluster_id, iteration_id, subproblem_id, twirl_mask, aggregated_counts):
        self.circuit_id = circuit_id
        self.cluster_id = cluster_id
        self.iteration_id = iteration_id
        self.subproblem_id = subproblem_id
        self.twirl_mask = twirl_mask
        self.aggregated_counts = aggregated_counts


class TestLoadTrexTwirlInstances(unittest.TestCase):
    def test_missing_file_falls_back_to_default(self):
        self.assertEqual(load_trex_twirl_instances("/tmp/piastq_does_not_exist.yaml"), DEFAULT_TREX_TWIRL_INSTANCES)

    def test_reads_configured_value(self):
        path = "/tmp/piastq_test_trex_instances.yaml"
        with open(path, "w") as f:
            f.write("trex_twirl_instances: 7\n")
        try:
            self.assertEqual(load_trex_twirl_instances(path), 7)
        finally:
            os.remove(path)


class TestAggregateTrexRecords(unittest.TestCase):
    def test_groups_by_cluster_iteration_subproblem_and_undoes_twirl(self):
        # Two twirl instances for circuit A (cluster=0, iteration=1), one for
        # circuit B (cluster=1, iteration=1) -- same iteration_id, must not
        # be merged together since cluster_id differs.
        records = [
            _FakeTrexRecord("A_trex0", cluster_id=0, iteration_id=1, subproblem_id=None,
                             twirl_mask=[1, 0], aggregated_counts={"00": 10, "11": 5}),
            _FakeTrexRecord("A_trex1", cluster_id=0, iteration_id=1, subproblem_id=None,
                             twirl_mask=[0, 1], aggregated_counts={"01": 8, "10": 2}),
            _FakeTrexRecord("B_trex0", cluster_id=1, iteration_id=1, subproblem_id=None,
                             twirl_mask=[0, 0], aggregated_counts={"00": 3}),
        ]
        result = aggregate_trex_records(records)
        self.assertEqual(set(result.keys()), {(0, 1, None), (1, 1, None)})
        # matches test_aggregate_trex_instances_sums_undone_counts's hand-verified sum
        self.assertEqual(result[(0, 1, None)], {"01": 10, "10": 5, "11": 8, "00": 2})
        self.assertEqual(result[(1, 1, None)], {"00": 3})

    def test_igdec_style_keys_use_subproblem_not_cluster(self):
        # IGDec-QAOA never sets cluster_id (stays None); iteration+subproblem
        # is what distinguishes circuits there.
        records = [
            _FakeTrexRecord("x_trex0", cluster_id=None, iteration_id=1, subproblem_id=0,
                             twirl_mask=[1], aggregated_counts={"0": 5, "1": 5}),
            _FakeTrexRecord("x_trex1", cluster_id=None, iteration_id=1, subproblem_id=1,
                             twirl_mask=[1], aggregated_counts={"0": 5, "1": 5}),
        ]
        result = aggregate_trex_records(records)
        self.assertEqual(set(result.keys()), {(None, 1, 0), (None, 1, 1)})

    def test_record_without_twirl_mask_raises(self):
        records = [
            _FakeTrexRecord("not_trex", cluster_id=0, iteration_id=1, subproblem_id=None,
                             twirl_mask=None, aggregated_counts={"0": 1}),
        ]
        with self.assertRaises(ValueError):
            aggregate_trex_records(records)


class TestCalibrationStore(unittest.TestCase):
    def test_save_and_load_round_trip_mem(self):
        matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
        record = make_mem_calibration_record(
            matrix, physical_qubits=[3], backend_name="offline_simulator_no_noise",
            backend_version="1.0", shots_per_calibration_circuit=200,
            calibration_wall_clock_seconds=1.5,
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
            calibration_wall_clock_seconds=0.8,
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
            shots_per_calibration_circuit=200, calibration_wall_clock_seconds=0.5,
        )
        result = correct_counts("mem", {"0": 900, "1": 100}, calibration_record=record, num_qubits=1)
        self.assertAlmostEqual(result.get("0", 0.0), 1.0, places=6)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            correct_counts("bogus", {"0": 1}, calibration_record=None, num_qubits=1)

    def test_trex_method_normalizes_already_aggregated_counts(self):
        # "trex" counts arriving here are assumed already twirl-undone/
        # aggregated (aggregate_trex_records()) -- correct_counts() only
        # normalizes, same as "raw".
        result = correct_counts("trex", {"0": 3, "1": 1}, calibration_record=None, num_qubits=1)
        self.assertAlmostEqual(result["0"], 0.75)
        self.assertAlmostEqual(result["1"], 0.25)


if __name__ == "__main__":
    unittest.main()
