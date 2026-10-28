"""Synthetic unit test for piastq_execution.run_calibration.

Uses a fake sampler (scripted quasi-distributions, no AQT/backend call at
all) and a tiny 1-qubit width so the resulting calibration data can be
verified by hand.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.mitigation import CalibrationStore
from piastq_execution.qubit_layout import physical_layout_for_width
from piastq_execution.run_calibration import calibrate_m3, calibrate_mem, compare_m3_calibrations


class FakeQuasiDist:
    def __init__(self, probabilities):
        self._probabilities = probabilities

    def binary_probabilities(self):
        return self._probabilities


class FakeResult:
    def __init__(self, probabilities):
        self.quasi_dists = [FakeQuasiDist(probabilities)]


class FakeSamplerOptions:
    shots = None


class FakeSampler:
    """Scripted responses, one per call, in the exact order the calibration
    circuits are submitted: for width=1, build_mem_calibration_circuits(1)
    yields "0" then "1"; build_m3_calibration_circuits(1) yields
    (0,0) then (0,1). Also records every set_transpile_options(...) call so
    tests can verify the pinned qubit layout (see qubit_layout.py) is applied
    identically to every calibration circuit, not just assumed."""

    def __init__(self, probabilities_per_call):
        self.options = FakeSamplerOptions()
        self._responses = list(probabilities_per_call)
        self.transpile_options_calls = []

    def set_transpile_options(self, **fields):
        self.transpile_options_calls.append(dict(fields))

    def run(self, circuits):
        probabilities = self._responses.pop(0)
        return _RunHandle(probabilities)


class _RunHandle:
    def __init__(self, probabilities):
        self._probabilities = probabilities

    def result(self):
        return FakeResult(self._probabilities)


class FakeBackend:
    name = "fake_offline_simulator"
    backend_version = "0.0-test"


class TestCalibrateMem(unittest.TestCase):
    def test_width_1_perfect_readout_round_trips(self):
        # prepared "0" always measures "0"; prepared "1" always measures "1"
        sampler = FakeSampler([{"0": 1.0}, {"1": 1.0}])
        backend = FakeBackend()

        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            calibrate_mem(1, sampler, backend, store)

            record = store.load("mem", [0])
            self.assertIsNotNone(record)
            matrix = record.confusion_matrix()
            self.assertEqual(matrix.shape, (2, 2))
            self.assertAlmostEqual(matrix[0, 0], 1.0)
            self.assertAlmostEqual(matrix[1, 1], 1.0)
            self.assertEqual(record.backend_name, "fake_offline_simulator")
            self.assertIsInstance(record.calibration_wall_clock_seconds, float)
            self.assertGreaterEqual(record.calibration_wall_clock_seconds, 0.0)

            # Every calibration circuit must be pinned to the same physical
            # qubits a real width-1 QAOA circuit would use -- not left to the
            # transpiler's automatic layout selection.
            expected_layout = physical_layout_for_width(1)
            self.assertEqual(len(sampler.transpile_options_calls), 2)
            for call in sampler.transpile_options_calls:
                self.assertEqual(call.get("initial_layout"), expected_layout)
                self.assertEqual(call.get("optimization_level"), 3)


class TestCalibrateM3(unittest.TestCase):
    def test_width_1_perfect_readout_round_trips(self):
        # (qubit0, prepared=0) -> measures "0"; (qubit0, prepared=1) -> measures "1"
        sampler = FakeSampler([{"0": 1.0}, {"1": 1.0}])
        backend = FakeBackend()

        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(base_dir=tmp)
            calibrate_m3(1, sampler, backend, store)

            record = store.load("m3", [0])
            self.assertIsNotNone(record)
            cals = record.single_qubit_cals()
            self.assertEqual(len(cals), 1)
            self.assertAlmostEqual(cals[0][0, 0], 1.0)
            self.assertAlmostEqual(cals[0][1, 1], 1.0)
            self.assertIsInstance(record.calibration_wall_clock_seconds, float)
            self.assertGreaterEqual(record.calibration_wall_clock_seconds, 0.0)

            expected_layout = physical_layout_for_width(1)
            self.assertEqual(len(sampler.transpile_options_calls), 2)
            for call in sampler.transpile_options_calls:
                self.assertEqual(call.get("initial_layout"), expected_layout)
                self.assertEqual(call.get("optimization_level"), 3)




class TestCompareM3Calibrations(unittest.TestCase):
    def test_reports_max_change_per_qubit(self):
        import numpy as np
        from piastq_execution.mitigation import make_m3_calibration_record

        a = make_m3_calibration_record([np.array([[0.99, 0.02], [0.01, 0.98]])], [0], "b", "v", 200, 1.0)
        b = make_m3_calibration_record([np.array([[0.97, 0.02], [0.03, 0.98]])], [0], "b", "v", 200, 1.0)
        report = compare_m3_calibrations(a, b)
        self.assertAlmostEqual(report["max_abs_change"], 0.02)
        self.assertEqual(len(report["max_abs_change_per_qubit"]), 1)
        self.assertEqual(report["width"], 1)


if __name__ == "__main__":
    unittest.main()
