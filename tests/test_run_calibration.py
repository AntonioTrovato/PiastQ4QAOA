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
from piastq_execution.run_calibration import calibrate_m3, calibrate_mem


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
    (0,0) then (0,1)."""

    def __init__(self, probabilities_per_call):
        self.options = FakeSamplerOptions()
        self._responses = list(probabilities_per_call)

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


if __name__ == "__main__":
    unittest.main()
