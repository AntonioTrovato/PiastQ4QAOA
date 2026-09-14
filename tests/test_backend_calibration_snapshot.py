"""Synthetic unit tests for piastq_execution.backend_calibration_snapshot.

Uses fake backend objects (no AQT/backend call at all) to verify the
snapshot is extracted defensively (missing/partial data becomes None, never
an exception) and written as its own timestamped file.
"""

import glob
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.backend_calibration_snapshot import (
    save_backend_calibration_snapshot,
    snapshot_backend_calibration,
)


class FakeQubitProperties:
    def __init__(self, t1, t2, frequency):
        self.t1 = t1
        self.t2 = t2
        self.frequency = frequency


class FakeBackend:
    name = "fake_piastq"
    backend_version = "1.2.3"
    num_qubits = 3

    def __init__(self, qubit_properties):
        self._qubit_properties = qubit_properties

    @property
    def qubit_properties(self):
        return self._qubit_properties


class FakeBackendNoCalibrationData:
    """Stands in for a backend that reports nothing beyond a qubit count --
    every field must degrade to None, not raise."""

    name = "fake_bare"
    backend_version = "0.0.0"
    num_qubits = 2


class TestSnapshotBackendCalibration(unittest.TestCase):
    def test_extracts_t1_t2_frequency_per_qubit(self):
        backend = FakeBackend([
            FakeQubitProperties(t1=100e-6, t2=80e-6, frequency=5.1e9),
            FakeQubitProperties(t1=None, t2=90e-6, frequency=None),
            FakeQubitProperties(t1=110e-6, t2=None, frequency=5.3e9),
        ])

        snapshot = snapshot_backend_calibration(backend)

        self.assertEqual(snapshot["backend_name"], "fake_piastq")
        self.assertEqual(snapshot["backend_version"], "1.2.3")
        self.assertEqual(snapshot["num_qubits"], 3)
        self.assertIsInstance(snapshot["timestamp"], str)
        self.assertEqual(len(snapshot["qubits"]), 3)

        self.assertEqual(snapshot["qubits"][0], {"qubit_index": 0, "t1": 100e-6, "t2": 80e-6, "frequency": 5.1e9})
        self.assertEqual(snapshot["qubits"][1], {"qubit_index": 1, "t1": None, "t2": 90e-6, "frequency": None})
        self.assertEqual(snapshot["qubits"][2], {"qubit_index": 2, "t1": 110e-6, "t2": None, "frequency": 5.3e9})

    def test_degrades_gracefully_when_backend_reports_nothing(self):
        """A provider that doesn't populate qubit_properties at all must still
        produce one entry per qubit, every field None -- never raise."""
        backend = FakeBackendNoCalibrationData()

        snapshot = snapshot_backend_calibration(backend)

        self.assertEqual(snapshot["num_qubits"], 2)
        self.assertEqual(snapshot["qubits"], [
            {"qubit_index": 0, "t1": None, "t2": None, "frequency": None},
            {"qubit_index": 1, "t1": None, "t2": None, "frequency": None},
        ])

    def test_handles_qubit_properties_as_a_callable(self):
        class CallableBackend:
            name = "fake_callable"
            backend_version = "0.0.0"
            num_qubits = 1

            def qubit_properties(self):
                return [FakeQubitProperties(t1=1.0, t2=2.0, frequency=3.0)]

        snapshot = snapshot_backend_calibration(CallableBackend())
        self.assertEqual(snapshot["qubits"], [{"qubit_index": 0, "t1": 1.0, "t2": 2.0, "frequency": 3.0}])


class TestSaveBackendCalibrationSnapshot(unittest.TestCase):
    def test_writes_a_new_timestamped_file_per_call(self):
        backend = FakeBackendNoCalibrationData()

        with tempfile.TemporaryDirectory() as tmp:
            path1 = save_backend_calibration_snapshot(backend, output_dir=tmp)
            path2 = save_backend_calibration_snapshot(backend, output_dir=tmp)

            self.assertNotEqual(path1, path2, "each call must produce its own dated snapshot, never overwrite")
            self.assertTrue(os.path.exists(path1))
            self.assertTrue(os.path.exists(path2))

            files = glob.glob(os.path.join(tmp, "*.json"))
            self.assertEqual(len(files), 2)

            with open(path1) as f:
                contents = json.load(f)
            self.assertEqual(contents["backend_name"], "fake_bare")
            self.assertEqual(contents["num_qubits"], 2)

    def test_filename_contains_backend_name(self):
        backend = FakeBackendNoCalibrationData()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backend_calibration_snapshot(backend, output_dir=tmp)
            self.assertIn("fake_bare", os.path.basename(path))


if __name__ == "__main__":
    unittest.main()
