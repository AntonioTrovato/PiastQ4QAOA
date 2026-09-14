"""Synthetic unit tests for piastq_execution.qubit_layout.

No AQT/backend involved -- this module only reads YAML and slices lists.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.qubit_layout import load_physical_qubits, physical_layout_for_width


class TestLoadPhysicalQubits(unittest.TestCase):
    def test_defaults_to_0_through_6_when_config_missing(self):
        missing_path = os.path.join(tempfile.mkdtemp(), "does_not_exist.yaml")
        self.assertEqual(load_physical_qubits(missing_path), [0, 1, 2, 3, 4, 5, 6])

    def test_reads_configured_physical_qubits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "backend.yaml")
            with open(path, "w") as f:
                f.write("physical_qubits: [3, 4, 5, 6, 7, 8, 9]\n")

            self.assertEqual(load_physical_qubits(path), [3, 4, 5, 6, 7, 8, 9])

    def test_defaults_when_key_absent_but_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "backend.yaml")
            with open(path, "w") as f:
                f.write("api_token: ACCESS_TOKEN\nbackend_name: offline_simulator_no_noise\n")

            self.assertEqual(load_physical_qubits(path), [0, 1, 2, 3, 4, 5, 6])


class TestPhysicalLayoutForWidth(unittest.TestCase):
    def test_returns_prefix_slice_of_given_qubits(self):
        qubits = [10, 11, 12, 13, 14, 15, 16]
        self.assertEqual(physical_layout_for_width(1, qubits), [10])
        self.assertEqual(physical_layout_for_width(3, qubits), [10, 11, 12])
        self.assertEqual(physical_layout_for_width(7, qubits), qubits)

    def test_same_width_always_returns_the_same_slice(self):
        """The whole point: calibration and real execution must agree, so
        calling this twice for the same width must be identical every time."""
        qubits = [0, 1, 2, 3, 4, 5, 6]
        self.assertEqual(physical_layout_for_width(4, qubits), physical_layout_for_width(4, qubits))

    def test_raises_when_width_exceeds_configured_qubits(self):
        with self.assertRaises(ValueError):
            physical_layout_for_width(4, [0, 1, 2])

    def test_uses_configured_qubits_when_none_given_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "backend.yaml")
            with open(path, "w") as f:
                f.write("physical_qubits: [20, 21, 22]\n")

            configured = load_physical_qubits(path)
            self.assertEqual(physical_layout_for_width(2, configured), [20, 21])


if __name__ == "__main__":
    unittest.main()
