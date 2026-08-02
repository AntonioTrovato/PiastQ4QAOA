"""Synthetic unit tests for piastq_execution.qubo_io.

Builds tiny real QuadraticProgram objects (qiskit_optimization) -- local,
classical construction only, no circuits/backend/hardware involved.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.qubo_io import json_dict_to_cluster_qubo, qubo_to_json_dict


class TestQuboToJsonDict(unittest.TestCase):
    def test_linear_only_qubo(self):
        from qiskit_optimization import QuadraticProgram

        qubo = QuadraticProgram()
        for i in range(3):
            qubo.binary_var(f"x{i}")
        qubo.minimize(linear=[1.5, -2.0, 0.0])

        d = qubo_to_json_dict(qubo)
        self.assertEqual(d["num_qubits"], 3)
        self.assertEqual(d["linear"], [1.5, -2.0, 0.0])
        self.assertEqual(d["quadratic"], [])

    def test_linear_and_quadratic_qubo(self):
        from qiskit_optimization import QuadraticProgram

        qubo = QuadraticProgram()
        for i in range(3):
            qubo.binary_var(f"x{i}")
        qubo.minimize(linear=[1.0, 2.0, 3.0], quadratic={(0, 1): 0.5, (1, 2): -0.25})

        d = qubo_to_json_dict(qubo)
        self.assertEqual(d["num_qubits"], 3)
        self.assertEqual(d["linear"], [1.0, 2.0, 3.0])
        triples = {(i, j): c for i, j, c in d["quadratic"]}
        self.assertAlmostEqual(triples[(0, 1)], 0.5)
        self.assertAlmostEqual(triples[(1, 2)], -0.25)

    def test_sparse_linear_terms_fill_zeros(self):
        # qiskit's linear.to_dict() only reports nonzero coefficients --
        # qubo_to_json_dict() must still produce one entry per variable.
        from qiskit_optimization import QuadraticProgram

        qubo = QuadraticProgram()
        for i in range(4):
            qubo.binary_var(f"x{i}")
        qubo.minimize(linear=[0.0, 7.0, 0.0, -3.0])

        d = qubo_to_json_dict(qubo)
        self.assertEqual(d["linear"], [0.0, 7.0, 0.0, -3.0])


class TestRoundTrip(unittest.TestCase):
    def test_round_trips_through_json_shape(self):
        from qiskit_optimization import QuadraticProgram

        qubo = QuadraticProgram()
        for i in range(3):
            qubo.binary_var(f"x{i}")
        qubo.minimize(linear=[1.0, -1.0, 2.0], quadratic={(0, 2): 3.0})

        d = qubo_to_json_dict(qubo)
        linear, quadratic, num_qubits = json_dict_to_cluster_qubo(d)

        self.assertEqual(num_qubits, 3)
        self.assertEqual(linear, [1.0, -1.0, 2.0])
        self.assertEqual(quadratic, {(0, 2): 3.0})

    def test_round_trip_matches_qubo_energy(self):
        # End-to-end: reconstructed (linear, quadratic, num_qubits) must
        # feed piastq_execution.evaluation.qubo_energy() identically to the
        # original QuadraticProgram's own objective.evaluate().
        from qiskit_optimization import QuadraticProgram

        from piastq_execution.evaluation import qubo_energy

        qubo = QuadraticProgram()
        for i in range(3):
            qubo.binary_var(f"x{i}")
        qubo.minimize(linear=[1.0, -2.0, 0.5], quadratic={(0, 1): 2.0})

        bitstring = [1, 0, 1]
        expected = qubo.objective.evaluate(bitstring)

        d = qubo_to_json_dict(qubo)
        linear, quadratic, _num_qubits = json_dict_to_cluster_qubo(d)
        actual = qubo_energy(bitstring, linear, quadratic)

        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
