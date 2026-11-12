"""Synthetic unit tests for piastq_execution.ideal_simulation.

Tiny hand-built circuits and QUBOs only; no repository data, no backend.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hermetic_env  # noqa: F401,E402

from qiskit import QuantumCircuit  # noqa: E402

from piastq_execution.ideal_simulation import (  # noqa: E402
    SHOT_CONFIGS,
    _argmax_optimal_fraction,
    compare_with_shot_matched,
    exact_distribution,
    sample_counts,
    simulate_combo,
    stable_seed,
)


def _x_on_qubit0():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.measure_all()
    return qc


def _hadamard():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    return qc


class SeedTests(unittest.TestCase):
    def test_seed_is_deterministic_and_distinguishes_inputs(self):
        self.assertEqual(stable_seed("gzip", "ideal_shots200", 3), stable_seed("gzip", "ideal_shots200", 3))
        self.assertNotEqual(stable_seed("gzip", "ideal_shots200", 3), stable_seed("gzip", "ideal_shots200", 4))
        self.assertLess(stable_seed("a"), 2 ** 32)


class SimulationTests(unittest.TestCase):
    def test_exact_distribution_uses_the_measurement_bit_order(self):
        self.assertEqual(exact_distribution(_x_on_qubit0()), {"01": 1.0})
        dist = exact_distribution(_hadamard())
        self.assertAlmostEqual(dist["0"], 0.5)
        self.assertAlmostEqual(dist["1"], 0.5)

    def test_sampled_counts_match_exact_bit_order_and_shot_budget(self):
        counts = sample_counts([_x_on_qubit0(), _hadamard()], shots=200, seed=7)
        self.assertEqual(counts[0], {"01": 200})
        self.assertEqual(sum(counts[1].values()), 200)

    def test_sampling_is_reproducible_for_a_seed(self):
        a = sample_counts([_hadamard()], shots=200, seed=11)
        b = sample_counts([_hadamard()], shots=200, seed=11)
        self.assertEqual(a, b)

    def test_simulate_combo_writes_every_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            simulated = simulate_combo(root, "toy", repetitions=2, circuits=[_x_on_qubit0(), _hadamard()])
            self.assertEqual(set(simulated), set(SHOT_CONFIGS))
            self.assertEqual(simulated["ideal_exact"][1][0], {"01": 1.0})
            self.assertEqual(sum(simulated["ideal_shots600"][2][1].values()), 600)
            for label in SHOT_CONFIGS:
                path = os.path.join(root, "results", "ideal_simulation", "toy", f"toy-rep-1-{label}-counts.jsonl")
                with open(path) as f:
                    self.assertEqual(len(f.readlines()), 4)  # 2 circuits x 2 repetitions


class EvaluationHelperTests(unittest.TestCase):
    def test_argmax_optimal_fraction(self):
        # one-variable QUBOs: linear -1 -> optimum "1"; linear +1 -> optimum "0"
        qubos = [([-1.0], {}, 1), ([1.0], {}, 1)]
        self.assertEqual(_argmax_optimal_fraction(["1", "0"], qubos), 1.0)
        self.assertEqual(_argmax_optimal_fraction(["1", "1"], qubos), 0.5)

    def test_hardware_methods_are_compared_with_their_shot_matched_reference(self):
        results = {
            "raw": [{"iteration_id": i, "qubo_energy": -1.0 - 0.01 * i} for i in range(1, 11)],
            "trex": [{"iteration_id": i, "qubo_energy": -1.2 - 0.01 * i} for i in range(1, 11)],
            "ideal_shots200": [{"iteration_id": i, "qubo_energy": -1.0 - 0.011 * i} for i in range(1, 11)],
            "ideal_shots600": [{"iteration_id": i, "qubo_energy": -1.2 - 0.012 * i} for i in range(1, 11)],
            "ideal_exact": [{"iteration_id": i, "qubo_energy": -1.5} for i in range(1, 11)],
        }
        out = compare_with_shot_matched(results)
        self.assertIn("raw vs ideal_shots200", out["tests"])
        self.assertIn("trex vs ideal_shots600", out["tests"])
        self.assertNotIn("mem vs ideal_shots200", out["tests"])
        self.assertEqual(out["descriptive"]["ideal_exact"]["qubo_energy"]["sd"], 0.0)


if __name__ == "__main__":
    unittest.main()
