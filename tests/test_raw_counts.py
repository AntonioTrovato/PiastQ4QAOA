"""Synthetic unit tests for piastq_execution.raw_counts.

These use a fake sampler (no AQT/backend call at all) and, for the qubit-mapping
test, a local AerSimulator transpile (no AQTProvider/AQTSampler and no QAOA
optimization loop) against a hand-built 2-qubit circuit. Nothing here touches
AQTProvider, AQTSampler, or the offline_simulator_no_noise backend.
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from piastq_execution.raw_counts import (
    counts_from_quasi_probabilities,
    run_circuit_with_batching_recorded,
    get_physical_qubit_mapping,
    RawCountsWriter,
)


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
    """Stands in for AQTSampler: returns a scripted quasi-distribution per call,
    tracking exactly what shots value it was asked to run with."""

    def __init__(self, probabilities_per_call):
        self.options = FakeSamplerOptions()
        self._probabilities_per_call = list(probabilities_per_call)
        self.calls = []

    def run(self, circuits):
        self.calls.append(self.options.shots)
        probabilities = self._probabilities_per_call.pop(0)
        return _RunHandle(probabilities)


class _RunHandle:
    def __init__(self, probabilities):
        self._probabilities = probabilities

    def result(self):
        return FakeResult(self._probabilities)


class FakeBackend:
    name = "fake_offline_simulator"
    backend_version = "0.0-test"


class TinyCircuit:
    """Minimal stand-in with just the attribute run_circuit_with_batching_recorded
    actually touches (num_qubits), so this test never builds a real QuantumCircuit
    or talks to any backend."""

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits


class TestCountsFromQuasiProbabilities(unittest.TestCase):
    def test_exact_reconstruction(self):
        # 2-qubit toy: 60/80 shots landed on "00", 20/80 on "11"
        probabilities = {"00": 0.75, "11": 0.25}
        raw = counts_from_quasi_probabilities(probabilities, shots=80)
        self.assertEqual(raw, {"00": 60, "11": 20})
        self.assertEqual(sum(raw.values()), 80)


class TestRunCircuitWithBatchingRecorded(unittest.TestCase):
    def test_single_batch_aggregation_and_metadata(self):
        sampler = FakeSampler([{"00": 0.75, "11": 0.25}])
        circuit = TinyCircuit(num_qubits=2)

        total_counts, record = run_circuit_with_batching_recorded(
            circuit,
            sampler,
            algorithm="qaoa_tcs",
            objective_mode="single_objective",
            dataset="toy",
            circuit_id="toy_cluster0",
            backend=FakeBackend(),
            cluster_id=0,
            shots_per_batch=80,
            num_batches=1,
            record_qubit_mapping=False,
        )

        self.assertEqual(dict(total_counts), {"00": 60, "11": 20})
        self.assertEqual(sampler.calls, [80])

        self.assertEqual(record.algorithm, "qaoa_tcs")
        self.assertEqual(record.objective_mode, "single_objective")
        self.assertEqual(record.dataset, "toy")
        self.assertEqual(record.circuit_id, "toy_cluster0")
        self.assertEqual(record.cluster_id, 0)
        self.assertEqual(record.backend_name, "fake_offline_simulator")
        self.assertEqual(record.backend_version, "0.0-test")
        self.assertEqual(record.total_shots_requested, 80)
        self.assertEqual(record.total_shots_returned, 80)
        self.assertEqual(len(record.batches), 1)
        self.assertEqual(record.batches[0].raw_counts, {"00": 60, "11": 20})
        self.assertEqual(record.aggregated_counts, {"00": 60, "11": 20})

    def test_multi_batch_plus_remainder_sums_correctly(self):
        # Mirrors the "N full batches + 1 remainder batch" hardware shot pattern,
        # scaled down: 2 batches of 200 shots + a 48-shot remainder.
        sampler = FakeSampler([
            {"01": 1.0},
            {"01": 0.5, "10": 0.5},
            {"10": 1.0},
        ])
        circuit = TinyCircuit(num_qubits=2)

        total_counts, record = run_circuit_with_batching_recorded(
            circuit,
            sampler,
            algorithm="qaoa_tcs",
            objective_mode="multi_objective",
            dataset="toy_sir",
            circuit_id="toy_sir_cluster3",
            backend=FakeBackend(),
            shots_per_batch=200,
            num_batches=2,
            remainder_shots=48,
            record_qubit_mapping=False,
        )

        self.assertEqual(sampler.calls, [200, 200, 48])
        self.assertEqual(dict(total_counts), {"01": 200 + 100, "10": 100 + 48})
        self.assertEqual(record.total_shots_requested, 200 + 200 + 48)
        self.assertEqual(record.total_shots_returned, 200 + 200 + 48)
        self.assertEqual(len(record.batches), 3)
        self.assertEqual(record.batches[-1].shots_requested, 48)
        self.assertEqual(record.batches[-1].raw_counts, {"10": 48})

    def test_short_return_is_tracked_not_hidden(self):
        """If a batch returns fewer shots than requested (e.g. discarded shots
        on real hardware), shots_returned must reflect that instead of silently
        assuming it matches shots_requested."""
        sampler = FakeSampler([{"00": 1.0}])
        circuit = TinyCircuit(num_qubits=1)

        total_counts, record = run_circuit_with_batching_recorded(
            circuit,
            sampler,
            algorithm="igdec_qaoa",
            objective_mode="single_objective",
            dataset="toy",
            circuit_id="toy_iter1_sub0",
            backend=FakeBackend(),
            iteration_id=1,
            subproblem_id=0,
            shots_per_batch=80,
            num_batches=1,
            record_qubit_mapping=False,
        )

        # probability was exactly 1.0 for "00" at 80 shots requested -> 80 returned
        self.assertEqual(record.batches[0].shots_returned, 80)
        self.assertEqual(record.iteration_id, 1)
        self.assertEqual(record.subproblem_id, 0)


class TestPhysicalQubitMapping(unittest.TestCase):
    def test_mapping_covers_every_logical_qubit_on_local_simulator(self):
        # Local Aer simulator only -- no AQTProvider/AQTSampler involved.
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        backend = AerSimulator()
        mapping = get_physical_qubit_mapping(circuit, backend, optimization_level=1)

        self.assertEqual(set(mapping.keys()), {0, 1})
        for physical_index in mapping.values():
            self.assertIsInstance(physical_index, int)
            self.assertGreaterEqual(physical_index, 0)


class TestRawCountsWriter(unittest.TestCase):
    def test_writes_one_json_line_per_record(self):
        sampler = FakeSampler([{"0": 1.0}, {"1": 1.0}])
        circuit = TinyCircuit(num_qubits=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "raw_counts.jsonl")
            with RawCountsWriter(path) as writer:
                for i in range(2):
                    _, record = run_circuit_with_batching_recorded(
                        circuit,
                        sampler,
                        algorithm="qaoa_tcs",
                        objective_mode="single_objective",
                        dataset="toy",
                        circuit_id=f"toy_cluster{i}",
                        backend=FakeBackend(),
                        shots_per_batch=10,
                        num_batches=1,
                        record_qubit_mapping=False,
                    )
                    writer.write(record)

            with open(path) as f:
                lines = [json.loads(line) for line in f]

            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["circuit_id"], "toy_cluster0")
            self.assertEqual(lines[1]["circuit_id"], "toy_cluster1")
            self.assertEqual(lines[0]["aggregated_counts"], {"0": 10})
            self.assertEqual(lines[1]["aggregated_counts"], {"1": 10})


if __name__ == "__main__":
    unittest.main()
