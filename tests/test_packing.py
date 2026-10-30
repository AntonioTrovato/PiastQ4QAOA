"""Synthetic tests for piastq_execution.packing and the packed-job runner.

The key test packs real circuits, simulates the packed job exactly with Aer,
and checks that every member's split counts equal that circuit simulated on
its own -- any bit-order or offset mistake would show up there."""

import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hermetic_env  # noqa: F401,E402  (scrubs PIASTQ_* variables)

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler as AerSampler

from piastq_execution.packing import (
    DEFAULT_PACK_SLOTS,
    build_packed_circuit,
    count_jobs,
    marginalize_counts,
    plan_packs,
    slot_qubits,
)
from piastq_execution.raw_counts import run_circuit_with_batching_recorded, run_packed_circuits

SLOTS = DEFAULT_PACK_SLOTS  # [(0,7),(7,7),(14,6)]


def _rotated_circuit(width, seed):
    """A product circuit with distinct angles per qubit, so a qubit mix-up
    changes the distribution."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(width)
    for q in range(width):
        qc.ry(float(rng.uniform(0.2, 2.9)), q)
    return qc


class FakeBackend:
    name = "fake"
    backend_version = "0"


class ExactSampler:
    """Exact (shot-noise-free) simulation of whatever circuit it is given."""

    def __init__(self):
        self.options = type("O", (), {"shots": None})()
        self._sampler = AerSampler()
        self._sampler.options.shots = None

    def set_transpile_options(self, **kw):
        pass

    def run(self, circuits):
        return self._sampler.run(circuits)


class _Dist:
    def __init__(self, probabilities):
        self._p = probabilities

    def binary_probabilities(self):
        return self._p


class _Result:
    def __init__(self, probabilities):
        self.quasi_dists = [_Dist(probabilities)]


class _Handle:
    def __init__(self, probabilities):
        self._p = probabilities

    def result(self):
        return _Result(self._p)

    def job_id(self):
        return "j"


class ShotSampler(ExactSampler):
    """Like real hardware: draws `options.shots` integer shots from the exact
    distribution of the circuit it is given and reports counts / shots."""

    seed = 7

    def run(self, circuits):
        exact = self._sampler.run(circuits).result().quasi_dists[0].binary_probabilities()
        keys = sorted(exact)
        probs = np.array([max(exact[k], 0.0) for k in keys])
        shots = self.options.shots
        drawn = np.random.default_rng(self.seed).multinomial(shots, probs / probs.sum())
        return _Handle({k: int(c) / shots for k, c in zip(keys, drawn) if c})


class TestPlanPacks(unittest.TestCase):
    def test_no_overlap_and_inside_slots(self):
        widths = [7, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 1, 1, 1, 6, 2]
        packs = plan_packs(widths, SLOTS)
        seen = set()
        for pack in packs:
            used = set()
            for p in pack:
                qubits = set(range(p.start, p.start + p.width))
                self.assertFalse(used & qubits)
                used |= qubits
                slot = next(s for s in SLOTS if s[0] <= p.start and p.start + p.width <= s[0] + s[1])
                self.assertIsNotNone(slot)
                seen.add(p.member_index)
            offsets = sorted((p.offset, p.width) for p in pack)
            running = 0
            for offset, width in offsets:                 # combined offsets are contiguous
                self.assertEqual(offset, running)
                running += width
        self.assertEqual(seen, set(range(len(widths))))   # every circuit placed exactly once

    def test_deterministic(self):
        widths = [3, 7, 2, 5, 5, 1, 4]
        self.assertEqual(plan_packs(widths, SLOTS), plan_packs(widths, SLOTS))

    def test_packing_reduces_job_count(self):
        widths = [1] * 20
        self.assertEqual(count_jobs(widths, SLOTS), 1)
        self.assertEqual(count_jobs([7, 7, 7], SLOTS), 2)   # only two 7-wide slots
        self.assertEqual(count_jobs([], SLOTS), 0)

    def test_too_wide_circuit_rejected(self):
        with self.assertRaises(ValueError):
            plan_packs([8], SLOTS)

    def test_slot_qubits(self):
        self.assertEqual(slot_qubits(SLOTS), list(range(20)))


class TestMarginalizeCounts(unittest.TestCase):
    def test_extracts_the_right_clbits(self):
        # joint string "c3 c2 c1 c0": clbits 0..1 = "10", clbits 2..3 = "01"
        counts = {"0110": 7, "0111": 3, "1010": 5}
        self.assertEqual(marginalize_counts(counts, 0, 2), {"10": 12, "11": 3})
        self.assertEqual(marginalize_counts(counts, 2, 2), {"01": 10, "10": 5})


class TestPackedExecutionMatchesSeparateExecution(unittest.TestCase):
    def test_split_counts_match_each_circuit_alone(self):
        circuits = [_rotated_circuit(w, seed) for seed, w in enumerate([3, 2, 1, 3, 4, 2, 5])]
        specs = [dict(circuit=c, algorithm="a", objective_mode="m", dataset="d", circuit_id=f"c{i}",
                      iteration_id=1, backend=FakeBackend(), shots_per_batch=200, num_batches=1,
                      remainder_shots=0, completed=None) for i, c in enumerate(circuits)]
        written = []
        results = run_packed_circuits(specs, ShotSampler, on_record=written.append, slots=SLOTS, label="t")
        self.assertEqual(len(written), len(circuits))
        for i, circuit in enumerate(circuits):
            measured = circuit.copy()
            measured.measure_all()
            exact = ExactSampler()._sampler.run([measured]).result().quasi_dists[0].binary_probabilities()
            packed, record = results[i]
            self.assertEqual(sum(packed.values()), 200)
            for key in set(exact) | set(packed):               # every outcome within 5 sigma of the exact value
                p = exact.get(key, 0.0)
                sigma = (200 * p * (1 - p)) ** 0.5
                self.assertLessEqual(abs(packed.get(key, 0) - 200 * p), 5 * sigma + 2, (i, key, p))
            self.assertEqual(len(next(iter(packed))), circuit.num_qubits)
            self.assertEqual(record.circuit_id, f"c{i}")
            mapping = record.physical_qubit_mapping
            self.assertEqual(sorted(mapping), list(range(circuit.num_qubits)))
            self.assertEqual(list(mapping.values()), list(range(mapping[0], mapping[0] + circuit.num_qubits)))
            self.assertGreaterEqual(record.batches[0].job_metadata["pack"]["members"], 1)

    def test_a_bit_order_mistake_would_be_caught(self):
        """Sanity check of the test itself: reversing the member's bits no longer matches."""
        circuit = _rotated_circuit(3, 0)
        spec = dict(circuit=circuit, algorithm="a", objective_mode="m", dataset="d", circuit_id="c",
                    iteration_id=1, backend=FakeBackend(), shots_per_batch=200, num_batches=1,
                    remainder_shots=0, completed=None)
        (packed, _), = run_packed_circuits([spec], ShotSampler, slots=SLOTS, label="t")
        measured = circuit.copy()
        measured.measure_all()
        exact = ExactSampler()._sampler.run([measured]).result().quasi_dists[0].binary_probabilities()
        reversed_error = sum(abs(packed.get(k[::-1], 0) - 200 * p) for k, p in exact.items())
        straight_error = sum(abs(packed.get(k, 0) - 200 * p) for k, p in exact.items())
        self.assertLess(straight_error, reversed_error)

    def test_records_share_the_jobs_time_by_qubit_share(self):
        from unittest import mock
        specs = [dict(circuit=_rotated_circuit(w, w), algorithm="a", objective_mode="m", dataset="d",
                      circuit_id=f"c{w}", iteration_id=1, backend=FakeBackend(), shots_per_batch=200,
                      num_batches=1, remainder_shots=0) for w in (3, 1)]
        with mock.patch.dict(os.environ, {"PIASTQ_BACKEND_NAME": "piast_q", "PIASTQ_QPU_LIMIT_SECONDS": ""}):
            results = run_packed_circuits(specs, ShotSampler, slots=SLOTS, label="t")
        billed = {r.circuit_id: r.total_billed_seconds for _, r in results}
        self.assertAlmostEqual(billed["c3"] + billed["c1"], 18.4974, places=4)   # one job, shared
        self.assertAlmostEqual(billed["c3"], 3 * billed["c1"], places=6)

    def test_resume_skips_fully_completed_jobs_and_never_rewrites(self):
        circuits = [_rotated_circuit(2, 1), _rotated_circuit(3, 2)]
        done = {(f"c{i}", 1): {"aggregated_counts": {"0" * c.num_qubits: 200}} for i, c in enumerate(circuits)}
        specs = [dict(circuit=c, algorithm="a", objective_mode="m", dataset="d", circuit_id=f"c{i}",
                      iteration_id=1, backend=FakeBackend(), shots_per_batch=200, num_batches=1,
                      remainder_shots=0, completed=done) for i, c in enumerate(circuits)]

        class Exploding(ExactSampler):
            def run(self, circuits):
                raise AssertionError("must not run")

        written = []
        results = run_packed_circuits(specs, Exploding, on_record=written.append, slots=SLOTS, label="t")
        self.assertEqual(written, [])
        self.assertEqual(dict(results[0][0]), {"00": 200})

    def test_partially_completed_job_writes_only_the_missing_circuits(self):
        circuits = [_rotated_circuit(2, 1), _rotated_circuit(3, 2)]
        done = {("c0", 1): {"aggregated_counts": {"00": 200}}}
        specs = [dict(circuit=c, algorithm="a", objective_mode="m", dataset="d", circuit_id=f"c{i}",
                      iteration_id=1, backend=FakeBackend(), shots_per_batch=200, num_batches=1,
                      remainder_shots=0, completed=done) for i, c in enumerate(circuits)]
        written = []
        results = run_packed_circuits(specs, ShotSampler, on_record=written.append, slots=SLOTS, label="t")
        self.assertEqual([r.circuit_id for r in written], ["c1"])
        self.assertEqual(dict(results[0][0]), {"00": 200})     # the saved result is kept


class TestBuildPackedCircuit(unittest.TestCase):
    def test_layout_and_drop_of_member_measurements(self):
        a = _rotated_circuit(2, 1)
        a.measure_all()                                         # members may carry their own measurements
        b = _rotated_circuit(1, 2)
        pack = plan_packs([2, 1], [(0, 3), (5, 2)])[0]
        combined, layout = build_packed_circuit([a, b], pack)
        self.assertEqual(combined.num_qubits, 3)
        self.assertEqual(sorted(layout), layout)
        self.assertEqual(sum(1 for inst in combined.data if inst.operation.name == "measure"), 3)


if __name__ == "__main__":
    unittest.main()
