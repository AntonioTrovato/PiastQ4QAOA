"""Synthetic tests for the concurrent executor, the failed-job retry and the
billed-time recording (piastq_execution.raw_counts / qpu_usage / evaluation).
Fake samplers only: no network, no backend."""

import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hermetic_env  # noqa: F401,E402  (scrubs PIASTQ_* variables)

from qiskit import QuantumCircuit

from piastq_execution import qpu_usage
from piastq_execution.evaluation import compute_execution_time_seconds
from piastq_execution.raw_counts import (
    max_jobs_in_flight,
    run_circuit_with_batching_recorded,
    run_circuits_concurrently,
)


class _Dist:
    def binary_probabilities(self):
        return {"0": 1.0}


class _Result:
    quasi_dists = [_Dist()]


class _Options:
    shots = None


class SlowSampler:
    """Records how many sampler.run() calls overlap in time."""
    active = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self):
        self.options = _Options()

    def set_transpile_options(self, **kw):
        pass

    def run(self, circuits):
        cls = SlowSampler
        with cls.lock:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
        time.sleep(0.05)
        with cls.lock:
            cls.active -= 1
        return _Handle()


class _Handle:
    def result(self):
        return _Result()

    def job_id(self):
        return "j"


class FlakySampler:
    """First `failures` runs end as 'failed' like the PCSS service does."""

    def __init__(self, failures):
        self.options = _Options()
        self.failures = failures
        self.calls = 0

    def set_transpile_options(self, **kw):
        pass

    def run(self, circuits):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("Your task has status: failed")
        return _Handle()


def _circuit():
    qc = QuantumCircuit(1)
    qc.measure_all()
    return qc


def _spec(i, **extra):
    spec = dict(circuit=_circuit(), algorithm="a", objective_mode="m", dataset="d",
                circuit_id=f"c{i}", iteration_id=1, backend=object(), shots_per_batch=200, num_batches=1, record_qubit_mapping=False)
    spec.update(extra)
    return spec


class TestConcurrentExecutor(unittest.TestCase):
    def setUp(self):
        SlowSampler.active = 0
        SlowSampler.peak = 0

    def test_runs_in_parallel_and_keeps_spec_order(self):
        specs = [_spec(i) for i in range(8)]
        written = []
        results = run_circuits_concurrently(specs, SlowSampler, on_record=written.append, max_workers=4)
        self.assertEqual([r[1].circuit_id for r in results], [f"c{i}" for i in range(8)])
        self.assertEqual(len(written), 8)                 # every record written, in the calling thread
        self.assertGreater(SlowSampler.peak, 1)            # really overlapped
        self.assertLessEqual(SlowSampler.peak, 4)          # never more than max_workers

    def test_sequential_when_one_worker(self):
        results = run_circuits_concurrently([_spec(i) for i in range(3)], SlowSampler, max_workers=1)
        self.assertEqual(SlowSampler.peak, 1)
        self.assertEqual(len(results), 3)

    def test_resumed_circuits_are_not_run_or_written(self):
        completed = {("c0", 1): {"aggregated_counts": {"0": 200}}}
        written = []
        results = run_circuits_concurrently(
            [_spec(0, completed=completed), _spec(1, completed=completed)], SlowSampler,
            on_record=written.append, max_workers=2)
        self.assertIsNone(results[0][1])
        self.assertEqual([r.circuit_id for r in written], ["c1"])

    def test_error_cancels_pending_keeps_finished_and_reraises(self):
        calls = {"n": 0}

        class Exploding(SlowSampler):
            def run(self, circuits):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise qpu_usage.BudgetExceeded("limit")
                return super().run(circuits)

        written = []
        with self.assertRaises(qpu_usage.BudgetExceeded):
            run_circuits_concurrently([_spec(i) for i in range(20)], Exploding,
                                      on_record=written.append, max_workers=2)
        self.assertLess(calls["n"], 20)   # the not-yet-started ones were cancelled

    def test_max_jobs_in_flight_env(self):
        with mock.patch.dict(os.environ, {"PIASTQ_MAX_IN_FLIGHT": "12"}):
            self.assertEqual(max_jobs_in_flight(), 12)
        with mock.patch.dict(os.environ, {"PIASTQ_MAX_IN_FLIGHT": "garbage"}):
            self.assertEqual(max_jobs_in_flight(), 1)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PIASTQ_MAX_IN_FLIGHT", None)
            self.assertEqual(max_jobs_in_flight(), 1)


class TestFailedJobRetry(unittest.TestCase):
    def test_failed_job_is_resubmitted(self):
        sampler = FlakySampler(failures=2)
        with mock.patch("piastq_execution.raw_counts.time.sleep"):
            counts, record = run_circuit_with_batching_recorded(
                _circuit(), sampler, algorithm="a", objective_mode="m", dataset="d",
                circuit_id="c", backend=object(), shots_per_batch=200, num_batches=1, record_qubit_mapping=False)
        self.assertEqual(sampler.calls, 3)
        self.assertEqual(dict(counts), {"0": 200})

    def test_gives_up_after_three_attempts(self):
        sampler = FlakySampler(failures=99)
        with mock.patch("piastq_execution.raw_counts.time.sleep"):
            with self.assertRaises(RuntimeError):
                run_circuit_with_batching_recorded(
                    _circuit(), sampler, algorithm="a", objective_mode="m", dataset="d",
                    circuit_id="c", backend=object(), shots_per_batch=200, num_batches=1, record_qubit_mapping=False)
        self.assertEqual(sampler.calls, 3)

    def test_other_errors_are_not_retried(self):
        class Broken(FlakySampler):
            def run(self, circuits):
                self.calls += 1
                raise RuntimeError("something else")

        sampler = Broken(0)
        with self.assertRaises(RuntimeError):
            run_circuit_with_batching_recorded(
                _circuit(), sampler, algorithm="a", objective_mode="m", dataset="d",
                circuit_id="c", backend=object(), shots_per_batch=200, num_batches=1, record_qubit_mapping=False)
        self.assertEqual(sampler.calls, 1)


class TestBilledTime(unittest.TestCase):
    def test_model_matches_measured_billing(self):
        with mock.patch.dict(os.environ, {"PIASTQ_BACKEND_NAME": "piast_q", "PIASTQ_QPU_LIMIT_SECONDS": ""}):
            self.assertAlmostEqual(qpu_usage.modeled_billed_seconds(200), 18.4974, places=4)
            self.assertAlmostEqual(qpu_usage.modeled_billed_seconds(100), 10.0774, places=4)

    def test_none_on_simulators(self):
        with mock.patch.dict(os.environ, {"PIASTQ_BACKEND_NAME": "offline_simulator_no_noise"}):
            self.assertIsNone(qpu_usage.modeled_billed_seconds(200))

    def test_record_carries_billed_seconds_and_evaluation_prefers_it(self):
        with mock.patch.dict(os.environ, {"PIASTQ_BACKEND_NAME": "piast_q", "PIASTQ_QPU_LIMIT_SECONDS": ""}):
            _, record = run_circuit_with_batching_recorded(
                _circuit(), SlowSampler(), algorithm="a", objective_mode="m", dataset="d",
                circuit_id="c", backend=object(), shots_per_batch=200, num_batches=2, record_qubit_mapping=False)
        self.assertAlmostEqual(record.total_billed_seconds, 2 * 18.4974, places=4)
        # queue wait would make wall-clock huge; billed time is what is reported
        record.total_wall_clock_seconds = 900.0
        self.assertAlmostEqual(compute_execution_time_seconds([record]), 36.9948, places=4)

    def test_simulator_records_keep_wall_clock(self):
        with mock.patch.dict(os.environ, {"PIASTQ_BACKEND_NAME": "offline_simulator_no_noise"}):
            _, record = run_circuit_with_batching_recorded(
                _circuit(), SlowSampler(), algorithm="a", objective_mode="m", dataset="d",
                circuit_id="c", backend=object(), shots_per_batch=200, num_batches=1, record_qubit_mapping=False)
        self.assertIsNone(record.total_billed_seconds)
        self.assertEqual(compute_execution_time_seconds([record]), record.total_wall_clock_seconds)


class TestConcurrentCalibration(unittest.TestCase):
    def test_calibration_circuits_overlap_and_keep_their_keys(self):
        from piastq_execution.run_calibration import _run_many

        SlowSampler.active = 0
        SlowSampler.peak = 0
        items = [(f"k{i}", _circuit()) for i in range(10)]
        with mock.patch.dict(os.environ, {"PIASTQ_MAX_IN_FLIGHT": "5"}):
            counts, seconds = _run_many(items, SlowSampler(), SlowSampler, [0])
        self.assertEqual(sorted(counts), sorted(k for k, _ in items))
        self.assertGreater(SlowSampler.peak, 1)
        self.assertLessEqual(SlowSampler.peak, 5)

    def test_sequential_without_a_factory(self):
        from piastq_execution.run_calibration import _run_many

        SlowSampler.active = 0
        SlowSampler.peak = 0
        with mock.patch.dict(os.environ, {"PIASTQ_MAX_IN_FLIGHT": "5"}):
            _run_many([(i, _circuit()) for i in range(4)], SlowSampler(), None, [0])
        self.assertEqual(SlowSampler.peak, 1)


if __name__ == "__main__":
    unittest.main()
