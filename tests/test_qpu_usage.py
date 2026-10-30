"""Synthetic unit tests for the budget stop, resume and PIAST-Q backend wiring
(piastq_execution.qpu_usage, raw_counts resume helpers, backend_config's
pcss path). No network, no backend: the PCSS client is a fake."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hermetic_env  # noqa: F401,E402  (scrubs PIASTQ_* variables)

import httpx
from qiskit import QuantumCircuit

from piastq_execution import backend_config, qpu_usage
from piastq_execution.raw_counts import (
    load_completed_records,
    run_circuit_with_batching_recorded,
)


class FakeClient:
    def __init__(self, used_us):
        self.used_us = used_us

    def get_my_usage(self, verbose=True):
        return [
            {"device": {"name": "MOCK"}, "member_quota": None},
            {"device": {"name": "PIAST-Q"}, "member_quota": {"usage_limit": 9 * 10 ** 10, "used": self.used_us}},
        ]


class ExplodingSampler:
    options = type("O", (), {"shots": None})()

    def set_transpile_options(self, **kw):
        pass

    def run(self, circuits):
        raise AssertionError("the QPU must not be touched")


class TestBilledUsage(unittest.TestCase):
    def test_reads_member_quota_in_seconds(self):
        with mock.patch.object(backend_config, "get_openqapi_client", return_value=FakeClient(28_574_800)):
            self.assertAlmostEqual(qpu_usage.billed_seconds(), 28.5748)

    def test_none_without_a_client_offline_simulator(self):
        with mock.patch.object(backend_config, "get_openqapi_client", return_value=None):
            self.assertIsNone(qpu_usage.billed_seconds())

    def test_check_budget_noop_when_limit_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(qpu_usage.LIMIT_ENV, None)
            with mock.patch.object(backend_config, "get_openqapi_client", return_value=FakeClient(10 ** 12)):
                qpu_usage.check_budget()  # must not raise

    def test_check_budget_raises_before_crossing_the_limit(self):
        with mock.patch.dict(os.environ, {qpu_usage.LIMIT_ENV: "100"}):
            with mock.patch.object(backend_config, "get_openqapi_client", return_value=FakeClient(90_000_000)):
                with self.assertRaises(qpu_usage.BudgetExceeded):
                    qpu_usage.check_budget(next_batch_seconds=18.5)  # 90 + 18.5 > 100
            with mock.patch.object(backend_config, "get_openqapi_client", return_value=FakeClient(80_000_000)):
                qpu_usage.check_budget(next_batch_seconds=18.5)  # 80 + 18.5 <= 100

    def test_execution_stops_before_submitting_when_over_budget(self):
        circuit = QuantumCircuit(1)
        circuit.measure_all()
        with mock.patch.dict(os.environ, {qpu_usage.LIMIT_ENV: "10"}):
            with mock.patch.object(backend_config, "get_openqapi_client", return_value=FakeClient(9_999_000)):
                with self.assertRaises(qpu_usage.BudgetExceeded):
                    run_circuit_with_batching_recorded(
                        circuit, ExplodingSampler(), algorithm="a", objective_mode="m", dataset="d",
                        circuit_id="c", backend=object(), shots_per_batch=200, num_batches=1,
                    )


class TestResume(unittest.TestCase):
    def _write(self, path, records, truncated_tail=""):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.write(truncated_tail)

    def test_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x-raw_counts.jsonl")
            self._write(path, [{"circuit_id": "c0", "iteration_id": 1, "aggregated_counts": {"0": 200}}])
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PIASTQ_RESUME", None)
                self.assertEqual(load_completed_records(path), {})

    def test_loads_completed_and_ignores_half_written_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x-raw_counts.jsonl")
            self._write(
                path,
                [{"circuit_id": "c0", "iteration_id": 1, "aggregated_counts": {"01": 200}}],
                truncated_tail='{"circuit_id": "c1", "iteration_id": 1, "aggreg',
            )
            with mock.patch.dict(os.environ, {"PIASTQ_RESUME": "1"}):
                done = load_completed_records(path)
            self.assertEqual(list(done), [("c0", 1)])

    def test_half_written_last_line_is_removed_so_appends_stay_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x-raw_counts.jsonl")
            self._write(
                path,
                [{"circuit_id": "c0", "iteration_id": 1, "aggregated_counts": {"01": 200}}],
                truncated_tail='{"circuit_id": "c1", "iteration_id": 1, "aggreg',
            )
            with mock.patch.dict(os.environ, {"PIASTQ_RESUME": "1"}):
                load_completed_records(path)
            with open(path, "a") as f:  # what RawCountsWriter does next
                f.write(json.dumps({"circuit_id": "c1", "iteration_id": 1, "aggregated_counts": {}}) + "\n")
            with open(path) as f:
                rows = [json.loads(line) for line in f]  # every line must parse
            self.assertEqual([r["circuit_id"] for r in rows], ["c0", "c1"])

    def test_missing_file_is_empty(self):
        with mock.patch.dict(os.environ, {"PIASTQ_RESUME": "1"}):
            self.assertEqual(load_completed_records("/nonexistent/x.jsonl"), {})

    def test_completed_circuit_is_not_sent_to_the_qpu(self):
        circuit = QuantumCircuit(2)
        circuit.measure_all()
        completed = {("c0", 3): {"aggregated_counts": {"10": 150, "11": 50}}}
        counts, record = run_circuit_with_batching_recorded(
            circuit, ExplodingSampler(), algorithm="a", objective_mode="m", dataset="d",
            circuit_id="c0", iteration_id=3, backend=object(), completed=completed,
        )
        self.assertIsNone(record)
        self.assertEqual(dict(counts), {"10": 150, "11": 50})


class TestPcssBackend(unittest.TestCase):
    def test_backend_name_spellings(self):
        for name in ("piast_q", "PIAST-Q", "piastq", " Piast_Q "):
            self.assertTrue(backend_config.is_pcss_backend(name), name)
        for name in ("offline_simulator_no_noise", "ibex", ""):
            self.assertFalse(backend_config.is_pcss_backend(name), name)

    def test_no_client_for_other_backends(self):
        with mock.patch.dict(os.environ, {"PIASTQ_BACKEND_NAME": "offline_simulator_no_noise"}):
            self.assertIsNone(backend_config.get_openqapi_client())

    def test_get_backend_uses_pcss_for_piast_q(self):
        class FakeResource:
            def result(self, job_id, *, timeout=None):
                return "done"

            def submit(self, circuit):
                return "id"

        class FakeOpenQAPI:
            def __init__(self, token):
                self.token = token

            def piast_q_backend(self):
                return FakeResource()

        fake_module = mock.MagicMock(OpenQAPI=FakeOpenQAPI)
        with mock.patch.dict(sys.modules, {"pcss_qapi": fake_module}), \
                mock.patch.dict(os.environ, {"PIASTQ_BACKEND_NAME": "piast_q", "PIASTQ_API_TOKEN": "tok"}), \
                mock.patch.object(backend_config, "_openqapi_client", None):
            backend = backend_config.get_backend()
            self.assertEqual(backend.result("x"), "done")
            self.assertEqual(backend.submit(object()), "id")

    def test_hardened_result_survives_transient_network_errors(self):
        class Flaky:
            calls = 0

            def result(self, job_id, *, timeout=None):
                print("[[0, 0]]")  # the vendor wrapper prints every shot
                Flaky.calls += 1
                if Flaky.calls < 3:
                    raise httpx.ConnectTimeout("handshake timed out")
                return "result"

            def submit(self, circuit):
                return "id"

        with mock.patch.object(backend_config, "_RETRY_SECONDS", 0):
            backend = backend_config._harden_pcss_backend(Flaky())
            self.assertEqual(backend.result("job"), "result")
        self.assertEqual(Flaky.calls, 3)

    def test_hardened_submit_retries_connect_errors_and_timeouts(self):
        class Backend:
            attempts = 0

            def result(self, job_id, *, timeout=None):
                return None

            def submit(self, circuit):
                Backend.attempts += 1
                if Backend.attempts == 1:
                    raise httpx.ConnectError("refused")
                if Backend.attempts == 2:
                    raise httpx.ReadTimeout("request may have reached the server")
                return "id"

        with mock.patch.object(backend_config, "_RETRY_SECONDS", 0):
            backend = backend_config._harden_pcss_backend(Backend())
            # a timeout may leave one orphan job server-side (~18 s billed); giving up would stop the run
            self.assertEqual(backend.submit(object()), "id")
        self.assertEqual(Backend.attempts, 3)


if __name__ == "__main__":
    unittest.main()
