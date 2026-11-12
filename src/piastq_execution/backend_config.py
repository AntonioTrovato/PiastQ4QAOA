"""Central AQT/PIAST-Q backend + API-token configuration.

Every script that opens a real backend (qaoa_tcs/single_obj.py,
qaoa_tcs/multi_obj.py, the three igdec_qaoa/loch_qaoa_*_extract_circuits.py
scripts, and piastq_execution/run_calibration.py) calls get_backend() here
instead of constructing AQTProvider/get_backend() itself, so switching from
the local offline simulator to real PIAST-Q hardware means editing
configs/backend.yaml (or setting the PIASTQ_API_TOKEN / PIASTQ_BACKEND_NAME
env vars) once, not six files.

Env vars take precedence over the YAML file when set, so a real token never
has to be committed: export PIASTQ_API_TOKEN before running instead of
editing the file, if preferred.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import uuid
from typing import Optional, Tuple

import yaml

_DEFAULT_API_TOKEN = "ACCESS_TOKEN"
_DEFAULT_BACKEND_NAME = "offline_simulator_no_noise"


def _repo_root() -> str:
    # This file lives at <repo_root>/src/piastq_execution/backend_config.py.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_config_path() -> str:
    return os.path.join(_repo_root(), "configs", "backend.yaml")


def load_backend_config(config_path: Optional[str] = None) -> Tuple[str, str]:
    """Returns (api_token, backend_name), resolved in priority order:
    PIASTQ_API_TOKEN/PIASTQ_BACKEND_NAME env vars, then configs/backend.yaml,
    then hardcoded defaults (matching the offline simulator every script used
    before this module existed).
    """
    path = config_path or _default_config_path()

    config = {}
    if os.path.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f) or {}

    api_token = os.environ.get("PIASTQ_API_TOKEN") or config.get("api_token", _DEFAULT_API_TOKEN)
    backend_name = os.environ.get("PIASTQ_BACKEND_NAME") or config.get("backend_name", _DEFAULT_BACKEND_NAME)
    return api_token, backend_name


def is_pcss_backend(backend_name: str) -> bool:
    """True for the real PIAST-Q machine, reached through the PCSS service
    (`pcss_qapi.OpenQAPI`), as opposed to the AQT cloud / offline simulators
    that `AQTProvider` serves. Accepted spellings: piast_q, piast-q, piastq."""
    return backend_name.strip().lower().replace("-", "_") in {"piast_q", "piastq"}


_openqapi_client = None


def get_openqapi_client(config_path: Optional[str] = None):
    """The (cached) authenticated PCSS client when the configured backend is
    PIAST-Q, else None. Needs `pcss-qapi` (in requirements.txt; Python >= 3.11)."""
    global _openqapi_client
    api_token, backend_name = load_backend_config(config_path)
    if not is_pcss_backend(backend_name):
        return None
    if _openqapi_client is None:
        from pcss_qapi import OpenQAPI

        _openqapi_client = OpenQAPI(token=api_token)
    return _openqapi_client


_RETRY_SECONDS = 5
_MAX_RETRIES = 360  # ~30 min of network trouble before giving up


class _JobId:
    """A job id exactly as the service returned it (`.int` mirrors the
    attribute of the UUID the vendor client expected)."""

    def __init__(self, raw):
        self.int = raw

    def __repr__(self):
        return f"_JobId({self.int!r})"


def _wrap_negative_phases(circuit):
    """The PCSS service validates `phi >= 0` on RZ/R gates (angles in units of
    pi) and rejects the whole job otherwise, while the transpiler can emit
    tiny negative angles. A phase is defined modulo 2 (i.e. 2*pi), so adding
    2 to a negative one gives the same operation (an RZ differs only by a
    global phase, which no measurement can see). Only negative angles are
    touched; everything else is submitted exactly as transpiled."""
    # the submitted payload wraps the gate list: payload.quantum_circuit.root
    gates = getattr(circuit, "quantum_circuit", circuit)
    for operation in getattr(gates, "root", []):
        gate = getattr(operation, "root", None)
        phi = getattr(gate, "phi", None)
        if phi is not None and phi < 0:
            gate.phi = phi % 2.0
    return circuit


def _harden_pcss_backend(backend):
    """Makes the PIAST-Q resource survive transient network errors and keeps
    its console output readable. The vendor wrapper (a) dies on any connection
    error while waiting for a result, even though the job is still running
    server-side, and (b) prints every single shot. Waiting is retried (the job
    id is kept, nothing is resubmitted); a submit is retried only when the
    request provably never reached the server (connect errors) or timed out
    (then the job may exist server-side; at worst one orphan job is billed
    ~18 s, while giving up would stop a multi-day run)."""
    import httpx

    original_result, original_submit = backend.result, backend.submit

    def poll_job(raw_id, timeout):
        """GET /jobs/<id> until the result exists (same protocol as the vendor
        loop), with `raw_id` used verbatim in the URL: the service started
        returning the job id as a string, which the vendor client (0.3.2)
        cannot handle (it builds a UUID from an int)."""
        while True:
            try:
                response = backend._http_client.get(
                    f"/jobs/{raw_id}", timeout=backend.timeout
                ).raise_for_status().json()
                if response["status"] in ["failed", "canceled"]:
                    raise RuntimeError(f"Your task has status: {response['status']}")
                if response["result"] is not None:
                    return response["result"]
            except httpx.ReadTimeout:
                pass
            time.sleep(backend.timeout)

    def result(job_id, *, timeout=None):
        attempts = 0
        while True:
            try:
                if not isinstance(job_id, _JobId):
                    with contextlib.redirect_stdout(io.StringIO()):
                        return original_result(job_id, timeout=timeout)
                payload = poll_job(job_id.int, timeout)
                formatted = [list(map(int, bitstring)) for bitstring in payload["results"]["Tasks"]]
                from qiskit_aqt_provider.api_client import models_direct as api_models_direct

                return api_models_direct.JobResult.model_validate({
                    "job_id": uuid.uuid5(uuid.NAMESPACE_URL, str(job_id.int)),
                    "payload": {"status": "finished", "result": formatted},
                })
            except httpx.TransportError:
                attempts += 1
                if attempts > _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_SECONDS)
            except httpx.HTTPStatusError as exc:
                # a 5xx on a GET is a service hiccup (one killed a run on 10 Nov); the job still exists
                if exc.response.status_code < 500:
                    raise
                attempts += 1
                if attempts > _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_SECONDS)

    def submit(circuit):
        _wrap_negative_phases(circuit)
        attempts = 0
        while True:
            try:
                backend.last_task_uid = None
                try:
                    return original_submit(circuit)
                except TypeError:
                    # the POST succeeded (the vendor code stored the id before failing to
                    # build a UUID from it): the service now returns the id as a string
                    if backend.last_task_uid is None:
                        raise
                    return _JobId(backend.last_task_uid)
            except (httpx.ConnectError, httpx.TimeoutException):
                attempts += 1
                if attempts > _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_SECONDS)

    backend.result = result
    backend.submit = submit
    return backend


def get_backend(config_path: Optional[str] = None):
    """Builds and returns the backend configured in configs/backend.yaml (or
    the PIASTQ_API_TOKEN/PIASTQ_BACKEND_NAME env vars). This is the only place
    a provider is constructed -- every execution script calls this instead of
    doing it inline.

    backend_name `piast_q` -> the real PIAST-Q machine via the PCSS service
    (pcss_qapi); anything else (offline_simulator_no_noise, ...) -> AQTProvider.
    """
    api_token, backend_name = load_backend_config(config_path)

    if is_pcss_backend(backend_name):
        client = get_openqapi_client(config_path)
        return _harden_pcss_backend(client.piast_q_backend())

    from qiskit_aqt_provider import AQTProvider

    provider = AQTProvider(api_token)
    return provider.get_backend(backend_name)
