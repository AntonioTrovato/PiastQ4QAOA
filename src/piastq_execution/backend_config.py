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

    def result(job_id, *, timeout=None):
        attempts = 0
        while True:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    return original_result(job_id, timeout=timeout)
            except httpx.TransportError:
                attempts += 1
                if attempts > _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_SECONDS)

    def submit(circuit):
        attempts = 0
        while True:
            try:
                return original_submit(circuit)
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
