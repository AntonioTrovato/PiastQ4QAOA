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

import os
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


def get_backend(config_path: Optional[str] = None):
    """Builds and returns the AQTProvider-backed backend configured in
    configs/backend.yaml (or the PIASTQ_API_TOKEN/PIASTQ_BACKEND_NAME env
    vars). This is the only place AQTProvider(...) should be constructed --
    every execution script calls this instead of doing it inline.
    """
    from qiskit_aqt_provider import AQTProvider

    api_token, backend_name = load_backend_config(config_path)
    provider = AQTProvider(api_token)
    return provider.get_backend(backend_name)
