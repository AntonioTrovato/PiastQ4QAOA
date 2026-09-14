"""Captures a timestamped snapshot of what the backend *itself* reports about
its qubits (T1, T2, frequency, and whatever else it publishes) -- distinct
from `piastq_execution.mitigation`'s own MEM/M3 calibration, which
characterizes readout error empirically from shots we run ourselves. This
module never runs a circuit; it only reads whatever metadata the provider
already exposes on the backend object.

Wired to run automatically, once per script invocation, from every
hardware-execution entry point (`qaoa_tcs/single_obj.py`,
`qaoa_tcs/multi_obj.py`, the three
`igdec_qaoa/loch_qaoa_*_extract_circuits.py` scripts' execution mode, and
`piastq_execution/run_calibration.py`) right after `get_backend()` -- nobody
has to remember to trigger it by hand, and every real-hardware session
(the first one and every one after it) gets its own dated snapshot, so drift
across however many sessions the full data collection ends up taking can be
checked after the fact.

Every provider populates a different subset of this (AQT's ion-trap hardware
won't necessarily report the same fields IBM's superconducting backends
document their own calibration with), so every field is best-effort:
missing/unavailable data is recorded as `None` rather than raising.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _qubit_property_to_dict(prop: Any) -> Dict[str, Any]:
    """Best-effort extraction of one qubit's reported properties.

    Qiskit's `BackendV2.qubit_properties` entries are typically a
    `QubitProperties` object with `.t1`, `.t2`, `.frequency` attributes (any
    of which may be `None` if the backend doesn't report it) -- but
    providers aren't required to use exactly that type, so every attribute
    access is defensive (`getattr(..., None)`, never assumed present).
    """
    if prop is None:
        return {"t1": None, "t2": None, "frequency": None}
    return {
        "t1": getattr(prop, "t1", None),
        "t2": getattr(prop, "t2", None),
        "frequency": getattr(prop, "frequency", None),
    }


def snapshot_backend_calibration(backend) -> Dict[str, Any]:
    """Best-effort extraction of the backend's own reported qubit calibration
    data (T1, T2, frequency, ...), independent of our own MEM/M3 measurement-
    based calibration in `piastq_execution.mitigation`.
    """
    from piastq_execution.raw_counts import get_backend_identity

    identity = get_backend_identity(backend)

    num_qubits = getattr(backend, "num_qubits", None)
    qubit_properties_raw = getattr(backend, "qubit_properties", None)
    if callable(qubit_properties_raw):
        try:
            qubit_properties_raw = qubit_properties_raw()
        except Exception:
            qubit_properties_raw = None

    qubits: List[Dict[str, Any]] = []
    if qubit_properties_raw:
        for i, prop in enumerate(qubit_properties_raw):
            entry = _qubit_property_to_dict(prop)
            entry["qubit_index"] = i
            qubits.append(entry)
    elif isinstance(num_qubits, int):
        qubits = [
            {"qubit_index": i, "t1": None, "t2": None, "frequency": None}
            for i in range(num_qubits)
        ]

    return {
        "backend_name": identity["backend_name"],
        "backend_version": identity["backend_version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_qubits": num_qubits,
        "qubits": qubits,
    }


def _default_output_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "calibration", "backend_snapshots")


def save_backend_calibration_snapshot(backend, output_dir: Optional[str] = None) -> str:
    """Writes `snapshot_backend_calibration(backend)` to a new timestamped
    file under `calibration/backend_snapshots/` (default) and returns its
    path. Never overwrites a previous snapshot -- each call is its own dated
    record, so calibration drift across sessions stays comparable rather than
    clobbered.
    """
    snapshot = snapshot_backend_calibration(backend)

    directory = output_dir or _default_output_dir()
    os.makedirs(directory, exist_ok=True)

    safe_timestamp = snapshot["timestamp"].replace(":", "-")
    filename = f"{safe_timestamp}_{snapshot['backend_name']}.json"
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return path
