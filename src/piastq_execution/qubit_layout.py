"""Single source of truth for which physical PIAST-Q qubits a circuit of a
given width should use.

Before this module existed, both the real QAOA/TREx circuit execution
(`raw_counts.run_circuit_with_batching_recorded`) and the MEM/M3 calibration
circuits (`run_calibration.py`) let the transpiler pick a physical layout
automatically (`optimization_level=3`, no `initial_layout`), independently of
each other. Nothing guaranteed they'd agree -- `run_calibration.py`'s own
former docstring flagged this explicitly as a known simplification ("using
logical qubit indices 0..width-1 as the physical qubit list"), not a
guarantee that real QAOA circuits would land on the same physical qubits. A
MEM/M3 calibration built from one set of physical qubits, silently applied to
counts collected from a different set, would be wrong.

Configure the actual physical qubits to use (in priority order) via
`configs/backend.yaml`'s `physical_qubits` list, e.g.:

    physical_qubits: [3, 4, 5, 6, 7, 8, 9]

A circuit of width `w` always gets `physical_qubits[:w]` -- the same slice,
every time, from every caller (real execution, TREx twirl passes, and MEM/M3
calibration alike) -- so they are provably pinned to identical physical
qubits instead of merely assumed to be. Defaults to `[0, 1, ..., 6]` if not
configured, matching the implicit assumption every script made before this
module existed.
"""

from __future__ import annotations

import os
from typing import List, Optional

_DEFAULT_PHYSICAL_QUBITS = [0, 1, 2, 3, 4, 5, 6]


def _repo_root() -> str:
    # This file lives at <repo_root>/src/piastq_execution/qubit_layout.py.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_config_path() -> str:
    return os.path.join(_repo_root(), "configs", "backend.yaml")


def load_physical_qubits(config_path: Optional[str] = None) -> List[int]:
    """Returns the configured, ordered list of physical qubits
    (`configs/backend.yaml`'s `physical_qubits`), or the default `[0..6]` if
    not configured.
    """
    import yaml

    path = config_path or _default_config_path()

    config = {}
    if os.path.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f) or {}

    qubits = config.get("physical_qubits", _DEFAULT_PHYSICAL_QUBITS)
    return [int(q) for q in qubits]


def physical_layout_for_width(width: int, physical_qubits: Optional[List[int]] = None) -> List[int]:
    """Returns the first `width` physical qubits from the configured (or
    given) ordered list -- the same slice every caller must use for a
    width-`width` circuit, so calibration and execution never disagree.

    Raises ValueError if fewer than `width` physical qubits are configured,
    rather than silently truncating or wrapping around.
    """
    qubits = physical_qubits if physical_qubits is not None else load_physical_qubits()
    if width > len(qubits):
        raise ValueError(
            f"Requested a width-{width} layout but only {len(qubits)} physical "
            f"qubits are configured ({qubits!r}) -- add more entries to "
            f"configs/backend.yaml's physical_qubits list."
        )
    return list(qubits[:width])
