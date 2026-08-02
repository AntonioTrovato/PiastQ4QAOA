"""Deterministic seed derivation for IGDec-QAOA's random-restart training.

IGDec-QAOA's training (`save_trained_circuits_and_initial_solutions()` in
each `igdec_qaoa/loch_qaoa_*_extract_circuits.py` script) picks a random
initial solution per sampling and never explicitly seeds the QAOA
optimizer's randomness -- unlike QAOA-TCS's deterministic clustering, this
means re-running `... train` after deleting `trained_qaoa_circuits/` does
not reproduce the same circuits.

`seed_for_sampling()` fixes that: a stable 32-bit integer seed derived from
`(dataset, sampling_id)` via SHA-256, deliberately not Python's built-in
`hash()` (which is randomized per-process for strings unless
`PYTHONHASHSEED` is fixed, so it would silently reintroduce the same
non-reproducibility this module exists to remove).
"""

from __future__ import annotations

import hashlib


def seed_for_sampling(dataset: str, sampling_id: int) -> int:
    """Deterministic 32-bit seed for one IGDec-QAOA sampling run.

    The same `(dataset, sampling_id)` always yields the same seed, across
    processes, machines, and Python versions -- SHA-256 is stable, unlike
    `hash()`. Callers should set this once per sampling, before generating
    that sampling's initial random solution, and reuse it as
    `algorithm_globals.random_seed` for every subproblem solved within that
    sampling (the QAOA optimizer's internal RNG stream stays deterministic
    across that sampling's full sequence of solves as long as the seed
    isn't reset mid-sampling).
    """
    digest = hashlib.sha256(f"{dataset}_sampling_{sampling_id}".encode()).hexdigest()
    return int(digest[:8], 16)
