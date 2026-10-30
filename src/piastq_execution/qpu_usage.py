"""Billed QPU time of the PIAST-Q account, and the hard budget stop.

The 25 QPU-hour allowance is counted by the PCSS service as *billed* device
time (`/me/usage`), not as wall-clock: queue wait is free, and a 200-shot
batch is billed a fixed 18.4974 s regardless of how many qubits the circuit
has (measured: 1, 3 and 7 qubits all billed 18.4974 s; 100 shots billed
10.0774 s). Everything that reasons about the budget therefore reads this
counter instead of timing calls itself.

`check_budget()` is called before every hardware batch. When the environment
variable PIASTQ_QPU_LIMIT_SECONDS is set (run_all.py sets it) and the next
batch would push the billed time past it, it raises `BudgetExceeded`: the
stage stops cleanly, everything executed so far is already on disk, and the
run can be continued with `python run_all.py --resume`. On offline simulators
there is no usage counter, so every function here is a no-op / returns None.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

LIMIT_ENV = "PIASTQ_QPU_LIMIT_SECONDS"
DEVICE_NAME = "PIAST-Q"
# Measured billing of one 200-shot batch (see module docstring).
BILLED_SECONDS_PER_200_SHOT_BATCH = 18.4974
# Billing model fitted on the account usage counter: 100 shots -> 10.0774 s,
# 200 shots -> 18.4974 s (1-, 3- and 7-qubit circuits alike).
BILLED_FIXED_SECONDS = 1.6574
BILLED_SECONDS_PER_SHOT = 0.0842


class BudgetExceeded(RuntimeError):
    pass


def modeled_billed_seconds(shots: int) -> Optional[float]:
    """QPU seconds one batch of `shots` shots is billed on PIAST-Q, or None on
    anything that is not the real machine (simulators bill nothing). Used for
    the execution-time metrics: on the real machine wall-clock is dominated by
    queue wait (minutes per job), which is not QPU time."""
    from piastq_execution.backend_config import is_pcss_backend, load_backend_config

    if not is_pcss_backend(load_backend_config()[1]):
        return None
    return BILLED_FIXED_SECONDS + BILLED_SECONDS_PER_SHOT * shots


def billed_seconds() -> Optional[float]:
    """QPU seconds billed to this account so far (member quota on PIAST-Q), or
    None if there is no usage counter (offline simulator) or it can't be read."""
    from piastq_execution.backend_config import get_openqapi_client

    client = get_openqapi_client()
    if client is None:
        return None
    try:
        for usage in client.get_my_usage(verbose=False):
            quota = usage.get("member_quota")
            if usage["device"]["name"] == DEVICE_NAME and quota:
                return quota["used"] / 1e6
    except Exception as exc:  # a failed read must never break an execution
        print(f"[qpu_usage] could not read billed usage: {type(exc).__name__}: {exc}", file=sys.stderr)
    return None


def check_budget(next_batch_seconds: float = BILLED_SECONDS_PER_200_SHOT_BATCH) -> None:
    """Raises BudgetExceeded if running one more batch would exceed the limit
    in PIASTQ_QPU_LIMIT_SECONDS. No-op when the variable is unset or the
    billed usage can't be read (the service enforces the quota itself)."""
    limit = os.environ.get(LIMIT_ENV)
    if not limit:
        return
    used = billed_seconds()
    if used is not None and used + next_batch_seconds > float(limit):
        raise BudgetExceeded(
            f"billed QPU time {used:.1f}s + next batch {next_batch_seconds:.1f}s would exceed the "
            f"limit of {float(limit):.1f}s ({float(limit) / 3600:.2f}h). Stopping before submitting it; "
            f"everything executed so far is saved (continue with: python run_all.py --resume)."
        )
