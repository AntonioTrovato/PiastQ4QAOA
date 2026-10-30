"""Packing several independent circuits into one hardware job.

Why: on PIAST-Q every job costs minutes of machine time in overhead, however
small the circuit (measured: 170-290 s per job, while the billed QPU time is
18.5 s for 200 shots whatever the qubit count). Our circuits use 1-7 of the
machine's 20 qubits, so most qubits idle during every job. Placing several
circuits side by side on DISJOINT qubits of one job, running them together
and splitting the counts afterwards, cuts the number of jobs ~5x. The circuits
stay independent: same circuits, same shots each; only the hardware is shared.

Layout: the 20 qubits are cut into fixed SLOTS (configs/backend.yaml
`pack_slots`, default [0..6], [7..13], [14..19]). A job holds up to one set of
circuits per slot; circuits are placed one after another inside a slot. The
calibration (MEM/M3) is measured per slot / per qubit and a circuit's window
inside a slot is derived from it (mitigation.CalibrationStore.load_window).

Bit order follows qiskit: in a counts key the rightmost character is the
lowest classical bit; combined qubit k is measured into clbit k.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_PACK_SLOTS: List[Tuple[int, int]] = [(0, 7), (7, 7), (14, 6)]  # (first physical qubit, width)


def load_pack_slots(config_path: Optional[str] = None) -> List[Tuple[int, int]]:
    """Slots from configs/backend.yaml's `pack_slots` ([[start, width], ...])."""
    import yaml

    if config_path is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(root, "configs", "backend.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    raw = config.get("pack_slots")
    slots = [(int(s), int(w)) for s, w in raw] if raw else list(DEFAULT_PACK_SLOTS)
    _validate_slots(slots)
    return slots


def _validate_slots(slots: Sequence[Tuple[int, int]]) -> None:
    covered = set()
    for start, width in slots:
        qubits = set(range(start, start + width))
        if width < 1 or covered & qubits:
            raise ValueError(f"invalid or overlapping pack slots: {list(slots)!r}")
        covered |= qubits


def slot_qubits(slots: Sequence[Tuple[int, int]]) -> List[int]:
    """All physical qubits of all slots, in slot order."""
    return [q for start, width in slots for q in range(start, start + width)]


@dataclass(frozen=True)
class Placement:
    member_index: int      # index into the caller's list of circuits
    start: int             # first physical qubit
    width: int
    offset: int = 0        # first qubit of this member in the COMBINED circuit


def plan_packs(widths: Sequence[int], slots: Optional[Sequence[Tuple[int, int]]] = None) -> List[List[Placement]]:
    """First-fit-decreasing placement of circuits of the given widths into jobs.

    Deterministic (ties broken by index). Returns one list of Placements per
    job; within a job the placements are ordered by physical qubit, and
    `offset` gives each member's position in the combined circuit.
    """
    slots = list(slots) if slots is not None else load_pack_slots()
    widest = max(width for _, width in slots)
    jobs: List[List[int]] = []                 # per job: used width of each slot
    placed: List[List[Tuple[int, int, int]]] = []   # per job: (member, start, width)

    for index in sorted(range(len(widths)), key=lambda i: (-widths[i], i)):
        width = widths[index]
        if width > widest:
            raise ValueError(f"circuit {index} is {width} qubits wide; the widest pack slot has {widest}")
        for job_used, job_placed in zip(jobs, placed):
            slot = next((s for s, (_, w) in enumerate(slots) if job_used[s] + width <= w), None)
            if slot is not None:
                job_placed.append((index, slots[slot][0] + job_used[slot], width))
                job_used[slot] += width
                break
        else:
            slot = next(s for s, (_, w) in enumerate(slots) if width <= w)
            used = [0] * len(slots)
            used[slot] = width
            jobs.append(used)
            placed.append([(index, slots[slot][0], width)])

    packs: List[List[Placement]] = []
    for job_placed in placed:
        offset = 0
        pack = []
        for member, start, width in sorted(job_placed, key=lambda t: t[1]):
            pack.append(Placement(member, start, width, offset))
            offset += width
        packs.append(pack)
    return packs


def count_jobs(widths: Sequence[int], slots: Optional[Sequence[Tuple[int, int]]] = None) -> int:
    return len(plan_packs(widths, slots)) if widths else 0


def build_packed_circuit(circuits: Sequence, pack: Sequence[Placement]):
    """Combines the member circuits of one job into one circuit.

    Returns (combined_circuit, layout): combined qubit k must run on physical
    qubit layout[k] (pass it as the transpiler's initial_layout). Any final
    measurements the members carry are dropped; the combined circuit measures
    every qubit once, qubit k into clbit k.
    """
    from qiskit import QuantumCircuit

    total = sum(p.width for p in pack)
    combined = QuantumCircuit(total)
    layout: List[int] = []
    for placement in pack:
        member = circuits[placement.member_index]
        body = member.remove_final_measurements(inplace=False)
        if body.num_qubits != placement.width:
            raise ValueError(f"member {placement.member_index} is {body.num_qubits} qubits, expected {placement.width}")
        combined.compose(body, qubits=list(range(placement.offset, placement.offset + placement.width)), inplace=True)
        layout.extend(range(placement.start, placement.start + placement.width))
    combined.measure_all()
    return combined, layout


def marginalize_counts(counts: Dict[str, int], offset: int, width: int) -> Dict[str, int]:
    """Counts of the `width` clbits starting at clbit `offset` of a joint
    bitstring (rightmost character = clbit 0)."""
    out: Counter = Counter()
    for bitstring, count in counts.items():
        low = len(bitstring) - offset - width
        out[bitstring[low:low + width]] += count
    return dict(out)
