"""Shot/time budget planner for PIAST-Q hardware execution.

The real machine time available to run the 14 QAOA-TCS / IGDec-QAOA combos in
this study is split across one or more named POOLS, each with its own
wall-clock hour budget and an explicit list of which combos it covers. A pool
covering every combo is a "global" setup; 14 pools of one combo each is
"per_combo"; anything in between (e.g. one pool per algorithm, or one pool per
dataset) is just a different assignment of the same `combos: [...]` list --
there is no separate code path per split.

Given a pool's budget, this planner degrades a target execution profile
(repetitions x shots-per-circuit x mitigation methods) in priority order:
  1. reduce the number of repetitions (QAOA-TCS: how many times the same,
     reusable circuit set is re-run; IGDec-QAOA: how many of the distinct,
     non-reusable per-sampling circuit sets are included),
  2. reduce shots-per-circuit, down to a configurable floor (never below one
     200-shot batch),
  3. only as a last resort, and with an explicit warning, subsample the
     circuit/cluster set -- reporting exactly which circuits get dropped and
     why.
Before any of that, if TREx alone is what's blowing the budget (it requires a
full second hardware pass over every raw circuit, since it cannot reuse raw
counts), the planner checks whether dropping TREx for the pool is enough on
its own, and flags that as the first, cheapest option.

Full MEM/M3 calibration circuits are counted once per distinct circuit width
actually used within a pool (reused across every circuit sharing that width,
regardless of which combo or repetition it belongs to), not once per combo or
per repetition.

Nothing in this module calls AQTProvider/AQTSampler or any backend. Circuit
widths are read locally from already-saved .qpy files via `qiskit.qpy.load`
(pure deserialization, no execution).
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

ALL_METHODS = ("raw", "mem", "m3", "trex")


# ---------------------------------------------------------------------------
# Circuit inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircuitRecord:
    circuit_id: str
    num_qubits: int


@dataclass
class ComboInventory:
    """What a combo actually needs executed, discovered from disk (or, in
    tests, hand-built) rather than assumed.
    """
    combo: str
    algorithm: str        # "qaoa_tcs" | "igdec_qaoa"
    objective_mode: str   # "single_objective" | "multi_objective"
    reusable_across_repetitions: bool
    # One entry per repetition unit. QAOA-TCS: exactly one entry (the
    # canonical circuit set, reused for every repetition). IGDec-QAOA: one
    # entry per discovered sampling_N folder (each is a distinct circuit set
    # that cannot be reused for a different repetition).
    repetition_units: List[List[CircuitRecord]] = field(default_factory=list)

    @property
    def available_repetitions(self) -> int:
        if self.reusable_across_repetitions:
            return math.inf
        return len(self.repetition_units)

    def effective_circuits(self, num_repetitions: int, dropped_ids: Optional[Set[str]] = None) -> List[CircuitRecord]:
        dropped_ids = dropped_ids or set()
        if num_repetitions <= 0 or not self.repetition_units:
            return []
        if self.reusable_across_repetitions:
            base = [r for r in self.repetition_units[0] if r.circuit_id not in dropped_ids]
            return base * int(num_repetitions)
        units = self.repetition_units[: int(num_repetitions)]
        result: List[CircuitRecord] = []
        for unit in units:
            result.extend(r for r in unit if r.circuit_id not in dropped_ids)
        return result


def _qpy_num_qubits(path: str) -> int:
    from qiskit import qpy

    with open(path, "rb") as f:
        circuits = qpy.load(f)
    return circuits[0].num_qubits


def discover_qaoa_tcs_inventory(
    combo: str,
    trained_circuits_dir: str,
    objective_mode: str = "single_objective",
    rep_label: str = "rep_1",
) -> ComboInventory:
    """Reads trained_qaoa_circuits/{combo}/{rep_label}/*.qpy -- the one
    circuit set QAOA-TCS reuses across every repetition/experiment."""
    circuit_dir = os.path.join(trained_circuits_dir, combo, rep_label)
    files = sorted(glob.glob(os.path.join(circuit_dir, "*.qpy")))
    records = [CircuitRecord(circuit_id=os.path.basename(f), num_qubits=_qpy_num_qubits(f)) for f in files]
    return ComboInventory(
        combo=combo,
        algorithm="qaoa_tcs",
        objective_mode=objective_mode,
        reusable_across_repetitions=True,
        repetition_units=[records],
    )


def discover_igdec_inventory(
    combo: str,
    trained_circuits_dir: str,
    objective_mode: str = "single_objective",
) -> ComboInventory:
    """Reads trained_qaoa_circuits/igdec_qaoa/{combo}/sampling_*/*.qpy --
    IGDec-QAOA trains a distinct, non-reusable circuit set per sampling run."""
    combo_dir = os.path.join(trained_circuits_dir, "igdec_qaoa", combo)
    sampling_dirs = sorted(
        (d for d in glob.glob(os.path.join(combo_dir, "sampling_*")) if os.path.isdir(d)),
        key=lambda d: int(d.rsplit("_", 1)[-1]),
    )
    repetition_units = []
    for sdir in sampling_dirs:
        files = sorted(glob.glob(os.path.join(sdir, "*.qpy")))
        label = os.path.basename(sdir)
        records = [
            CircuitRecord(circuit_id=f"{label}/{os.path.basename(f)}", num_qubits=_qpy_num_qubits(f))
            for f in files
        ]
        repetition_units.append(records)
    return ComboInventory(
        combo=combo,
        algorithm="igdec_qaoa",
        objective_mode=objective_mode,
        reusable_across_repetitions=False,
        repetition_units=repetition_units,
    )


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    seconds_per_batch: float = 15.0
    shots_per_batch_cap: int = 200
    calibration_shots_per_circuit: int = 200

    def batches(self, shots: int) -> int:
        if shots <= 0:
            return 0
        return -(-shots // self.shots_per_batch_cap)  # ceil division

    def circuit_seconds(self, shots: int) -> float:
        return self.batches(shots) * self.seconds_per_batch

    def calibration_seconds_per_circuit(self) -> float:
        return self.circuit_seconds(self.calibration_shots_per_circuit)

    @staticmethod
    def mem_calibration_circuits(width: int) -> int:
        """One computational-basis-state preparation per possible bitstring."""
        return 2 ** width

    @staticmethod
    def m3_calibration_circuits(width: int) -> int:
        """M3's per-qubit marginal calibration: two preparations (0/1) per qubit."""
        return 2 * width


# ---------------------------------------------------------------------------
# Pools and combo configuration
# ---------------------------------------------------------------------------

@dataclass
class PoolSpec:
    name: str
    total_hours: float
    combos: List[str]


@dataclass
class PlannerConfig:
    cost_model: CostModel
    target_repetitions: int
    target_shots_per_circuit: int
    min_shots_per_circuit: int
    methods_by_combo: Dict[str, List[str]]
    pools: List[PoolSpec]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class PoolPlan:
    pool: str
    total_hours_budget: float
    repetitions: int
    shots_per_circuit: int
    methods_by_combo: Dict[str, List[str]]
    dropped_trex_combos: List[str]
    dropped_circuits: Dict[str, List[str]]
    estimated_seconds: float
    fits_budget: bool
    notes: List[str]

    @property
    def estimated_hours(self) -> float:
        return self.estimated_seconds / 3600.0

    def to_json_dict(self) -> dict:
        return {
            "pool": self.pool,
            "total_hours_budget": self.total_hours_budget,
            "repetitions": self.repetitions,
            "shots_per_circuit": self.shots_per_circuit,
            "methods_by_combo": self.methods_by_combo,
            "dropped_trex_combos": self.dropped_trex_combos,
            "dropped_circuits": self.dropped_circuits,
            "estimated_hours": round(self.estimated_hours, 3),
            "fits_budget": self.fits_budget,
            "notes": self.notes,
        }


def _pool_seconds(
    combos: Sequence[str],
    inventories: Dict[str, ComboInventory],
    repetitions: int,
    shots: int,
    methods_by_combo: Dict[str, List[str]],
    cost_model: CostModel,
    dropped_circuits: Optional[Dict[str, List[str]]] = None,
) -> float:
    dropped_circuits = dropped_circuits or {}
    raw_seconds = 0.0
    trex_seconds = 0.0
    widths_needing_mem: Set[int] = set()
    widths_needing_m3: Set[int] = set()

    for combo in combos:
        inv = inventories[combo]
        dropped = set(dropped_circuits.get(combo, []))
        circuits = inv.effective_circuits(repetitions, dropped)
        n = len(circuits)
        per_circuit = cost_model.circuit_seconds(shots)
        raw_seconds += n * per_circuit

        methods = methods_by_combo.get(combo, ["raw"])
        if "trex" in methods:
            trex_seconds += n * per_circuit
        if n:
            widths = {c.num_qubits for c in circuits}
            if "mem" in methods:
                widths_needing_mem |= widths
            if "m3" in methods:
                widths_needing_m3 |= widths

    calibration_seconds = 0.0
    per_calib_circuit = cost_model.calibration_seconds_per_circuit()
    for w in widths_needing_mem:
        calibration_seconds += cost_model.mem_calibration_circuits(w) * per_calib_circuit
    for w in widths_needing_m3:
        calibration_seconds += cost_model.m3_calibration_circuits(w) * per_calib_circuit

    return raw_seconds + trex_seconds + calibration_seconds


def _shot_candidates(target_shots: int, min_shots: int, batch: int) -> List[int]:
    candidates = []
    s = target_shots
    while s > min_shots:
        candidates.append(s)
        s -= batch
    candidates.append(min_shots)
    return candidates


def plan_pool(
    pool: PoolSpec,
    inventories: Dict[str, ComboInventory],
    methods_by_combo: Dict[str, List[str]],
    cost_model: CostModel,
    target_repetitions: int,
    target_shots_per_circuit: int,
    min_shots_per_circuit: int,
) -> PoolPlan:
    combos = pool.combos
    budget_seconds = pool.total_hours * 3600.0
    notes: List[str] = []
    working_methods = {c: list(methods_by_combo.get(c, ["raw"])) for c in combos}
    dropped_trex: List[str] = []

    # Phase 1: target configuration, as requested.
    seconds = _pool_seconds(combos, inventories, target_repetitions, target_shots_per_circuit, working_methods, cost_model)
    if seconds <= budget_seconds:
        return PoolPlan(pool.name, pool.total_hours, target_repetitions, target_shots_per_circuit,
                         working_methods, [], {}, seconds, True, notes)

    # Phase 2: dropping TREx pool-wide is the cheapest single lever -- try it
    # before sacrificing repetitions or shots.
    trex_combos = [c for c in combos if "trex" in working_methods[c]]
    if trex_combos:
        no_trex_methods = {c: [m for m in ms if m != "trex"] for c, ms in working_methods.items()}
        seconds_no_trex = _pool_seconds(combos, inventories, target_repetitions, target_shots_per_circuit, no_trex_methods, cost_model)
        if seconds_no_trex <= budget_seconds:
            notes.append(
                f"Dropped TREx for {trex_combos} to fit pool budget at target repetitions/shots "
                f"(TREx requires a full second hardware pass and cannot reuse raw counts)."
            )
            return PoolPlan(pool.name, pool.total_hours, target_repetitions, target_shots_per_circuit,
                             no_trex_methods, trex_combos, {}, seconds_no_trex, True, notes)
        # Dropping TREx alone isn't enough, but we still drop it: keeping an
        # expensive method while also degrading everything else is wasteful.
        working_methods = no_trex_methods
        dropped_trex = trex_combos
        notes.append(f"Dropped TREx for {trex_combos} (insufficient alone; further degradation still required).")

    # Phase 3: reduce repetitions, from target down to 1.
    best_repetitions = None
    for r in range(int(target_repetitions), 0, -1):
        seconds = _pool_seconds(combos, inventories, r, target_shots_per_circuit, working_methods, cost_model)
        if seconds <= budget_seconds:
            best_repetitions = r
            break
    if best_repetitions is not None:
        if best_repetitions < target_repetitions:
            notes.append(f"Reduced repetitions from {target_repetitions} to {best_repetitions} to fit pool budget.")
        seconds = _pool_seconds(combos, inventories, best_repetitions, target_shots_per_circuit, working_methods, cost_model)
        return PoolPlan(pool.name, pool.total_hours, best_repetitions, target_shots_per_circuit,
                         working_methods, dropped_trex, {}, seconds, True, notes)

    # Phase 4: repetitions=1 still doesn't fit at target shots -- reduce shots.
    shot_candidates = _shot_candidates(target_shots_per_circuit, min_shots_per_circuit, cost_model.shots_per_batch_cap)
    best_shots = None
    for s in shot_candidates:
        seconds = _pool_seconds(combos, inventories, 1, s, working_methods, cost_model)
        if seconds <= budget_seconds:
            best_shots = s
            break
    if best_shots is not None:
        if best_shots < target_shots_per_circuit:
            notes.append(f"Reduced repetitions to 1 and shots/circuit from {target_shots_per_circuit} to {best_shots} to fit pool budget.")
        seconds = _pool_seconds(combos, inventories, 1, best_shots, working_methods, cost_model)
        return PoolPlan(pool.name, pool.total_hours, 1, best_shots,
                         working_methods, dropped_trex, {}, seconds, True, notes)

    # Phase 5: last resort -- subsample circuits at repetitions=1, shots=floor.
    notes.append(
        f"WARNING: even at repetitions=1 and the shot floor ({min_shots_per_circuit}), "
        f"pool '{pool.name}' does not fit its {pool.total_hours}h budget. Subsampling circuits."
    )
    dropped: Dict[str, List[str]] = defaultdict(list)
    max_iterations = sum(len(inventories[c].effective_circuits(1)) for c in combos) + 1
    fits = False
    for _ in range(max_iterations):
        seconds = _pool_seconds(combos, inventories, 1, min_shots_per_circuit, working_methods, cost_model, dropped)
        if seconds <= budget_seconds:
            fits = True
            break
        candidates: List[Tuple[int, str, CircuitRecord]] = []
        for c in combos:
            remaining = inventories[c].effective_circuits(1, set(dropped[c]))
            if remaining:
                candidates.append((len(remaining), c, remaining[-1]))
        if not candidates:
            notes.append(
                "INFEASIBLE: every circuit has been dropped and the pool still does not fit its budget "
                "(fixed MEM/M3 calibration overhead alone may exceed the budget -- consider dropping "
                "MEM/M3 for this pool or increasing total_hours)."
            )
            break
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, combo_to_trim, victim = candidates[0]
        dropped[combo_to_trim].append(victim.circuit_id)
        notes.append(
            f"Dropped circuit {victim.circuit_id} ({victim.num_qubits} qubits) from combo '{combo_to_trim}': "
            f"pool '{pool.name}' budget ({pool.total_hours}h) exceeded even at repetitions=1, "
            f"shots={min_shots_per_circuit}."
        )

    seconds = _pool_seconds(combos, inventories, 1, min_shots_per_circuit, working_methods, cost_model, dropped)
    return PoolPlan(pool.name, pool.total_hours, 1, min_shots_per_circuit,
                     working_methods, dropped_trex, dict(dropped), seconds, fits, notes)


def plan_all(config: PlannerConfig, inventories: Dict[str, ComboInventory]) -> List[PoolPlan]:
    return [
        plan_pool(
            pool,
            inventories,
            config.methods_by_combo,
            config.cost_model,
            config.target_repetitions,
            config.target_shots_per_circuit,
            config.min_shots_per_circuit,
        )
        for pool in config.pools
    ]


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> PlannerConfig:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    cost_model = CostModel(
        seconds_per_batch=raw.get("seconds_per_batch", 15.0),
        shots_per_batch_cap=raw.get("shots_per_batch_cap", 200),
        calibration_shots_per_circuit=raw.get("calibration_shots_per_circuit", 200),
    )

    methods_by_combo = {}
    for combo_name, combo_cfg in raw.get("combos", {}).items():
        methods_by_combo[combo_name] = list(combo_cfg.get("methods", ["raw"]))

    pools = [
        PoolSpec(name=p["name"], total_hours=p.get("total_hours", 15.0), combos=list(p["combos"]))
        for p in raw.get("pools", [])
    ]

    return PlannerConfig(
        cost_model=cost_model,
        target_repetitions=raw.get("target_repetitions", 10),
        target_shots_per_circuit=raw.get("target_shots_per_circuit", 2048),
        min_shots_per_circuit=raw.get("min_shots_per_circuit", 200),
        methods_by_combo=methods_by_combo,
        pools=pools,
    )


def build_inventories_from_config(path: str, trained_circuits_dir: str) -> Dict[str, ComboInventory]:
    """Reads combo definitions (algorithm/objective_mode/dataset/circuits_dir)
    from the same YAML file used by load_config() and discovers each combo's
    real circuit inventory from disk. Local .qpy deserialization only -- no
    backend, no AQTProvider/AQTSampler.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    inventories = {}
    for combo_name, combo_cfg in raw.get("combos", {}).items():
        algorithm = combo_cfg["algorithm"]
        objective_mode = combo_cfg.get("objective_mode", "single_objective")
        circuits_dir = combo_cfg.get("circuits_dir", combo_cfg.get("dataset", combo_name))
        if algorithm == "qaoa_tcs":
            inventories[combo_name] = discover_qaoa_tcs_inventory(
                circuits_dir, trained_circuits_dir, objective_mode=objective_mode
            )
        elif algorithm == "igdec_qaoa":
            inventories[combo_name] = discover_igdec_inventory(
                circuits_dir, trained_circuits_dir, objective_mode=objective_mode
            )
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}' for combo '{combo_name}'")
    return inventories


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_report(plans: List[PoolPlan]) -> str:
    lines = ["PIAST-Q execution budget plan", "=" * 32, ""]
    total_hours_budget = sum(p.total_hours_budget for p in plans)
    total_hours_estimated = sum(p.estimated_hours for p in plans)

    for plan in plans:
        lines.append(f"Pool: {plan.pool}")
        lines.append(f"  Budget: {plan.total_hours_budget}h | Estimated: {plan.estimated_hours:.2f}h | Fits: {plan.fits_budget}")
        lines.append(f"  Repetitions: {plan.repetitions}")
        lines.append(f"  Shots/circuit: {plan.shots_per_circuit}")
        lines.append(f"  Combos: {list(plan.methods_by_combo.keys())}")
        for combo, methods in plan.methods_by_combo.items():
            lines.append(f"    - {combo}: methods={methods}")
        if plan.dropped_trex_combos:
            lines.append(f"  Dropped TREx for: {plan.dropped_trex_combos}")
        if plan.dropped_circuits:
            for combo, ids in plan.dropped_circuits.items():
                lines.append(f"  Dropped {len(ids)} circuit(s) from {combo}: {ids}")
        for note in plan.notes:
            lines.append(f"  NOTE: {note}")
        lines.append("")

    lines.append("Summary")
    lines.append("-" * 32)
    lines.append(f"Total budget across pools: {total_hours_budget:.2f}h")
    lines.append(f"Total estimated: {total_hours_estimated:.2f}h")
    lines.append(f"All pools fit: {all(p.fits_budget for p in plans)}")

    return "\n".join(lines)


def write_report(plans: List[PoolPlan], output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "execution_plan_report.txt")
    json_path = os.path.join(output_dir, "execution_plan_report.json")

    with open(text_path, "w") as f:
        f.write(render_report(plans))

    with open(json_path, "w") as f:
        json.dump([p.to_json_dict() for p in plans], f, indent=2)

    return text_path, json_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "..", "configs", "execution_plan.yaml")
    circuits_dir = os.path.join(here, "..", "trained_qaoa_circuits")

    config = load_config(config_path)
    inventories = build_inventories_from_config(config_path, circuits_dir)
    plans = plan_all(config, inventories)

    report = render_report(plans)
    print(report)

    output_dir = os.path.join(here, "..", "results", "execution_plan")
    text_path, json_path = write_report(plans, output_dir)
    print(f"\nSaved report: {text_path}")
    print(f"Saved report: {json_path}")
