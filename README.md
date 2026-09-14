# PIAST-Q4QAOA — Real-Hardware Replication Package

This is a **full replication package** extending the QAOA-TCS
empirical study onto the **real PIAST-Q
quantum computer**, with readout-error mitigation.

It implements and evaluates two quantum approaches for Test Case Selection:

- **QAOA-TCS**, single- and multi-objective
- **IGDec-QAOA**, single-objective only

The pipeline, end to end:

1. **Decompose & cluster** the dataset into small (<=7-qubit) subproblems
   (already done, embedded in `single_obj.py`/`multi_obj.py`/the IGDec
   scripts — this math is untouched by this replication package).
2. **Train** the QAOA circuits that solve each subproblem (most combos
   already have their circuits ported — see the matrix below; ideal-simulator
   training/retraining code is included for all 14 combos, `train` mode,
   §4.1/§4.2).
3. **Execute** those trained circuits on real PIAST-Q hardware.
4. **Mitigate** readout error (three interchangeable methods: full MEM, M3,
   TREx).
5. **Plan** how much of that execution a given hardware-time budget can
   actually afford.
6. **Evaluate** and statistically compare raw vs. mitigated results.

---

## 1. Directory layout

```
PiastQ4QAOA/
├── src/                              # all importable code lives here
│   ├── qaoa_tcs/
│   │   ├── single_obj.py             # QAOA-TCS, single-objective (5 datasets)
│   │   └── multi_obj.py              # QAOA-TCS, multi-objective (4 datasets)
│   ├── igdec_qaoa/
│   │   ├── loch_qaoa_tcm_extract_circuits.py       # iofrol, paintcontrol, gsdtsr
│   │   ├── loch_qaoa_elev_two_extract_circuits.py  # elevator_o2 (elevator_two)
│   │   └── loch_qaoa_elev_three_extract_circuits.py# elevator_o3 (elevator_three)
│   └── piastq_execution/             # shared library, used by every script above
│       ├── raw_counts.py             #   shot-batched execution + full raw-data capture
│       ├── mitigation.py             #   full MEM / M3 / TREx
│       ├── qubo_io.py                #   persisting/loading trained circuits' QUBO coefficients
│       ├── sir_metrics.py            #   SIR static cost/fault/coverage data + Pareto fronts
│       ├── seeding.py                #   deterministic seeds for IGDec-QAOA's random sampling
│       ├── backend_config.py         #   single place to set the real PIAST-Q API token/backend
│       ├── qubit_layout.py           #   which physical qubits a width-w circuit uses (see §5)
│       ├── backend_calibration_snapshot.py #   T1/T2/etc. snapshot, once per session (see §5)
│       ├── run_calibration.py        #   MEM/M3 calibration entry point (see §5)
│       ├── budget_planner.py         #   hardware-time budget planning
│       ├── evaluation.py             #   per-combo x per-method metrics
│       ├── evaluate_all.py           #   runs evaluation.py across every combo x method (see §8)
│       ├── compare_all.py            #   runs statistics.py across every combo, RQ-mapped report
│       └── statistics.py             #   Shapiro-Wilk-gated statistical comparison
├── datasets/                         # input data (unchanged)
├── trained_qaoa_circuits/
│   ├── qaoa_tcs/<dataset>/rep_1/*.qpy, *_qubos.json   # circuits + their QUBO coefficients
│   └── igdec_qaoa/<dataset>/sampling_N/*.qpy, circuits_metadata.json  # + QUBO coefficients
├── results/
│   ├── qaoa_tcs/<dataset>/...        # single_obj.py / multi_obj.py output
│   ├── igdec_qaoa/<dataset>/...      # IGDec-QAOA output
│   └── evaluation/                   # evaluate_all.py / compare_all.py output (gitignored)
├── calibration/                      # MEM/M3 calibration + backend_snapshots/ (T1/T2/etc.)
├── execution_plan/                   # budget_planner.py output (regenerated on demand)
├── configs/
│   ├── execution_plan.yaml           # budget planner + evaluate_all.py configuration
│   └── backend.yaml                  # PIAST-Q API token/backend name (see §10 step 5)
└── tests/                            # synthetic unit tests (no AQT, no real hardware)
```

`src/` is the only place Python packages live; everything else is data,
configuration, or generated output. Every script in `src/` imports
`piastq_execution` as a sibling package via `sys.path.insert(0, "..")` at the
top of the file — this works regardless of which script you run because
`qaoa_tcs/`, `igdec_qaoa/`, and `piastq_execution/` all sit directly under
`src/`.

---

## 2. Requirements

- **Python 3.10.\***

```bash
python3.10 -m venv qiskit_env
source qiskit_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. The 14-combo matrix

| # | Algorithm | Objective mode | Dataset | Circuits                     | Circuits directory |
|---|-----------|-----------------|---------|------------------------------|----------------------|
| 1 | QAOA-TCS | single | gsdtsr | 79                           | `trained_qaoa_circuits/qaoa_tcs/gsdtsr/rep_1/` |
| 2 | QAOA-TCS | single | iofrol | 443                          | `trained_qaoa_circuits/qaoa_tcs/iofrol/rep_1/` |
| 3 | QAOA-TCS | single | paintcontrol | 24                           | `trained_qaoa_circuits/qaoa_tcs/paintcontrol/rep_1/` |
| 4 | QAOA-TCS | single | elevator (o2 formulation) | 881                          | `trained_qaoa_circuits/qaoa_tcs/elevator/rep_1/` |
| 5 | QAOA-TCS | single | elevator2 (o3 formulation) | 56                           | `trained_qaoa_circuits/qaoa_tcs/elevator2/rep_1/` |
| 6 | QAOA-TCS | multi | flex | 105                          | `trained_qaoa_circuits/qaoa_tcs/flex/rep_1/` |
| 7 | QAOA-TCS | multi | grep | 139                          | `trained_qaoa_circuits/qaoa_tcs/grep/rep_1/` |
| 8 | QAOA-TCS | multi | gzip | 61                           | `trained_qaoa_circuits/qaoa_tcs/gzip/rep_1/` |
| 9 | QAOA-TCS | multi | sed | 77                           | `trained_qaoa_circuits/qaoa_tcs/sed/rep_1/` |
| 10 | IGDec-QAOA | single | iofrol | 10 samplings x 1230 circuits | `trained_qaoa_circuits/igdec_qaoa/iofrol/sampling_{1..10}/` |
| 11 | IGDec-QAOA | single | paintcontrol | 10 samplings x 30 circuits   | `trained_qaoa_circuits/igdec_qaoa/paintcontrol/sampling_{1..10}/` |
| 12 | IGDec-QAOA | single | gsdtsr | **to do**                    | `trained_qaoa_circuits/igdec_qaoa/gsdtsr/` |
| 13 | IGDec-QAOA | single | elevator_o2 (elevator_two) | **to do**                    | `trained_qaoa_circuits/igdec_qaoa/elevator_two/` |
| 14 | IGDec-QAOA | single | elevator_o3 (elevator_three) | **to do**                    | `trained_qaoa_circuits/igdec_qaoa/elevator_three/` |

**Directory convention:** QAOA-TCS (single- and multi-objective alike) keeps
one *reusable* circuit set per dataset — the same circuits get re-run once
per repetition/experiment. IGDec-QAOA keeps a *distinct, non-reusable*
circuit set per random-initial-solution sampling, each with its own
`circuits_metadata.json` and `initial_random_solution.json`.

**Regenerating circuits that already exist here**: every combo's circuits
can now be regenerated from scratch by this repo alone (`train` mode, §4.1/
§4.2) — nothing is "ported-only" anymore.

- **QAOA-TCS** (`single_obj.py train` / `multi_obj.py train`): one QAOA
  training run per cluster (not iterative like IGDec-QAOA), so the compute
  cost is exactly the "Circuits" column above -- e.g. 79 QAOA-solves to
  retrain all of gsdtsr, 1,483 for all 5 single-objective datasets combined,
  1,865 for all 14 QAOA-TCS circuit sets combined. Deterministic clustering
  (not random like IGDec-QAOA), so retraining reproduces the same cluster
  structure and circuit indices every time -- but the QAOA optimization
  itself is still non-deterministic (random seed per run via
  `algorithm_globals`/COBYLA), so retrained circuits won't be bit-identical
  to the currently-ported ones even though they solve the same QUBOs.
- **IGDec-QAOA** (`loch_qaoa_tcm_extract_circuits.py train`): retrains
  whichever datasets are listed in
  `save_trained_circuits_and_initial_solutions()`'s loop (`["gsdtsr",
  "iofrol", "paintcontrol"]` by default), including iofrol/paintcontrol,
  from scratch. Since IGDec-QAOA training is stochastic (random initial
  solution each sampling), rerunning it **overwrites the existing sampling
  folders with a brand-new circuit set** — the old ones aren't kept side by
  side. Only gsdtsr gets the `rate > 0` filter (5,555 -> 287 rows);
  iofrol/paintcontrol always train unfiltered, matching how the
  already-ported ones were produced — this is applied per-dataset inside
  the loop, not by narrowing the loop itself, so it stays correct no matter
  which subset of the three you train.

---

## 4. How to run

Every script assumes you `cd` into its own directory first — that's what
makes its relative paths (`../../datasets/...`, `../../trained_qaoa_circuits/...`,
`../../results/...`) resolve correctly.

### 4.1 QAOA-TCS

Same two-mode shape as the IGDec-QAOA scripts (§4.2): plain invocation
**executes** already-trained circuits (default); `... train` **trains**
them on an ideal simulator instead.

```bash
cd src/qaoa_tcs
python single_obj.py     # EXECUTE (default): gsdtsr, iofrol, paintcontrol, elevator, elevator2
                          # -> ../../results/qaoa_tcs/<dataset>/
python multi_obj.py      # EXECUTE (default): flex, grep, gzip, sed
                          # -> ../../results/qaoa_tcs/<dataset>/

python single_obj.py train   # TRAIN: same 5 datasets, one QAOA-solve per cluster
python multi_obj.py train    # TRAIN: same 4 datasets, one QAOA-solve per cluster
                              # -> ../../trained_qaoa_circuits/qaoa_tcs/<dataset>/rep_1/
```

Training is not iterative (unlike IGDec-QAOA) — one ideal-simulator QAOA
optimization (COBYLA, 500 max evaluations) per cluster, so its cost is
exactly the "Circuits" column in §3's matrix (1,483 QAOA-solves for all 5
single-objective datasets combined, 382 for all 4 multi-objective ones).
Per-dataset alpha (`bootqa_alphas` in `single_obj.py`; `alpha = 0.5` in
`multi_obj.py`) matches what the already-ported circuits were trained with
(ported from `SelectQAOA/MOQ-Pipeline.ipynb`'s training cells). **Do not run
this training in an authoring/review session; run it on the machine that
will actually spend that compute**, same as IGDec-QAOA's training (§4.2).

**Reconfiguring scope**: edit `bootqa_programs` (single_obj.py) or
`sir_programs` (multi_obj.py) at the top of each file to change which
datasets run (applies to both execute and train mode). Only `rep_1` is
trained/executed by default — see `TRAINING_REPS` near the top of each
script's `save_trained_circuits()` if you need `rep_2`/`4`/`8`/`16` too
(matching the original SelectQAOA study).

### 4.2 IGDec-QAOA

The three `loch_qaoa_*_extract_circuits.py` script names are inherited from
the original SelectQAOA source and are slightly misleading: **each one does
two separate jobs**, selected by how you invoke it — plain
`python <script>.py` runs the *default* mode, which is hardware-like
**execution** of already-trained circuits; `python <script>.py train` runs
ideal-simulator **training** (extraction) instead. Nothing runs both in one
invocation.

```bash
cd src/igdec_qaoa
python loch_qaoa_tcm_extract_circuits.py            # EXECUTE (default mode): iofrol, gsdtsr, paintcontrol
python loch_qaoa_elev_two_extract_circuits.py       # EXECUTE (default mode): elevator_two
python loch_qaoa_elev_three_extract_circuits.py     # EXECUTE (default mode): elevator_three
# -> ../../results/igdec_qaoa/<dataset>.json + <dataset>-raw_counts.jsonl
```

**Training gsdtsr / elevator_two / elevator_three circuits** (not yet
trained anywhere — do this on the other machine, not here):

```bash
python loch_qaoa_tcm_extract_circuits.py train           # by default: gsdtsr, iofrol, paintcontrol
python loch_qaoa_elev_two_extract_circuits.py train       # elevator_two (elevator_o2)
python loch_qaoa_elev_three_extract_circuits.py train     # elevator_three (elevator_o3)
```

Each trains 10 independent random-initial-solution samplings x 10
impact-reordering decomposition iterations, via ideal-simulator QAOA
(COBYLA(500) + statevector sampler) — no hardware/AQT involved in training
itself. Rough compute estimates (subproblem QAOA-solves = samplings x
iterations x subproblems-per-iteration, where the last term is capped by the
15%-of-dataset impact window used by the decomposition):

| Dataset | Rows used | Subproblems/iteration | Total subproblem-solves |
|---------|-----------|------------------------|--------------------------|
| gsdtsr | 287 (filtered `rate > 0` out of 5,555) | 6 | ~600 |
| iofrol | 1,941 (unfiltered) | 41 | ~4,100 |
| paintcontrol | 89 (unfiltered) | 1 | ~100 |
| elevator_two | 1,925 (unfiltered) | 41 | ~4,100 |
| elevator_three | 1,925 (unfiltered) | 41 | ~4,100 |

`loch_qaoa_tcm_extract_circuits.py`'s iofrol/paintcontrol rows only apply if
you actually retrain them — see the "regenerating circuits" note in §3 for
why that overwrites the already-ported sampling folders.

Each subproblem-solve is a 7-qubit ideal-simulator QAOA optimization
(COBYLA, 500 max evaluations) — likely on the order of a few hours
wall-clock total per script invocation, depending on the machine. **Do not
run this training in an authoring/review session; run it on the machine
that will actually spend that compute.**

**Reproducible from a clean checkout**: each sampling's initial random
solution and QAOA optimizer randomness are seeded deterministically from
`(dataset, sampling_id)` via `piastq_execution.seeding.seed_for_sampling()`
— re-running `... train` after deleting `trained_qaoa_circuits/igdec_qaoa/`
reproduces the exact same 10 samplings every time. (This was not always
true: an earlier version of these scripts had a `run_alg()` function that
*looked* like it seeded things but was dead code, never called; the actual
training path was fully unseeded. Fixed, and `run_alg()` removed.)

Both `single_obj.py`/`multi_obj.py` and all three IGDec-QAOA scripts connect
via `piastq_execution.backend_config.get_backend()`, which reads the API
token/backend name from `configs/backend.yaml` (currently the placeholder
`offline_simulator_no_noise`) — see §10 step 5's note for how to point this
at real PIAST-Q hardware from that one file. **Run every hardware-execution
entry point (everything above except `... train`) on the machine that
actually has PIAST-Q/AQT access.**

---

## 5. Mitigation: how each method actually executes

`src/piastq_execution/mitigation.py` implements all three methods
identically regardless of algorithm/objective-mode/dataset — mitigation
operates purely on per-circuit counts, so the same functions apply to every
one of the 14 combos.

### Raw (baseline)

No mitigation. This is what every execution tail already produces by
default: `run_circuit_with_batching_recorded()` runs the trained circuit
once, argmax-selects the most likely bitstring per cluster, same as before —
now with the *full* counts distribution also saved to
`*-raw_counts.jsonl` (§6) so it can be corrected later without re-running
hardware.

### Full MEM

1. **Calibrate once per circuit width** (or per exact physical-qubit
   mapping, see below), on the same backend, before/alongside the QAOA runs:

   ```bash
   cd src/piastq_execution
   python run_calibration.py           # calibrates widths 1..7 (default)
   python run_calibration.py 7         # calibrate only width 7
   ```

   This builds `2**n` basis-state preparation circuits per width
   (`build_mem_calibration_circuits`), runs each through the same
   `AQTSampler`/backend every other script uses, builds the `2**n x 2**n`
   confusion matrix (`build_confusion_matrix`), and saves it under
   `calibration/mem_q<physical-qubits>.json` via `CalibrationStore` —
   tagged with a timestamp, backend snapshot, and the *measured* wall-clock
   time every calibration circuit's execution actually took
   (`calibration_wall_clock_seconds`, timed in `run_calibration.py`, not
   estimated — see §8's mitigation-overhead note).

2. **Correct any raw-counts entry** against that stored calibration,
   whenever you want (no hardware needed for this step):

   ```python
   from piastq_execution.mitigation import CalibrationStore, correct_counts

   store = CalibrationStore()
   calibration = store.load("mem", physical_qubits=[0, 1, 2, 3, 4, 5, 6])
   corrected_distribution = correct_counts(
       "mem", raw_counts, calibration, num_qubits=7
   )
   ```

   `correct_counts_with_matrix()` underneath uses non-negative least squares
   (NNLS) so the corrected distribution stays valid (non-negative,
   normalized to 1) — see `piastq_execution/evaluation.py` for how this
   feeds into the actual per-combo metrics.

### M3

Same two-step shape as MEM, but with `2*n` (not `2**n`) calibration
circuits per width — `run_calibration.py` runs both MEM and M3 calibration
in the same pass. Correction goes through the `mthree` package
(`M3Mitigation(system=None)` + `cals_from_matrices()` — confirmed to work
fully standalone, no IBM backend object needed). If `mthree` isn't
installed, `_correct_counts_m3_reduced()` is used instead — it restricts the
assignment matrix to just the bitstrings observed in that circuit's raw
counts (typically far fewer than `2**n` for a QAOA output distribution
concentrated on a handful of low-energy states) rather than building the
full `2**n x 2**n` matrix, mirroring how real M3 stays sub-exponential by
restricting correction to the noisy distribution's support instead of every
possible bitstring:

```python
calibration = store.load("m3", physical_qubits=[0, 1, 2, 3, 4, 5, 6])
corrected_distribution = correct_counts("m3", raw_counts, calibration, num_qubits=7)
```

### TREx (measurement twirling)

Unlike MEM/M3, TREx has **no separate calibration step**, and — unlike
MEM/M3 — it's actually wired into every execution script, not just available
as library functions: `single_obj.py` and `multi_obj.py`'s
`run_hardware_execution()`, and the three IGDec-QAOA single-objective
scripts' `run_hardware_like_from_saved_circuits()`, each run a second,
dedicated hardware pass per circuit, right after the raw pass — **but only
if the resolved plan for that combo actually kept TREx** (see §7's
`resolve_all_combo_settings()`); if the plan dropped it, the loop below is
skipped entirely rather than silently running anyway and blowing past the
budget it was computed for:

```python
if settings.trex_enabled:
    for twirl_idx in range(trex_twirl_instances):
        twirl_mask = generate_random_twirl_mask(circuit.num_qubits, rng=trex_rng)
        twirled_circuit = build_trex_twirled_circuit(circuit, twirl_mask)
        _trex_counts, trex_record = run_circuit_with_batching_recorded(
            twirled_circuit, sampling_sampler, algorithm=..., objective_mode=..., dataset=...,
            circuit_id=f"{circuit_id}_trex{twirl_idx}", backend=backend,
            cluster_id=..., iteration_id=...,
            shots_per_batch=settings.shots_per_batch, num_batches=settings.num_batches,
            remainder_shots=settings.remainder_shots, twirl_mask=twirl_mask,
        )
        trex_counts_writer.write(trex_record)
```

Each twirl instance is a distinct hardware execution — this is what "TREx
can't reuse raw counts, unlike MEM/M3" means in practice — written to a
**separate file**, `*-trex-raw_counts.jsonl`, alongside (not mixed into) the
existing `*-raw_counts.jsonl`. `RawCountsRecord` carries a `twirl_mask`
field (set only on these records) so the mask travels with its counts.

`trex_twirl_instances` (how many twirl instances per circuit — more
instances means better averaging of the readout bias, at the cost of more
hardware time) is **configurable in one place**: `trex_twirl_instances` in
`configs/execution_plan.yaml` (default `32`, matching the commonly used
default for measurement-twirling randomizations in Qiskit Runtime's
twirled-readout implementations — see
`piastq_execution.mitigation.load_trex_twirl_instances()`). Both the
execution scripts and `budget_planner.py`'s cost model (§7) read this same
value, so the plan and the actual run always agree on TREx's cost.

At evaluation time, `piastq_execution.mitigation.aggregate_trex_records()`
groups a combo's TREx records by `(cluster_id, iteration_id, subproblem_id)`
and undoes each group's twirls (`aggregate_trex_instances()` underneath),
producing the per-circuit aggregated counts dict that
`evaluate_single_objective_combo(method="trex", cluster_raw_counts=..., ...)`
(§8) expects — `correct_counts("trex", ...)` then just normalizes it, the
same as the raw baseline.

Full Pauli twirling of the circuit itself (not just before measurement) is a
materially larger follow-up and is not implemented here — see the module
docstring in `mitigation.py`.

### Physical qubit layout: pinned, and shared by calibration and execution

Before this round, neither the real QAOA/TREx circuit execution
(`raw_counts.run_circuit_with_batching_recorded`) nor the MEM/M3 calibration
circuits (`run_calibration.py`) pinned an explicit transpiler layout — both
let the transpiler pick physical qubits automatically
(`optimization_level=3`, no `initial_layout`), independently of each other.
Nothing guaranteed a calibration built from one set of physical qubits would
ever be applied to counts collected from the *same* physical qubits.
`run_calibration.py`'s own former docstring flagged this explicitly as a
known simplification, not a guarantee.

Fixed via `piastq_execution/qubit_layout.py`: a width-`w` circuit always uses
`configs/backend.yaml`'s `physical_qubits` list, sliced to its first `w`
entries (`physical_layout_for_width(w)`) — the exact same physical qubits,
every time, from every caller:

- `raw_counts.run_circuit_with_batching_recorded()` now accepts an
  `initial_layout` argument, applies it to the sampler via
  `set_transpile_options(initial_layout=..., optimization_level=...)`
  *before* the circuit is submitted (not just recorded afterwards), and
  derives the recorded `physical_qubit_mapping` directly from it
  (`physical_qubit_mapping_from_layout`) — no transpile-based guess
  involved anymore when a layout is pinned. All five execution scripts' raw
  and TREx-twirl passes now pass
  `initial_layout=physical_layout_for_width(circuit.num_qubits)`.
- `run_calibration.py`'s `calibrate_mem`/`calibrate_m3` pin the identical
  layout for every calibration circuit of a given width, instead of the
  previous `list(range(width))` placeholder.

Verified end-to-end against the real `AQTSampler`/`offline_simulator_no_noise`
stack (not just synthetic tests): a real execution and a real MEM/M3
calibration circuit, both pinned to a deliberately non-trivial layout
(`[3, 4]` on a 20-qubit backend), land on exactly the same physical qubits —
confirmed both from the recorded metadata and independently from the
transpiler's own resulting layout.

To point this at PIAST-Q's real qubit numbering (or dedicate specific
physical qubits to this study), edit only `configs/backend.yaml`'s
`physical_qubits` list — defaults to `[0, 1, 2, 3, 4, 5, 6]` if absent.

### Backend calibration snapshots (T1/T2/etc.)

Distinct from the MEM/M3 calibration above (which we measure ourselves from
shots we run), `piastq_execution/backend_calibration_snapshot.py` captures
whatever the *provider itself* reports about its qubits — T1, T2,
frequency, via the standard `BackendV2.qubit_properties` hook (AQT's
`AQTResource` exposes this; exactly how much PIAST-Q populates it isn't
knowable until real hardware is used, so every field is extracted
defensively and recorded as `None` when unavailable, never raised).

`save_backend_calibration_snapshot(backend)` is called automatically, once
per script invocation, at the top of every hardware-execution entry point
(all five execution scripts' hardware-execution function, plus
`run_calibration.py`) — right after `get_backend()`, before anything else
happens. Each call writes its own new, timestamped file under
`calibration/backend_snapshots/` (never overwrites a previous one), so
calibration drift across however many real-hardware sessions the full data
collection ends up taking can be checked after the fact, without anyone
having to remember to trigger it by hand.

---

## 6. Raw-counts persistence

Every execution tail uses the shared
`piastq_execution/raw_counts.py: run_circuit_with_batching_recorded()`
instead of collapsing every circuit's result down to a single argmax
bitstring. Alongside each combo's existing derived outputs (unchanged format:
`*-rep-N.json`, `*-subsuites.json`, `pareto_front_N`, `<dataset>.json`), a
`*-raw_counts.jsonl` file is written with, per circuit per shot-batch: the
full raw counts dict, circuit/cluster/(iteration+subproblem) id,
algorithm/objective_mode/dataset, shots requested vs. returned, physical
qubit mapping, backend name/version/timestamp, and wall-clock time.
Every combo (QAOA-TCS single- and multi-objective, and IGDec-QAOA
single-objective) also writes a second, separate `*-trex-raw_counts.jsonl`
(§5's TREx section) — same record shape, plus a populated `twirl_mask` field
on every row. These `.jsonl` dumps are gitignored since they're large,
per-run hardware artifacts, not authored content.

---

## 7. Budget planning

`src/piastq_execution/budget_planner.py` turns a per-pool hardware-time
budget into a concrete execution plan (repetitions, shots/circuit, which
mitigation methods survive) by reading the *actual* circuit counts under
`trained_qaoa_circuits/` — nothing is hardcoded.

```bash
cd src/piastq_execution
python budget_planner.py
# -> prints a report and writes it to ../../execution_plan/
```

**The plan is actually wired into execution, not just printed.**
`resolve_all_combo_settings()` (same module) runs the full planner once and
returns, per combo, exactly what a hardware-execution script should run:
`repetitions`, `shots_per_circuit` (already split into
`num_batches`/`shots_per_batch`/`remainder_shots` — the "N x 200 +
remainder" pattern every circuit is actually submitted in), `methods`, and
`trex_enabled`. Every one of the five execution scripts calls this once at
the top of its hardware-execution function and uses the resolved values
instead of a hardcoded placeholder:

```python
combo_settings = resolve_all_combo_settings()
...
settings = combo_settings[f"{dataset}_qaoa_tcs"]   # (or _igdec_qaoa)
num_piast_experiments = settings.repetitions
...
counts, raw_record = run_circuit_with_batching_recorded(
    circuit, sampling_sampler, ...,
    shots_per_batch=settings.shots_per_batch,
    num_batches=settings.num_batches,
    remainder_shots=settings.remainder_shots,
)
...
if settings.trex_enabled:
    for twirl_idx in range(trex_twirl_instances):
        ...
```

Before this, every script hardcoded `shots_per_batch=80, num_batches=1` at
every call site and always ran the TREx loop regardless of what the plan
said — a real disconnect between the plan and what would actually execute.
Fixed now: whatever `budget_planner.py` decides for a pool is exactly what
the corresponding execution script runs, including dropping TREx entirely
for a combo (not just computing that it *should* be dropped). Note this
doesn't touch the QAOA circuit-depth `p` parameter (`TRAINING_REPS`/`reps`
in the training loop) or the impact-reordering iteration count in IGDec-QAOA
(`NUM_ITERATIONS`/`num_impact_iterations`) — both stay exactly as inherited
from the original study; only the hardware-execution repetition count and
shot batching are resolved from the plan.

**Default configuration is *not* one global 15h run.** The shipped
`configs/execution_plan.yaml` defines **7 pools x 15h = 105h total budget**:
one pool for QAOA-TCS single-objective (5 combos), one for QAOA-TCS
multi-objective (4 combos), and one pool *per dataset* for IGDec-QAOA (5
pools of 1 combo each). Each pool's 15h is independent — a pool only budgets
for the combos listed in its own `combos:` list. To actually get a single
15h run covering everything, replace the whole `pools:` section with one
entry:

```yaml
pools:
  - name: global
    total_hours: 15
    combos: [gsdtsr_qaoa_tcs, iofrol_qaoa_tcs, paintcontrol_qaoa_tcs, elevator_qaoa_tcs, elevator2_qaoa_tcs,
             flex_qaoa_tcs, grep_qaoa_tcs, gzip_qaoa_tcs, sed_qaoa_tcs,
             gsdtsr_igdec_qaoa, iofrol_igdec_qaoa, paintcontrol_igdec_qaoa, elevator_o2_igdec_qaoa, elevator_o3_igdec_qaoa]
```

> **This single-pool example will almost certainly drop TREx for every
> combo.** Verified by actually running `budget_planner.py` against this
> exact config with the real circuit inventory: at 15h total for all 14
> combos, the planner reduces repetitions to 1, shots/circuit to the 200
> floor, and still drops TREx everywhere just to fit (`trex_twirl_instances`
> extra hardware passes per circuit, 32 by default, is expensive — see §5).
> The 7-pool default has the same problem at 15h/pool: every combo that
> already has real trained circuits on disk (`iofrol_igdec_qaoa`,
> `paintcontrol_igdec_qaoa`) has TREx dropped there too; the pools that
> currently show TREx "surviving" (`gsdtsr_igdec_qaoa`,
> `elevator_o2_igdec_qaoa`, `elevator_o3_igdec_qaoa`) only do so because
> those combos have no trained circuits generated yet in this environment
> (§4.2's `... train` step) — an empty inventory costs 0h regardless of
> method, which is a false positive, not a real budget win. Once you
> generate those circuits, expect TREx to be dropped there too at 15h. If
> you need TREx data for a specific combo, give it (or a small group of
> combos) its own pool with a larger `total_hours`, or lower
> `trex_twirl_instances` for that run — don't rely on a 15h pool to preserve
> it once real circuits are in place.

**Reconfiguring the plan**: edit `configs/execution_plan.yaml`.

- `combos:` — one entry per combo: `algorithm` (`qaoa_tcs` | `igdec_qaoa`),
  `objective_mode`, `dataset`, `circuits_dir` (the folder name under
  `trained_qaoa_circuits/qaoa_tcs/` or `trained_qaoa_circuits/igdec_qaoa/`),
  and `methods` (any subset of `raw`/`mem`/`m3`/`trex`).
- `pools:` — one entry per time budget: `name`, `total_hours`, and `combos`
  (a list of combo names from above). A pool covering every combo is a
  "global" setup; 14 pools of one combo each is "per_combo"; any mix (the
  shipped example: one pool per objective-mode for QAOA-TCS, one pool per
  dataset for IGDec-QAOA) is supported directly — just change which combos
  go in which pool.
- `seconds_per_batch`, `shots_per_batch_cap`, `target_shots_per_circuit`,
  `min_shots_per_circuit`, `target_repetitions`,
  `calibration_shots_per_circuit`, `trex_twirl_instances` — top-level knobs
  shared by every pool (`trex_twirl_instances` is also read directly by the
  execution scripts' TREx loop, see §5, so the plan and the actual run never
  disagree on TREx's cost).

Degradation priority when a pool's budget is too small for its target
repetitions/shots/methods: **(1)** drop TREx if that alone is enough, else
drop it anyway since it's the most expensive method, **(2)** reduce
repetitions, **(3)** reduce shots/circuit (floor: one 200-shot batch), **(4)**
as a last resort, subsample circuits/clusters (reported explicitly, per
circuit, with the reason). Full MEM/M3 calibration circuits are counted once
per distinct circuit width used within a pool, shared across every
combo/repetition using that width; TREx always costs `trex_twirl_instances`
extra hardware passes per circuit, not one (see §5) — dropping it is
correspondingly the single biggest lever the planner has.

**The underlying cost model, in one formula**: every circuit costs
`ceil(shots / 200) × 15 seconds` — 201 shots costs exactly as much as 400,
since it rounds up to a second whole 200-shot batch. Raw cost for a combo =
(circuits × repetitions) × per-circuit cost; TREx cost = that same raw cost
again, × `trex_twirl_instances`; MEM/M3 calibration is charged once per
distinct circuit width present in the pool, shared across every combo using
it, never per combo. Everything above is degradation applied to fit that sum
under a pool's hours.

**Is there a "no-TREx" mode?** No dedicated toggle — TREx inclusion is
controlled entirely by whether `"trex"` appears in each combo's `methods:`
list in `configs/execution_plan.yaml` (every combo lists it by default). The
planner always attempts whatever's listed at full target quality first and
automatically drops TREx — before touching repetitions or shots — if the
pool can't afford it, so in practice it already behaves as "always included,
shut off automatically once over budget." Forcing "no TREx everywhere" in
one shot means editing every combo's `methods:` list by hand; there's no
flag for it today.

This only reads local `.qpy` files (pure deserialization) and writes to
`execution_plan/` — no backend, no AQT, safe to run anywhere, including in an
authoring/review session.

### TREx timing: what a conference-scope subset actually costs

TREx is comfortably the most expensive method in this whole pipeline —
every included combo needs `trex_twirl_instances` (32 by default) *extra*
hardware passes per circuit on top of the raw baseline, since it cannot
reuse already-collected counts (§5). The example above already shows this in
practice: the shipped 7-pool config drops TREx for every pool that actually
has trained circuits, even after cutting repetitions to 1 and shots to the
floor.

To make an informed decision about where TREx is actually worth spending
real machine time on, here is the *real*, code-computed cost (via
`budget_planner._pool_seconds`, not hand arithmetic) of a full TREx pass per
combo — one repetition, the 200-shot floor, `trex_twirl_instances=32` — for
every combo that currently has trained circuits:

| Combo | Circuits (rep=1) | Hours (TREx incl., floor shots) | Hours (TREx incl., target 2048 shots) |
|---|---|---|---|
| paintcontrol (QAOA-TCS) | 24 | 3.3h | 36.3h |
| paintcontrol (IGDec-QAOA) | 30 | 4.1h | 45.4h |
| elevator2 (QAOA-TCS) | 56 | 7.7h | 84.7h |
| gzip (multi-obj) | 61 | 8.4h | 92.3h |
| sed (multi-obj) | 77 | 10.6h | 116.5h |
| gsdtsr (QAOA-TCS) | 79 | 10.9h | 119.5h |
| flex (multi-obj) | 105 | 14.4h | 158.8h |
| grep (multi-obj) | 139 | 19.1h | 210.2h |
| iofrol (QAOA-TCS) | 443 | 60.9h | 670.0h |
| elevator (QAOA-TCS) | 881 | 121.1h | 1332.5h |
| iofrol (IGDec-QAOA) | 1230 | 169.1h | 1860.4h |

**Reading this**: at the shot floor, six combos fit a full TREx pass inside
a single ~15h overnight session each (paintcontrol ×2, elevator2, gzip,
gsdtsr, sed — roughly 45h combined), `grep` is borderline (~19h, just over
one night), and the three large combos (`iofrol` under both algorithms,
`elevator`) are not realistic for TREx without either many dedicated
overnight sessions or a substantially reduced `trex_twirl_instances`.

**A realistic conference-scope TREx subset**, given this: the six
comfortably-fitting combos above, run as their own dedicated pool(s) in
`configs/execution_plan.yaml` with enough hours to actually keep TREx
(rather than the shipped default, which spreads a fixed 15h thin across
every combo and drops TREx everywhere real circuits exist). `iofrol` and
`elevator` remain a journal-scale ask: either several dedicated overnight
sessions, or cutting `trex_twirl_instances` well below 32 for those combos
specifically (e.g. 8 instead of 32 would roughly quarter these numbers, at
the cost of a noisier average over fewer twirl instances).

**One more thing worth knowing before committing to a subset**: while
wiring up the qubit-layout fix above, we found that `paintcontrol`'s path
through `loch_qaoa_tcm_extract_circuits.py` (the "small dataset"
decomposition branch, `problem_size <= 0.15 * len(df)`) never runs a TREx
pass at all — only the "large dataset" branch does. `iofrol` and `gsdtsr` go
through the large-dataset branch and are unaffected; `paintcontrol` is the
only tcm dataset that hits the small-dataset branch, so **IGDec-QAOA's
`paintcontrol` combo cannot produce TREx data today, regardless of budget**,
until that gap is closed. Not fixed as part of this round (out of scope for
the qubit-layout/calibration-snapshot work above) — worth deciding on
separately before relying on it for the subset.

No code changes were made for this section — it's planning information
only, using the real, already-implemented cost model.

**One more data point, computed the same way**: all 11 combos *together*,
at 1 repetition and the 200-shot floor with no TREx anywhere, cost only
14.31h — comfortably under one night, with nothing needing to be dropped.
The real constraint within a 15h or 24h budget isn't which combos fit, it's
how much of that budget is left over to spend on TREx for a few of the
cheapest ones — see `EXPLAINATION.md` §10.2/§10.3 for the full breakdown
(every single combo and every meaningful combination's full-target hours,
plus the exact 15h/24h answer).

---

## 8. Evaluation and statistics

`src/piastq_execution/evaluation.py` computes, per combo x method
(raw/MEM/M3/TREx), branching on `objective_mode`:

- **Single-objective**: QUBO energy of the corrected/selected solution,
  probability of the brute-forced optimal bitstring (<=7-qubit subproblems,
  128 states, trivial), execution cost + the dataset's effectiveness
  metric(s) of the final merged suite (`EFFECTIVENESS_METRICS`: failure rate
  for gsdtsr/iofrol/paintcontrol; input diversity for QAOA-TCS's own
  `elevator`; passenger count + travel distance for QAOA-TCS's `elevator2`
  *and* IGDec-QAOA's `elevator_o2`/`elevator_o3` alike — confirmed by diffing
  the `elevator_two`/`elevator_three` scripts to use the identical
  cost+pcount+dist QUBO formulation; `elevator_o2` was previously
  mislabeled `input_diversity`, a dataset name mismatch that would have
  raised `Unknown dataset` for QAOA-TCS's own `elevator`/`elevator2` too,
  since neither was registered at all), quantum-hardware execution time,
  mitigation overhead, classical post-processing time.
- **Multi-objective**: non-dominated solutions contributed to an a posteriori
  reference Pareto frontier (union of every compared method's non-dominated
  points), Hypervolume, IGD, plus the same execution-time/overhead/
  post-processing metrics as single-objective.

**The multi-objective QUBO -- and its HV/IGD/Pareto evaluation -- has
*three* objectives, not two: execution cost (minimize), statement coverage
(maximize), fault coverage (maximize).** `make_linear_terms()`'s linear term
trades off cost against fault coverage; `make_quadratic_terms()`'s quadratic
penalty pushes toward statement-coverage diversity. This is exactly the
`(total_cost, total_coverage, total_faults)` / `pareto_dominance()` triple
`SelectQAOA/MOQ-Pipeline.ipynb` itself uses for HV/IGD. `build_pareto_front()`
in `multi_obj.py` was corrected to match: it now computes all three per
growing-prefix candidate and filters via true 3-objective non-domination
(`piastq_execution.statistics.pareto_front_indices()`, cost negated to
`-execution_cost` so "maximize every objective" applies uniformly) — it
previously only tracked fault/statement coverage (missing cost as a
Pareto dimension entirely) and had a real bug in how it accumulated
statement coverage (summed `covered_lines()` once *per test case* in the
candidate instead of once per candidate, inflating it by suite size).
`evaluate_multi_objective_combo()`'s `pareto_points`/`all_methods_points`/
`reference_point` are these same 3-tuples; `hypervolume()`
(`piastq_execution/statistics.py`) is a dimension-agnostic exact recursive
implementation (not the old rectangle-sweep `hypervolume_2d`, kept only as
a thin 2D-specific wrapper for other 2D uses), so nothing besides these two
functions' inputs needed to change to go from 2 to 3 objectives.

So `execution_cost` (single-objective's own metric — sum of the selected
tests' `time`/`cost` column, one of that QUBO's objective terms alongside
effectiveness) has no `MultiObjectiveEvaluation` field of its own for a
different reason than before: multi-objective's cost is already *inside*
the 3-objective Pareto/HV/IGD analysis (as `-execution_cost`, the first
coordinate of every point), not absent from it.

*Quantum-hardware* execution time (`execution_time_seconds`) is a separate,
**always-present metric on both dataclasses** — not to be confused with the
QUBO's execution-cost objective above: for one experiment repetition, it's
the sum of the wall-clock time spent executing each subproblem's QAOA
circuit (`compute_execution_time_seconds()`, one `RawCountsRecord` per
subproblem/cluster) — computed identically for all three pipelines this
module serves (QAOA-TCS single-objective, QAOA-TCS multi-objective,
IGDec-QAOA single-objective). For TREx, pass in every twirl instance's
`RawCountsRecord`, not just one per subproblem — the sum naturally reflects
TREx's extra hardware passes. `mitigation_overhead` (`.total_shots`,
`.calibration_circuits`, `.calibration_wall_clock_seconds`) tracks the cost
of mitigation itself (MEM/M3 calibration — zero for raw/TREx, which don't
use a calibration record), separate from QAOA execution time.
`calibration_wall_clock_seconds` is *measured*, not estimated: it comes
straight from `CalibrationRecord.calibration_wall_clock_seconds`, which
`run_calibration.py` fills in by timing every calibration circuit's actual
`sampler.run()` call (§5) — this is what lets you compare mitigation methods
by cost/benefit (RQ3b, below), not just by how much they improve quality.
`classical_post_processing_seconds` is the (non-hardware) correction +
metric-computation time.

### IGDec-QAOA's final-suite metrics: raw only

`evaluate_single_objective_combo()`'s built-in merge
(`merge_selected_tests()`, union across independent clusters) is correct for
QAOA-TCS, whose clusters never share a test case. IGDec-QAOA's subproblems
are **adaptive** instead: `solution[case_list[i]] = bitstring[i]` overwrites
specific positions of an evolving global solution, and which subproblem gets
solved next depends on that evolving solution — a saved circuit sequence
only reflects the *one* trajectory the raw hardware run actually took.
Feeding a corrected (MEM/M3/TREx) method's per-circuit selections through
the same trajectory would silently pretend IGDec-QAOA made the same
adaptive decisions under a different correction method, which isn't true
and isn't verified anywhere.

So for IGDec-QAOA combos: `execution_cost`/`effectiveness` are only ever
computed for `raw` (via `evaluate_all.py`'s `merge_igdec_solution()`, which
replays the real last-write-wins update rule instead of
`merge_selected_tests()`'s union) — MEM/M3/TREx get `execution_cost=None`,
`effectiveness={}`. `qubo_energy`/`probability_of_optimal` remain valid for
every method regardless (they're scored per circuit against its own fixed
QUBO, independent of the merge).

### Running the evaluation + comparisons

**Automatic, across every combo** (the intended way to run this):

```bash
cd src/piastq_execution
python evaluate_all.py   # -> ../../results/evaluation/<combo>.json
python compare_all.py    # -> ../../results/evaluation/comparison_report.{txt,json}
```

`evaluate_all.py` reads `configs/execution_plan.yaml`'s `combos:` section
(the same combo list `budget_planner.py` uses) and, for every combo x method
x repetition, assembles everything `evaluate_single_objective_combo()` /
`evaluate_multi_objective_combo()` needs — trained circuits' persisted QUBO
coefficients (`*_qubos.json` / extended `circuits_metadata.json`, §4),
`*-raw_counts.jsonl` / `*-trex-raw_counts.jsonl`, `*-subsuites.json`, and
calibration (looked up per circuit width, §5) — and writes one JSON file per
combo. `compare_all.py` reads those files and runs `compare_groups()` for
every combo (method vs `raw`) and, for datasets with both a QAOA-TCS and an
IGDec-QAOA combo (gsdtsr/iofrol/paintcontrol — the elevator variants use
different naming per algorithm and aren't auto-paired), QAOA-TCS vs
IGDec-QAOA — writing an RQ-mapped report:

```
Combo: gsdtsr_qaoa_tcs
  RQ1/RQ3a (quality, cost-agnostic):
    qubo_energy:
      normal=False, omnibus=kruskal_wallis, p=...
        mem vs raw: p_adj=..., a12=...
  RQ2 (effectiveness):
    execution_cost: ...
  RQ3b (cost):
    execution_time_seconds: ...
    mitigation_overhead.calibration_wall_clock_seconds: ...

RQ4: QAOA-TCS vs IGDec-QAOA
  gsdtsr (gsdtsr_qaoa_tcs vs gsdtsr_igdec_qaoa), metric=execution_time_seconds:
    method=raw: ...
```

Both scripts only read already-collected files and do classical
post-processing — no backend, no AQT, safe to run anywhere once Phase 4
(calibration) and Phase 5 (hardware execution) data exists (§10).

**Manual, one comparison at a time** (if you want more control, or aren't
going through `configs/execution_plan.yaml`'s combo list): call
`evaluate_single_objective_combo()` / `evaluate_multi_objective_combo()`
yourself per repetition, pull out the metric you want with
`piastq_execution.evaluation.extract_metric_samples()` (accepts dotted paths
like `"mitigation_overhead.calibration_wall_clock_seconds"`), and hand the
groups to `piastq_execution.statistics.compare_groups()` — this is exactly
what `evaluate_all.py`/`compare_all.py` do internally, just wired up by hand:

```python
from piastq_execution.evaluation import extract_metric_samples
from piastq_execution.statistics import compare_groups

groups = {
    "qaoa_tcs": extract_metric_samples(qaoa_tcs_results, "execution_time_seconds"),
    "igdec_qaoa": extract_metric_samples(igdec_results, "execution_time_seconds"),
}
comparison = compare_groups(groups)

print(comparison.normal, comparison.omnibus_test, comparison.omnibus_p_value)
for pair in comparison.pairwise:
    print(pair.group_a, "vs", pair.group_b, "p_adj=", pair.p_adjusted,
          pair.effect_size_name, "=", pair.effect_size)
```

`comparison.normal` tells you which path was taken (Shapiro-Wilk gated);
`comparison.omnibus_test`/`.omnibus_statistic`/`.omnibus_p_value` is the
ANOVA/Kruskal-Wallis result; `comparison.pairwise` is a list of
`PairwiseComparison(group_a, group_b, p_value, p_adjusted, effect_size,
effect_size_name)` — one entry per pair of groups, already
Tukey/Bonferroni-adjusted, with Cohen's d or Vargha-Delaney A12 depending on
which path was taken. `compare_groups()` also works directly on plain
`{group_name: [values]}` dicts (2 groups: use `compare_two_groups(x, y)`
instead for the two-sample special case).

---

## 9. Tests

```bash
./qiskit_env/bin/python -m unittest discover -s tests
```

Every test is synthetic (hand-built 1-3 qubit toy circuits/QUBOs, fake
samplers, tiny Pareto fronts) — none of them call `AQTProvider`/`AQTSampler`,
run a QAOA training loop, or touch the real trained circuits. They run in
about a second (173 tests total).

---

## 10. Quickstart: clone to results (PyCharm)

1. **Clone**: PyCharm -> `Get from VCS` -> this repo's URL.
2. **Interpreter**: `Settings -> Project -> Python Interpreter -> Add ->
   Virtualenv Environment`, base interpreter Python 3.10, location
   `qiskit_env/` inside the project (matches §2's `python3.10 -m venv
   qiskit_env`). Then `pip install -r requirements.txt` in PyCharm's terminal.
3. **Sanity check (safe anywhere)**: run `python -m unittest discover -s
   tests` — should show 173 passing tests in ~1s.
4. **Plan the budget (safe anywhere)**: edit `configs/execution_plan.yaml`
   (§7), then run `src/piastq_execution/budget_planner.py` to see the
   resulting repetitions/shots/methods per pool before spending any hardware
   time.
5. **On the machine with PIAST-Q/AQT access** — everything below writes
   under `results/` and `trained_qaoa_circuits/`:
   a. **Switch from the placeholder to real PIAST-Q first** — see the note
      below.
   b. `... train` (§4.1/§4.2) for any combo you don't already have circuits
      for, or want to regenerate (all 14 combos are trainable from this
      repo).
   c. `python single_obj.py` / `multi_obj.py` (§4.1) and the IGDec scripts'
      default mode (§4.2) — hardware execution, produces raw counts
      (`*-raw_counts.jsonl`) and the existing derived outputs. Every one of
      these five scripts (`single_obj.py`, `multi_obj.py`, and the three
      `loch_qaoa_*_extract_circuits.py`) also automatically runs the TREx
      twirl-instance pass (§5) and writes `*-trex-raw_counts.jsonl` — no
      separate step needed, but budget for it (§7).
   d. `run_calibration.py` (§5) — MEM/M3 calibration, once per circuit
      width in use, stored under `calibration/`. Not needed for TREx (no
      calibration step).

   > **Switching from the placeholder to real PIAST-Q**: every execution
   > entry point gets its API token and backend name from one place —
   > `configs/backend.yaml`, read by `piastq_execution.backend_config
   > .get_backend()` — so there is exactly **one file to edit**, not six:
   > ```yaml
   > api_token: <your real API key>
   > backend_name: <piast-q backend name>
   > ```
   > (Alternatively, set the `PIASTQ_API_TOKEN` / `PIASTQ_BACKEND_NAME`
   > environment variables — they take precedence over the YAML file, so a
   > real token never has to be committed.)
   >
   > Every script that opens a backend calls `get_backend()` instead of
   > constructing `AQTProvider(...)` itself:
   > - `src/qaoa_tcs/single_obj.py` (`run_hardware_execution()`)
   > - `src/qaoa_tcs/multi_obj.py` (`run_hardware_execution()`)
   > - `src/igdec_qaoa/loch_qaoa_tcm_extract_circuits.py` (`run_hardware_like_from_saved_circuits()`)
   > - `src/igdec_qaoa/loch_qaoa_elev_two_extract_circuits.py` (`run_hardware_like_from_saved_circuits()`)
   > - `src/igdec_qaoa/loch_qaoa_elev_three_extract_circuits.py` (`run_hardware_like_from_saved_circuits()`)
   > - `src/piastq_execution/run_calibration.py` (`run_calibration()`)
   >
   > `sampler.set_transpile_options(optimization_level=3)` right below each
   > of these stays as-is. Training functions (`save_trained_circuits()` /
   > `save_trained_circuits_and_initial_solutions()`) don't touch AQT at
   > all — they always use a local ideal `AerSampler`, so nothing there
   > needs to change.
6. **Evaluate (safe anywhere, once step 5's data exists)**:
   ```bash
   cd src/piastq_execution
   python evaluate_all.py
   # -> ../../results/evaluation/<combo>.json, one per combo in
   #    configs/execution_plan.yaml, {method: [per-repetition metrics, ...]}
   ```
   Reads every combo's raw-counts + calibration + persisted QUBOs and
   computes every metric (§8) once per combo x method x repetition — no
   manual per-combo wiring needed.
7. **Compare (safe anywhere)**:
   ```bash
   python compare_all.py
   # -> ../../results/evaluation/comparison_report.{txt,json}
   ```
   Runs `compare_groups()` (§8) for every combo (method vs. `raw`) and, for
   datasets present under both algorithms, QAOA-TCS vs. IGDec-QAOA —
   producing the RQ1/RQ2/RQ3a/RQ3b/RQ4-mapped report below directly.

**Research questions this package is built to answer:**

- **RQ1 (mitigation vs. solution quality)**: does MEM/M3/TREx improve QUBO
  energy and optimal-bitstring probability over raw execution, on real
  PIAST-Q hardware?
- **RQ2 (mitigation vs. practical effectiveness)**: does that improvement
  translate into a better selected test suite (failure rate / input
  diversity / passenger+distance, and its execution cost) for
  single-objective combos, or a better Pareto frontier (non-dominated count,
  HV, IGD) for multi-objective combos?
- **RQ3a (mitigation vs. quality, cost-agnostic)**: among MEM, M3, and TREx,
  which yields the largest quality improvement (QUBO energy,
  probability-of-optimal, and downstream effectiveness/HV/IGD) over raw,
  regardless of what it cost to get there? If MEM/M3 already recover most of
  the lost performance and TREx adds only a marginal improvement on top,
  that's evidence readout error (which MEM/M3 target directly) dominates
  over other noise sources for these circuits — plausible if they're shallow
  enough — rather than TREx's broader (twirling-based) error suppression
  being needed.
- **RQ3b (mitigation cost/benefit)**: normalizing RQ3a's quality improvement
  by what it actually cost in measured hardware time — MEM/M3's
  `mitigation_overhead.calibration_wall_clock_seconds` (timed per circuit in
  `run_calibration.py`, not estimated) plus TREx's extra
  `execution_time_seconds` from its dedicated twirl-instance pass (§5) — which
  method delivers the best quality improvement *per second of hardware time
  spent*? A method that wins RQ3a can still lose RQ3b if its overhead is
  disproportionately large (full MEM's `2**n` calibration circuits or TREx's
  `trex_twirl_instances` extra passes vs. M3's much cheaper `2*n` circuits),
  which matters directly for planning a constrained budget (§7).
- **RQ4 (QAOA-TCS vs. IGDec-QAOA)**: how do the two algorithms compare on
  the same dataset/metric, with and without mitigation, on real hardware --
  including quantum-hardware execution cost/time (`execution_time_seconds`),
  not just solution quality/effectiveness?

---

## 11. Every parameter, and why it has that value

Organized by where the value actually comes from, verified against the code
and the paper rather than assumed.

**A. Given by the PIAST-Q team (hardware constraints, not a choice):**

| Parameter | Value | Source |
|---|---|---|
| `shots_per_batch_cap` | 200 | Machine operators: "there is a constraint of 200 shots per run" |
| `seconds_per_batch` | 15 | Machine operators: "200 shots... take around 15 seconds" |

**B. Inherited from the original QAOA-TCS/IGDec-QAOA paper (grounded there, not re-derived here):**

| Parameter | Value | Source |
|---|---|---|
| Max cluster/subproblem size | 7 qubits | Paper: chosen "consistently with the prior work on IGDec-QAOA (Wang et al., 2024a)" |
| QAOA circuit depth `p` (`TRAINING_REPS`) | 1 | Paper's RQ1 finding: depth doesn't significantly affect quality, so `p=1` minimizes cost without compromising it -- a tested result, not a guess |
| Repetitions per algorithm (paper's own convention) | 10 | Paper: "we ran all stochastic algorithms ten times, consistent with Wang et al. (2024b); Trovato et al. (2024)" |
| IGDec-QAOA decomposition window threshold | `problem_size > 0.15 × len(df)` | Inherited from the original SelectQAOA implementation; not found derived or explained in the paper text -- an implementation-level heuristic, provenance beyond that unverified |
| Per-dataset QUBO weight `alpha` (e.g. paintcontrol=0.45, iofrol=0.50, gsdtsr=0, elevator=0.20) | dataset-specific | Paper: output of an Optuna hyperparameter search over these weights; the paper doesn't tabulate the final numbers, so *how* they were chosen is confirmed but each specific value isn't independently verifiable against a published table |
| Per-dataset Ward-clustering target cluster count (elevator=800, iofrol=324, gsdtsr=60, paintcontrol=16, SIR programs=50) | dataset-specific | Not explained in code or paper; empirically tuned per dataset, not a documented formula |
| `COBYLA(maxiter=500)` | 500 | Not discussed in the paper or code comments; an unremarkable, common default for COBYLA-based QAOA training, inherited as-is |

**C. Decided for this replication package, with modest justification:**

| Parameter | Value | Reasoning |
|---|---|---|
| `min_shots_per_circuit` | 200 | Matches the hardware floor -- one full batch, can't go lower |
| `target_shots_per_circuit` | 2048 | Traced to the original SelectQAOA script's own comment ("Total target shots = 2048 x 30 = 61,440") -- inherited from that pre-hardware-constrained design's own target, not newly derived |
| `target_repetitions` | 10 | Matches item B's paper convention, reused as the planner's "ideal" target |
| `calibration_shots_per_circuit` | 200 | Same floor logic as `min_shots_per_circuit` |
| `physical_qubits` default | `[0,1,2,3,4,5,6]` | Arbitrary identity-mapping placeholder until real PIAST-Q qubit numbers are known -- documented as such in `configs/backend.yaml` |
| `optimization_level=3` | 3 | Qiskit's own highest built-in transpiler optimization level -- standard practice for real hardware |

**D. Still not rigorously justified:**

| Parameter | Value | Status |
|---|---|---|
| `trex_twirl_instances` | 32 | The code comment claims a documented rationale exists (`load_trex_twirl_instances()`'s docstring) -- it doesn't; that docstring only explains why the value is centralized, not why 32. The value does appear as an example configuration in TREx tutorial code (e.g. Mitiq's own TREx docs use `num_randomizations=32`), but is **not** actually "the Qiskit Runtime default" as previously documented here -- Qiskit Runtime's real default is `"auto"` = `max(64, ceil(shots/32))`, a different parameter being conflated with this one. Treat 32 as a reasonable, precedented-in-examples starting point, not a validated choice. |

**What used to be a placeholder and is now fixed**: `shots_per_batch=80` was hardcoded at every one of the 15 hardware-execution call sites across all 5 scripts, completely disconnected from what `budget_planner.py` actually recommended -- running any script would have used 80 shots/circuit regardless of the plan. Fixed via `resolve_all_combo_settings()` (§7): every script now resolves and uses the plan's actual `shots_per_circuit`/`repetitions`/`trex_enabled` per combo.


