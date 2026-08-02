# PIAST-Q4QAOA — Real-Hardware Replication Package

This is a **full replication package** extending the QAOA-TCS / IGDec-QAOA
empirical study (see
`An_Empirical_Study_on_Clustering_based_QAOA_for_Regression_Test_Case_Selection.pdf`,
and the reference implementation in `../SelectQAOA`) onto the **real PIAST-Q
quantum computer**, with readout-error mitigation.

It implements and evaluates two quantum approaches for Test Case Selection:

- **QAOA-TCS**, single- and multi-objective
- **IGDec-QAOA**, single-objective only

The pipeline, end to end:

1. **Decompose & cluster** the dataset into small (<=7-qubit) subproblems
   (already done, embedded in `single_obj.py`/`multi_obj.py`/the IGDec
   scripts — this math is untouched by this replication package).
2. **Train** the QAOA circuits that solve each subproblem (already done for
   most combos — see the matrix below; ideal-simulator training code is
   included for the rest).
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
│       ├── run_calibration.py        #   MEM/M3 calibration entry point (see §5)
│       ├── budget_planner.py         #   hardware-time budget planning
│       ├── evaluation.py             #   per-combo x per-method metrics
│       └── statistics.py             #   Shapiro-Wilk-gated statistical comparison
├── datasets/                         # input data (unchanged)
├── trained_qaoa_circuits/
│   ├── qaoa_tcs/<dataset>/rep_1/*.qpy            # one reusable circuit set per dataset
│   └── igdec_qaoa/<dataset>/sampling_N/*.qpy     # one circuit set per sampling run
├── results/
│   ├── qaoa_tcs/<dataset>/...        # single_obj.py / multi_obj.py output
│   └── igdec_qaoa/<dataset>/...      # IGDec-QAOA output
├── calibration/                      # MEM/M3 calibration snapshots (run_calibration.py output)
├── execution_plan/                   # budget_planner.py output (regenerated on demand)
├── configs/execution_plan.yaml       # budget planner configuration
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

| # | Algorithm | Objective mode | Dataset | Circuits | Circuits directory |
|---|-----------|-----------------|---------|----------|----------------------|
| 1 | QAOA-TCS | single | gsdtsr | ported (79) | `trained_qaoa_circuits/qaoa_tcs/gsdtsr/rep_1/` |
| 2 | QAOA-TCS | single | iofrol | ported (443) | `trained_qaoa_circuits/qaoa_tcs/iofrol/rep_1/` |
| 3 | QAOA-TCS | single | paintcontrol | ported (24) | `trained_qaoa_circuits/qaoa_tcs/paintcontrol/rep_1/` |
| 4 | QAOA-TCS | single | elevator (o2 formulation) | ported (881) | `trained_qaoa_circuits/qaoa_tcs/elevator/rep_1/` |
| 5 | QAOA-TCS | single | elevator2 (o3 formulation) | ported (56) | `trained_qaoa_circuits/qaoa_tcs/elevator2/rep_1/` |
| 6 | QAOA-TCS | multi | flex | ported (105) | `trained_qaoa_circuits/qaoa_tcs/flex/rep_1/` |
| 7 | QAOA-TCS | multi | grep | ported (139) | `trained_qaoa_circuits/qaoa_tcs/grep/rep_1/` |
| 8 | QAOA-TCS | multi | gzip | ported (61) | `trained_qaoa_circuits/qaoa_tcs/gzip/rep_1/` |
| 9 | QAOA-TCS | multi | sed | ported (77) | `trained_qaoa_circuits/qaoa_tcs/sed/rep_1/` |
| 10 | IGDec-QAOA | single | iofrol | ported (10 samplings x 1230 circuits) | `trained_qaoa_circuits/igdec_qaoa/iofrol/sampling_{1..10}/` |
| 11 | IGDec-QAOA | single | paintcontrol | ported (10 samplings x 30 circuits) | `trained_qaoa_circuits/igdec_qaoa/paintcontrol/sampling_{1..10}/` |
| 12 | IGDec-QAOA | single | gsdtsr | **train on the other machine** | `trained_qaoa_circuits/igdec_qaoa/gsdtsr/` |
| 13 | IGDec-QAOA | single | elevator_o2 (elevator_two) | **train on the other machine** | `trained_qaoa_circuits/igdec_qaoa/elevator_two/` |
| 14 | IGDec-QAOA | single | elevator_o3 (elevator_three) | **train on the other machine** | `trained_qaoa_circuits/igdec_qaoa/elevator_three/` |

**Directory convention:** QAOA-TCS (single- and multi-objective alike) keeps
one *reusable* circuit set per dataset — the same circuits get re-run once
per repetition/experiment. IGDec-QAOA keeps a *distinct, non-reusable*
circuit set per random-initial-solution sampling, each with its own
`circuits_metadata.json` and `initial_random_solution.json`.

---

## 4. How to run

Every script assumes you `cd` into its own directory first — that's what
makes its relative paths (`../../datasets/...`, `../../trained_qaoa_circuits/...`,
`../../results/...`) resolve correctly.

### 4.1 QAOA-TCS

```bash
cd src/qaoa_tcs
python single_obj.py     # gsdtsr, iofrol, paintcontrol, elevator, elevator2
                          # -> ../../results/qaoa_tcs/<dataset>/
python multi_obj.py      # flex, grep, gzip, sed
                          # -> ../../results/qaoa_tcs/<dataset>/
```

**Reconfiguring scope**: edit `bootqa_programs` (single_obj.py) or
`sir_programs` (multi_obj.py) at the top of each file to change which
datasets run.

### 4.2 IGDec-QAOA

```bash
cd src/igdec_qaoa
python loch_qaoa_tcm_extract_circuits.py            # hardware-like execution: iofrol
python loch_qaoa_elev_two_extract_circuits.py       # hardware-like execution: elevator_two
python loch_qaoa_elev_three_extract_circuits.py     # hardware-like execution: elevator_three
# -> ../../results/igdec_qaoa/<dataset>.json + <dataset>-raw_counts.jsonl
```

**Training gsdtsr / elevator_two / elevator_three circuits** (not yet
trained anywhere — do this on the other machine, not here):

```bash
python loch_qaoa_tcm_extract_circuits.py train           # gsdtsr
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
| elevator_two | 1,925 (unfiltered) | 41 | ~4,100 |
| elevator_three | 1,925 (unfiltered) | 41 | ~4,100 |

Each subproblem-solve is a 7-qubit ideal-simulator QAOA optimization
(COBYLA, 500 max evaluations) — likely on the order of a few hours
wall-clock total, depending on the machine. **Do not run this training in an
authoring/review session; run it on the machine that will actually spend
that compute.**

Both `single_obj.py`/`multi_obj.py` and all three IGDec-QAOA scripts connect
via `AQTProvider("ACCESS_TOKEN").get_backend("offline_simulator_no_noise")`
exactly as before — a known placeholder ahead of real hardware access, left
untouched. **Run every hardware-execution entry point (everything above
except `... train`) on the machine that actually has PIAST-Q/AQT access.**

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
   tagged with a timestamp and backend snapshot.

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
fully standalone, no IBM backend object needed), with a local
Kronecker-product + NNLS fallback if `mthree` isn't installed:

```python
calibration = store.load("m3", physical_qubits=[0, 1, 2, 3, 4, 5, 6])
corrected_distribution = correct_counts("m3", raw_counts, calibration, num_qubits=7)
```

### TREx (measurement twirling)

Unlike MEM/M3, TREx has **no separate calibration step** — it's applied
per circuit execution, at the point where a script would otherwise call
`run_circuit_with_batching_recorded()` once for the raw pass. Run it several
times with a different random twirl mask each time, then aggregate:

```python
import random
from piastq_execution.mitigation import (
    build_trex_twirled_circuit, generate_random_twirl_mask, aggregate_trex_instances,
)
from piastq_execution.raw_counts import run_circuit_with_batching_recorded

rng = random.Random()
instances = []
for _ in range(NUM_TWIRL_INSTANCES):
    mask = generate_random_twirl_mask(circuit.num_qubits, rng=rng)
    twirled_circuit = build_trex_twirled_circuit(circuit, mask)
    counts, raw_record = run_circuit_with_batching_recorded(
        twirled_circuit, sampler, algorithm=..., objective_mode=..., dataset=...,
        circuit_id=f"{circuit_id}_trex{mask}", backend=backend,
    )
    instances.append((mask, counts))

trex_corrected_counts = aggregate_trex_instances(instances)  # already twirl-undone
```

Each twirl instance is a distinct hardware execution (its own raw-counts
record, tagged `_trex<mask>` in `circuit_id` so it's distinguishable from the
raw baseline in `*-raw_counts.jsonl`) — this is what "TREx can't reuse raw
counts, unlike MEM/M3" means in practice, and why the budget planner (§7)
prices it as a full second hardware pass. `NUM_TWIRL_INSTANCES` is a
judgment call for whoever runs this (more instances = better averaging of
the readout bias, more hardware time); it isn't hardcoded anywhere so you
can tune it per pool.

Full Pauli twirling of the circuit itself (not just before measurement) is a
materially larger follow-up and is not implemented here — see the module
docstring in `mitigation.py`.

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
qubit mapping, backend name/version/timestamp, and wall-clock time. These
`.jsonl` dumps are gitignored since they're large, per-run hardware
artifacts, not authored content.

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
  `calibration_shots_per_circuit` — top-level knobs shared by every pool.

Degradation priority when a pool's budget is too small for its target
repetitions/shots/methods: **(1)** drop TREx if that alone is enough, else
drop it anyway since it's the most expensive method, **(2)** reduce
repetitions, **(3)** reduce shots/circuit (floor: one 200-shot batch), **(4)**
as a last resort, subsample circuits/clusters (reported explicitly, per
circuit, with the reason). Full MEM/M3 calibration circuits are counted once
per distinct circuit width used within a pool, shared across every
combo/repetition using that width; TREx is always a full second hardware
pass (see §5).

This only reads local `.qpy` files (pure deserialization) and writes to
`execution_plan/` — no backend, no AQT, safe to run anywhere, including in an
authoring/review session.

---

## 8. Evaluation and statistics

`src/piastq_execution/evaluation.py` computes, per combo x method
(raw/MEM/M3/TREx), branching on `objective_mode`:

- **Single-objective**: QUBO energy of the corrected/selected solution,
  probability of the brute-forced optimal bitstring (<=7-qubit subproblems,
  128 states, trivial), execution cost + the dataset's effectiveness
  metric(s) of the final merged suite (failure rate for
  gsdtsr/iofrol/paintcontrol; input diversity for elevator_o2; passenger
  count + travel distance for elevator_o3), mitigation overhead, classical
  post-processing time.
- **Multi-objective**: non-dominated solutions contributed to an a posteriori
  reference Pareto frontier (union of every compared method's non-dominated
  points), Hypervolume, IGD, plus the same overhead/post-processing metrics.

`src/piastq_execution/statistics.py` implements the same statistical-
comparison approach used throughout `SelectQAOA/stat_tests/*.R`:
Shapiro-Wilk normality check first, then ANOVA + Tukey HSD + Cohen's d
(normal data) or Kruskal-Wallis + Bonferroni-adjusted Dunn's test +
Vargha-Delaney A12 (non-normal data). `compare_groups(groups)` is the
top-level entry point — pass it `{method_name: [sample values]}` for
whichever metric you're comparing across raw/MEM/M3/TREx (or across
datasets/repetitions).

---

## 9. Tests

```bash
./qiskit_env/bin/python -m unittest discover -s tests
```

Every test is synthetic (hand-built 1-3 qubit toy circuits/QUBOs, fake
samplers, tiny Pareto fronts) — none of them call `AQTProvider`/`AQTSampler`,
run a QAOA training loop, or touch the real trained circuits. They run in
about a second (80 tests total).

---

## 10. What runs where

- **This machine / an authoring or review session**: writing/reviewing code,
  running the test suite above, running the budget planner against the local
  `trained_qaoa_circuits/` directory (local file reads only).
- **The other machine** (the one with PIAST-Q/AQT access and the compute
  budget for training): `... train` for the three untrained IGDec-QAOA
  combos, every hardware-execution entry point (`single_obj.py`,
  `multi_obj.py`, the three IGDec scripts' default mode), and
  `run_calibration.py` for MEM/M3.
