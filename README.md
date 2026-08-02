# PIAST-Q4QAOA — Real-Hardware Replication Package

This is a **full replication package** extending the QAOA-TCS / IGDec-QAOA
empirical study (see
`An_Empirical_Study_on_Clustering_based_QAOA_for_Regression_Test_Case_Selection.pdf`,
and the reference implementation in `../SelectQAOA`) onto the **real PIAST-Q
quantum computer**, with readout-error mitigation.

It implements and evaluates two quantum approaches for Test Case Selection:

- **QAOA-TCS**, single- and multi-objective
- **IGDec-QAOA**, single-objective only

The core idea is to:

1. Extract/train the trained circuits (already done for most combos — see
   below).
2. Re-execute them on real PIAST-Q hardware, with three optional mitigation
   methods (full MEM, M3, TREx).
3. Plan how much of that execution a given hardware-time budget can actually
   afford.
4. Evaluate and statistically compare the results.

---

## 1. The 14-combo matrix

| # | Algorithm | Objective mode | Dataset | Circuits | Circuits directory |
|---|-----------|-----------------|---------|----------|----------------------|
| 1 | QAOA-TCS | single | gsdtsr | ported (79) | `trained_qaoa_circuits/gsdtsr/rep_1/` |
| 2 | QAOA-TCS | single | iofrol | ported (443) | `trained_qaoa_circuits/iofrol/rep_1/` |
| 3 | QAOA-TCS | single | paintcontrol | ported (24) | `trained_qaoa_circuits/paintcontrol/rep_1/` |
| 4 | QAOA-TCS | single | elevator (o2 formulation) | ported (881) | `trained_qaoa_circuits/elevator/rep_1/` |
| 5 | QAOA-TCS | single | elevator2 (o3 formulation) | ported (56) | `trained_qaoa_circuits/elevator2/rep_1/` |
| 6 | QAOA-TCS | multi | flex | ported (105) | `trained_qaoa_circuits/flex/rep_1/` |
| 7 | QAOA-TCS | multi | grep | ported (139) | `trained_qaoa_circuits/grep/rep_1/` |
| 8 | QAOA-TCS | multi | gzip | ported (61) | `trained_qaoa_circuits/gzip/rep_1/` |
| 9 | QAOA-TCS | multi | sed | ported (77) | `trained_qaoa_circuits/sed/rep_1/` |
| 10 | IGDec-QAOA | single | iofrol | ported (10 samplings, 1230 circuits each) | `trained_qaoa_circuits/igdec_qaoa/iofrol/sampling_{1..10}/` |
| 11 | IGDec-QAOA | single | paintcontrol | ported (10 samplings, 30 circuits each) | `trained_qaoa_circuits/igdec_qaoa/paintcontrol/sampling_{1..10}/` |
| 12 | IGDec-QAOA | single | gsdtsr | **train on the other machine** | `trained_qaoa_circuits/igdec_qaoa/gsdtsr/` |
| 13 | IGDec-QAOA | single | elevator_o2 (elevator_two) | **train on the other machine** | `trained_qaoa_circuits/igdec_qaoa/elevator_two/` |
| 14 | IGDec-QAOA | single | elevator_o3 (elevator_three) | **train on the other machine** | `trained_qaoa_circuits/igdec_qaoa/elevator_three/` |

**Directory convention:** QAOA-TCS (single- and multi-objective alike) keeps
one *reusable* circuit set per dataset under `trained_qaoa_circuits/<dataset>/rep_1/`
— the same circuits get re-run once per repetition/experiment.
IGDec-QAOA keeps a *distinct, non-reusable* circuit set per random-initial-
solution sampling under `trained_qaoa_circuits/igdec_qaoa/<dataset>/sampling_N/`
— each sampling has its own circuits, metadata (`circuits_metadata.json`) and
initial solution (`initial_random_solution.json`).

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

## 3. How to run

### 3.1 QAOA-TCS

```bash
cd qaoa_tcs
python single_obj.py     # gsdtsr, iofrol, paintcontrol, elevator, elevator2
python multi_obj.py      # flex, grep, gzip, sed
```

Both connect to `AQTProvider("ACCESS_TOKEN").get_backend("offline_simulator_no_noise")`
exactly as before — a known placeholder ahead of real hardware access, left
untouched. **Run this on the machine that actually has PIAST-Q/AQT access —
not needed to author or review this code.**

### 3.2 IGDec-QAOA

```bash
cd ../igdec_qaoa
python loch_qaoa_tcm_extract_circuits.py            # hardware-like execution: iofrol
python loch_qaoa_elev_two_extract_circuits.py       # hardware-like execution: elevator_two
python loch_qaoa_elev_three_extract_circuits.py     # hardware-like execution: elevator_three
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

---

## 4. Budget planning

`piastq_execution/budget_planner.py` turns a per-pool hardware-time budget
into a concrete execution plan (repetitions, shots/circuit, which mitigation
methods survive) by reading the *actual* circuit counts under
`trained_qaoa_circuits/` — nothing is hardcoded. See
`configs/execution_plan.yaml` for the pool/combo configuration format (any
mix of "one big pool" and "one pool per combo" is supported directly).

```bash
cd piastq_execution
python budget_planner.py
```

This only reads local `.qpy` files (pure deserialization) and writes a report
to `results/execution_plan/` — no backend, no AQT, safe to run anywhere.

Degradation priority when a pool's budget is too small for its target
repetitions/shots/methods: **(1)** drop TREx if that alone is enough, else
drop it anyway since it's the most expensive method, **(2)** reduce
repetitions, **(3)** reduce shots/circuit (floor: one 200-shot batch), **(4)**
as a last resort, subsample circuits/clusters (reported explicitly). Full
MEM/M3 calibration circuits are counted once per distinct circuit width used
within a pool, shared across every combo/repetition using that width; TREx is
always a full second hardware pass (it can't reuse raw counts).

---

## 5. Mitigation

`piastq_execution/mitigation.py` implements all three methods identically
regardless of algorithm/objective-mode/dataset (mitigation operates purely on
per-circuit counts):

- **Full MEM** — `build_mem_calibration_circuits(n)` builds the 2**n
  basis-state preparation circuits for a given circuit width; run them
  through the same sampler as everything else, then
  `build_confusion_matrix()` + `correct_counts_with_matrix()` (NNLS, with a
  documented pseudo-inverse fallback) recovers a valid (non-negative,
  normalized) corrected distribution.
- **M3** — `build_m3_calibration_circuits(n)` builds 2 per-qubit calibration
  circuits; `correct_counts_m3()` feeds them to the `mthree` package
  (`M3Mitigation(system=None)` + `cals_from_matrices()` — confirmed to work
  fully standalone, no IBM backend object needed) with a local
  Kronecker-product + NNLS fallback if `mthree` isn't installed.
- **TREx** — `build_trex_twirled_circuit()` / `undo_trex_twirl()` /
  `aggregate_trex_instances()` implement the measurement-twirling MVP (random
  X before measurement, classically XOR-undone per twirl instance). Full
  Pauli twirling of the circuit itself is a materially larger follow-up and
  is not implemented here.

`CalibrationStore` persists MEM/M3 calibration under `calibration/`, keyed by
(method, physical-qubit signature) and timestamped/backend-tagged, so it can
be reapplied to any raw-counts file without rerunning hardware.

---

## 6. Raw-counts persistence

Every execution tail (`qaoa_tcs/single_obj.py`, `qaoa_tcs/multi_obj.py`,
`igdec_qaoa/loch_qaoa_tcm_extract_circuits.py`, and the two elevator IGDec
scripts) uses the shared `piastq_execution/raw_counts.py:
run_circuit_with_batching_recorded()` instead of collapsing every circuit's
result down to a single argmax bitstring. Alongside each combo's existing
derived outputs (unchanged: `*-rep-N.json`, `*-subsuites.json`,
`pareto_front_N`), a `*-raw_counts.jsonl` file is written with, per circuit
per shot-batch: the full raw counts dict, circuit/cluster/(iteration+
subproblem) id, algorithm/objective_mode/dataset, shots requested vs.
returned, physical qubit mapping, backend name/version/timestamp, and
wall-clock time. These `.jsonl` dumps are gitignored (see `.gitignore`) since
they're large, per-run hardware artifacts, not authored content.

---

## 7. Evaluation and statistics

`piastq_execution/evaluation.py` computes, per combo x method
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

`piastq_execution/statistics.py` implements the same statistical-comparison
approach used throughout `SelectQAOA/stat_tests/*.R`: Shapiro-Wilk normality
check first, then ANOVA + Tukey HSD + Cohen's d (normal data) or
Kruskal-Wallis + Bonferroni-adjusted Dunn's test + Vargha-Delaney A12
(non-normal data).

---

## 8. Tests

```bash
./qiskit_env/bin/python -m unittest discover -s tests
```

Every test is synthetic (hand-built 1-3 qubit toy circuits/QUBOs, fake
samplers, tiny Pareto fronts) — none of them call `AQTProvider`/`AQTSampler`,
run a QAOA training loop, or touch the real trained circuits. They run in
about a second.

---

## 9. What runs where

- **This machine / an authoring or review session**: writing/reviewing code,
  running the test suite above, running the budget planner against the local
  `trained_qaoa_circuits/` directory (local file reads only).
- **The other machine** (the one with PIAST-Q/AQT access and the compute
  budget for training): `... train` for the three untrained IGDec-QAOA
  combos, and every hardware-execution entry point
  (`single_obj.py`, `multi_obj.py`, the three IGDec scripts' default mode,
  and MEM/M3/TREx calibration circuit execution).
