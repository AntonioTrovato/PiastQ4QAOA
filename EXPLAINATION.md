# EXPLAINATION.md — The study, the code, every decision, and the results

> **Purpose.** This is the reference document of the replication package. It is
> written so that a reader (or another Claude instance) who has never seen the
> project can write the paper from it: what was studied and why, exactly what
> was executed on the real quantum computer, every design and parameter choice
> with its justification, the final results with their statistical support, and
> every known limitation. Statements are tied to files so they can be checked.
> Where something was *not* verified, or is an inherited quirk, it says so.
>
> **Status (as of 9 Nov 2026): the experimental work is finished.** The study
> ran on the real PIAST-Q machine between 30 Oct and 6 Nov 2026; all raw data,
> calibrations, evaluation files and the statistical report are in this
> repository. If you only read one section, read **§1 (summary)** and
> **§12 (results)**, then **§14 (threats to validity)**.
>
> File names: the document is called `EXPLAINATION.md` (sic) for historical
> reasons; other files refer to it by that name.

---

## Table of contents

1. [Summary of the study](#1-summary-of-the-study)
2. [Background: the problem and the algorithm](#2-background-the-problem-and-the-algorithm)
3. [What was studied: scope, subjects, research questions](#3-what-was-studied-scope-subjects-research-questions)
4. [Subjects in detail: datasets, QUBOs, clustering, circuits](#4-subjects-in-detail-datasets-qubos-clustering-circuits)
5. [Training of the circuits (offline, before the study)](#5-training-of-the-circuits-offline-before-the-study)
6. [The machine and how it is reached](#6-the-machine-and-how-it-is-reached)
7. [Execution design: how circuits were run on the hardware](#7-execution-design-how-circuits-were-run-on-the-hardware)
8. [The four methods: raw, MEM, M3, TREx — as implemented](#8-the-four-methods-raw-mem-m3-trex--as-implemented)
9. [Calibration design and what it measured](#9-calibration-design-and-what-it-measured)
10. [Evaluation: metrics, statistics, how RQs are answered](#10-evaluation-metrics-statistics-how-rqs-are-answered)
11. [What was actually run: configuration and chronology](#11-what-was-actually-run-configuration-and-chronology)
12. [Results](#12-results)
13. [Answers to the research questions](#13-answers-to-the-research-questions)
14. [Threats to validity and known limitations](#14-threats-to-validity-and-known-limitations)
15. [Every parameter and why it has that value](#15-every-parameter-and-why-it-has-that-value)
16. [Repository map and data dictionary](#16-repository-map-and-data-dictionary)
17. [How to reproduce, re-evaluate or extend](#17-how-to-reproduce-re-evaluate-or-extend)
18. [Out of scope but implemented (IGDec-QAOA, other datasets)](#18-out-of-scope-but-implemented-igdec-qaoa-other-datasets)
19. [Glossary](#19-glossary)
20. [Decision log](#20-decision-log)

---

## 1. Summary of the study

**What.** Regression Test Case Selection (TCS) picks a cheap but effective
subset of a test suite. QAOA-TCS (an existing method) formulates TCS as a QUBO,
splits the suite into clusters of at most 7 test cases, and solves each cluster
with a small, pre-trained QAOA circuit (`p = 1`). Until now QAOA-TCS had been
evaluated on simulators. This study runs it on a **real 20-qubit trapped-ion
quantum computer, PIAST-Q**, and asks whether **readout-error mitigation**
improves the result, and at what cost in scarce QPU time.

**Three mitigation methods** are compared with the unmitigated baseline
("raw"): full measurement-error mitigation (**MEM**, full confusion matrix),
**M3** (per-qubit calibration, scalable matrix-free correction) and **TREx**
(measurement twirling). The comparison is a *paired design* for raw, MEM and M3
(all three are computed from the same stored counts) and a separate set of
hardware passes for TREx.

**Subjects.** QAOA-TCS only: two single-objective datasets (`gsdtsr`,
`paintcontrol`) and two multi-objective programs (`flex`, `gzip`), i.e.
79 + 24 + 105 + 61 = **269 circuits** of 1–7 qubits (287 + 89 + 567 + 214 test
cases). Each circuit is run with 200 shots; the full set of circuits is run
**10 times** ("repetitions") for each method; TREx uses **3 twirled passes** per
circuit.

**What was run.** About 2,510 execution jobs + 130 calibration jobs + 2 drift
jobs on the real machine, **13.6 QPU hours billed out of the 25 available**
(account counter 13.67 h at the end), over 30 Oct – 6 Nov 2026. To make the
wall-clock fit the deadline, several independent circuits were *packed side by
side on disjoint qubits of one 20-qubit job* (§7.2).

**Headline findings** (details and statistics in §12–§13):

* Readout errors on this machine are **very small and one-sided**: mean
  P(read 0 | prepared 1) = 0.5 % (max 1.5 %), P(read 1 | prepared 0) = 0 % on
  all 20 qubits (§9.3). There is little for mitigation to correct, which
  frames every result below.
* **RQ1 (quality).** Mitigation does *not* improve quality in general. A small
  but statistically robust gain appears only on the single-objective subject
  `gsdtsr` (QUBO energy −8.517 raw → −8.560/−8.562/−8.562 for MEM/M3/TREx, all
  p_adj < 0.001, ≈ 19 % of the remaining gap to the noiseless optimum −8.743),
  and MEM raises the probability of the optimal bitstring by +0.4 pp
  (`gsdtsr`) and +1.4 pp (`paintcontrol`). On the multi-objective subjects there
  is no benefit at all; TREx is significantly *worse* on `gzip` (IGD).
* **RQ2 (suite-level effectiveness).** No consistent difference, apart from
  `gsdtsr` where the effectiveness indicator is mathematically the energy
  again (α = 0).
* **RQ3 (cost-effectiveness).** TREx costs ≈ 3× the QPU time of a raw run and
  never helps. M3 costs 37 s once (≈ 1–4 % of a subject's QPU time over 10
  repetitions) and gives the same energy gain as MEM (on P(optimal) MEM's gain
  is significant, M3's is not); MEM costs 2,368 s once (+40 % to +276 %
  of a subject's QPU time). **M3 is the most cost-effective method.**

---

## 2. Background: the problem and the algorithm

### 2.1 Test case selection (TCS)

A project has `n` automated tests. After a code change, re-running all of them
is too expensive; TCS selects a subset that is cheap (low execution cost) yet
effective (high fault-revealing power). Each test has attributes — execution
time/cost, historical failure rate, and, for multi-objective programs,
statement coverage and fault coverage.

### 2.2 QUBO

A QUBO (Quadratic Unconstrained Binary Optimization) has one bit `x_i ∈ {0,1}`
per test (1 = selected) and minimizes `Σ_i l_i x_i + Σ_{i<j} q_ij x_i x_j`. The
repository's QUBOs are built in `src/qaoa_tcs/single_obj.py` and
`src/qaoa_tcs/multi_obj.py` (details in §4.2). **An important structural fact:**
the single-objective QUBOs are **purely linear** (no `q_ij` terms), and in the
multi-objective ones the quadratic terms survive in only one cluster of `flex`
and one of `gzip` (see §4.4). Consequently the QAOA circuits are essentially
**products of independent single-qubit circuits with no entanglement**: of the
269 circuits, 0/79 (`gsdtsr`), 0/24 (`paintcontrol`), 1/105 (`flex`) and 1/61
(`gzip`) contain any two-qubit gate after unrolling. The problem is thus
classically trivial (a bit is 1 iff its linear coefficient is negative); the
study is about *running the quantum algorithm on a real machine*, not about
quantum advantage, and the dominant noise source for these shallow,
non-entangling circuits is readout (plus single-qubit gate error).

### 2.3 QAOA and "training"

QAOA turns the QUBO into a parameterized circuit (cost layer + mixer layer,
`reps = p`; here `p = 1`). A classical optimizer (COBYLA, ≤ 500 evaluations)
tunes the parameters on an **ideal simulator** so that the circuit's output
distribution concentrates on low-energy bitstrings. The tuned ("frozen")
circuit is saved as a `.qpy` file. At execution time the frozen circuit is
sampled on the machine (200 shots); the **most frequent bitstring** (argmax) is
taken as the cluster's answer.

### 2.4 QAOA-TCS decomposition

Hardware (and simulators) cannot hold thousands of qubits, so the suite is
decomposed **once, deterministically**:

1. standardize the per-test attributes (`StandardScaler`);
2. Ward hierarchical clustering into a target number of clusters
   (`fcluster(..., criterion='maxclust')`);
3. any cluster with more than 7 members is split into balanced parts of ≤ 7;
4. one QUBO and one QAOA circuit per final cluster; the cluster answers are
   merged (union of selected tests) into the final suite.

Because the decomposition is deterministic and the circuits are frozen, the
**same 269 circuits are re-run in every repetition** — repetitions measure
machine/shot variability, not training variability.

### 2.5 Single- vs multi-objective

* **Single-objective** (`gsdtsr`, `paintcontrol`): one weighted objective
  `α·(normalized cost) − (1−α)·failure_rate`; the output is one suite.
* **Multi-objective** (`flex`, `gzip`; SIR benchmark programs): the same kind
  of per-cluster QAOA solving, but the merged selection is turned into a
  **Pareto front** by the "Additional-Greedy" prefix method: every growing
  prefix of the selected-test list is a candidate sub-suite, scored on
  `(−execution_cost, statement_coverage, fault_coverage)` (cost negated so all
  three are maximized), and the non-dominated candidates form the front
  (`multi_obj.build_pareto_front`).

---

## 3. What was studied: scope, subjects, research questions

### 3.1 Goal

To evaluate QAOA-based test case selection on real quantum hardware and to
quantify how much readout-error mitigation improves it, and at what cost.

### 3.2 Research questions (final numbering)

| RQ | Question | Metrics |
|---|---|---|
| **RQ1** | Does mitigation (MEM, M3, TREx) improve solution **quality** over raw hardware output? | single-objective: QUBO energy of the selected solution, probability of the optimal bitstring; multi-objective: hypervolume (HV), inverted generational distance (IGD), number of non-dominated solutions |
| **RQ2** | Does better quality translate into a better **selected test suite**? | single-objective: `execution_cost` and the dataset effectiveness indicator (`failure_rate`) of the final suite. Multi-objective programs have no separate suite-level indicator beyond RQ1's Pareto metrics |
| **RQ3** | Which technique gives the best **improvement per second of billed QPU time**? | billed QPU seconds of the executions and of the calibration, related by hand to RQ1/RQ2 gains |

History (useful when reading older commits/docs): an earlier design had RQ3a
("which method improves quality most, regardless of cost"), RQ3b (cost) and RQ4
(QAOA-TCS vs IGDec-QAOA). RQ3a was merged into RQ1 as a discussion point (it is
the same comparison), RQ3b became RQ3, and RQ4 was dropped because IGDec-QAOA
was not executed (§3.4).

### 3.3 Subjects and why these four

QAOA-TCS on `gsdtsr` and `paintcontrol` (single-objective) and `flex` and
`gzip` (multi-objective). Two of each type so conclusions are not tied to one
problem type. They were the subset that fits the 25-QPU-hour allowance and the
6 Nov deadline: `iofrol` (443 circuits), `elevator` (881) and `grep`/`sed`
need several times the machine time; `elevator2`'s 56 saved circuits do not
match the current clustering (800 target clusters) and cannot be evaluated.

### 3.4 What is explicitly out of scope

* **IGDec-QAOA** (the second algorithm of the original study) is implemented
  and unit-tested here (§18) but was **not run on the hardware**, so no
  algorithm comparison is made.
* Mitigation scope is **readout error only**. Gate-error mitigation is not
  studied.
* Only `p = 1` circuits (`rep_1`) are used.

---

## 4. Subjects in detail: datasets, QUBOs, clustering, circuits

### 4.1 Datasets (`datasets/`)

| Subject | Mode | Source | Tests | Attributes used |
|---|---|---|---|---|
| `gsdtsr` | single | Google shared test-suite results (CSV, rows with failure rate > 0 only) | 287 | execution time (`cost`), failure rate |
| `paintcontrol` | single | ABB Robotics painting-robot suite (CSV, rate > 0) | 89 | execution time, failure rate |
| `flex` | multi | SIR benchmark C program (`datasets/sir_programs`) | 567 | cost, fault coverage, statement coverage |
| `gzip` | multi | SIR benchmark C program | 214 | cost, fault coverage, statement coverage |

(`single_obj.get_data` filters `rate > 0` for the TCM datasets; for `gsdtsr`
the raw CSV has 5,555 rows, 287 remain.)

### 4.2 QUBO formulations (inherited, not changed here)

* **Single-objective** (`make_linear_terms_bootqa`): for test `t` in a cluster,
  `l_t = α · cost_t / max_cost − (1 − α) · failure_rate_t`, **linear terms
  only**. α per dataset: `gsdtsr` **0**, `paintcontrol` **0.45** (also
  `iofrol` 0.50, `elevator` 0.20, `elevator2` (0.1, 0.9, 0.9), unused here).
  Per the original paper these α values come from an Optuna search; the exact
  final values are carried over as-is.
  * **Consequence for interpretation:** with α = 0 (`gsdtsr`) the QUBO ignores
    cost entirely, so QUBO energy = −(summed failure rate of the selected
    tests) *exactly*. For `gsdtsr` the RQ2 indicator `failure_rate` is therefore
    **the energy with the opposite sign — not independent evidence** — and a
    higher suite `execution_cost` simply means more tests were selected.
    For `paintcontrol` (α = 0.45) energy and failure rate are different.
* **Multi-objective** (`multi_obj.make_linear_terms` /
  `make_quadratic_terms`, α = 0.5): linear term
  `0.5·cost_t/max_cost − 0.5·fault_t`; a quadratic penalty
  (`2·penalty` per pair, with `penalty = max single-test term + 1`) between two
  tests of the same cluster that cover the same statement, discouraging
  redundant coverage.
  * **Inherited quirk (observation, not verified against the original
    paper):** `make_quadratic_terms` tests `i not in variables`, comparing a
    *global* test id with the cluster's *local* variable indices `0..k−1`, so
    most pairs are skipped. This matches what is observed in the saved QUBOs:
    non-empty quadratic parts in only 1 of 105 `flex` clusters and 1 of 61
    `gzip` clusters. The multi-objective QUBOs are therefore effectively linear.
    The saved trained circuits were used exactly as they are.

### 4.3 Clustering (deterministic)

Ward linkage on the standardized attributes, `fcluster(maxclust)` with target
cluster counts `gsdtsr` 60, `paintcontrol` 16, SIR programs 50, then splitting
of clusters larger than 7 into balanced parts. These cluster counts are
inherited engineering defaults (not derived in the paper); after splitting the
final numbers of clusters/circuits are 79, 24, 105 and 61.

### 4.4 Resulting circuits (measured from `trained_qaoa_circuits/`)

Circuit widths (number of circuits with that many qubits):

| Subject | clusters = circuits | 1q | 2q | 3q | 4q | 5q | 6q | 7q | circuits with a 2-qubit gate | circuits with a non-empty quadratic QUBO part |
|---|---|---|---|---|---|---|---|---|---|---|
| `gsdtsr` | 79 | 22 | 11 | 7 | 8 | 10 | 7 | 14 | 0 | 0 |
| `paintcontrol` | 24 | 6 | 4 | 0 | 3 | 5 | 4 | 2 | 0 | 0 |
| `flex` | 105 | 4 | 5 | 1 | 12 | 25 | 29 | 29 | 1 | 1 |
| `gzip` | 61 | 13 | 7 | 6 | 18 | 10 | 2 | 5 | 1 | 1 |

After unrolling, the circuits consist of single-qubit `u3`/`u` gates (plus
`cx`/`u1` in the two circuits that have a quadratic part), followed by
measurement.

**Noiseless reference** (computed from the exact statevectors of the saved
circuits by `src/piastq_execution/ideal_reference.py`, file
`results/evaluation/ideal_reference.json`; infinite shots, no noise):

| Subject | mean P(optimal) | clusters where the most likely bitstring is the optimum | Σ QUBO energy of the argmax bitstrings | Σ energy of the brute-force optima |
|---|---|---|---|---|
| `gsdtsr` | 0.3770 | 79 / 79 | −8.7432 | −8.7432 |
| `paintcontrol` | 0.7884 | 24 / 24 | −8.5357 | −8.5357 |
| `flex` | 0.9301 | 105 / 105 | −304.0214 | −304.0214 |
| `gzip` | 0.7531 | 60 / 61 | −120.0745 | −190.6219 |

Two take-aways: (i) an ideal machine would recover the exact optimum in every
cluster of three subjects, so any shortfall on hardware is attributable to
noise and finite shots; (ii) in `gzip` one trained circuit (the one with the
quadratic term) does not reach its optimum even noiselessly and accounts for
the whole energy gap. Note that the *probability* of the optimum is not close to
1 even without noise (`gsdtsr` 0.377), because the `p = 1` QAOA output is
spread over many bitstrings: **the argmax is robust while P(optimal) is a
noisy, low-contrast indicator.**

### 4.5 Persisted QUBO coefficients (`*_qubos.json`)

Evaluation needs each circuit's QUBO. The originally ported circuits had none,
so `single_obj.py qubos` / `multi_obj.py qubos` ("QUBO backfill", stage 1b of
`run_all.py`) rebuild them from the deterministic clustering **without
retraining** and verify them against the saved circuits (cluster count and
every circuit's qubit count must match, otherwise nothing is written). The
files `trained_qaoa_circuits/qaoa_tcs/<subject>/rep_1/<subject>_rep1_qubos.json`
are required for `evaluate_all.py`; they are part of the committed package.

---

## 5. Training of the circuits (offline, before the study)

Circuit training is **not** part of the hardware study and was done earlier
(ported from the original `SelectQAOA/MOQ-Pipeline.ipynb`). For reproducibility
the training code lives in `single_obj.py train` / `multi_obj.py train`:

* ideal noiseless simulator (`qiskit_aer` `Sampler`, `shots = None`),
* `qiskit_algorithms.QAOA(sampler, COBYLA(maxiter=500), reps=1)`,
* one optimization per cluster; the optimal point is bound into the ansatz and
  saved as `.qpy`.

**Reproducibility caveat:** QAOA-TCS's training is not explicitly seeded, so
re-running `train` is not guaranteed to reproduce the same `.qpy` files. The
committed `.qpy` files are the artifacts the study used. (IGDec-QAOA's training
*is* seeded, via `seeding.seed_for_sampling`.)

---

## 6. The machine and how it is reached

### 6.1 PIAST-Q

A 20-qubit trapped-ion quantum computer reached through the PCSS (Poznań)
service. The Python access library is `pcss-qapi` (0.3.2), which exposes an
AQT-style resource; circuits are submitted through `qiskit-aqt-provider`
(`AQTSampler`). Constraints: **at most 200 shots per run**; **25 QPU hours** of
*billed* time for the study. Records report the backend as `direct-access`,
version `2`.

### 6.2 Billing and latency — the measurements that shaped the design

Measured on 29 Oct 2026 from the service's usage counter (`/me/usage`):

* A **200-shot job is billed 18.4974 s regardless of the number of qubits**
  (1, 3 and 7 qubits all identical); a 100-shot job 10.0774 s. Fitted model:
  `1.6574 + 0.0842 × shots` (`qpu_usage.py`). The operators had announced "about
  15 s"; the measurement supersedes it.
* Idle time inside a session is not billed; failed jobs bill 0.
* **Wall-clock is the real bottleneck.** Every job occupies the machine for
  roughly 2–5 minutes of unbilled overhead and waits in a shared queue. A
  one-circuit-per-job run (≈ 4,400 jobs) would take 9–15 days. Jobs outside a
  session have priority 1, session jobs 20, but sessions gave no speed-up.
* The service API supports cancelling queued jobs (`POST /jobs/{id}/cancel`,
  409 if running).

### 6.3 Where the code runs

* Reaching the machine needs a VPN, a token and Python ≥ 3.11. The study ran on
  the PCSS **Slurm cluster** (login `150.254.192.221`; the VPN is only needed to
  reach the login node, the compute node reaches the service API), in a
  Python 3.12 virtual environment (`requirements.txt` pins `qiskit==1.4.6`,
  `qiskit-aqt-provider==1.15.0`, `pcss-qapi==0.3.2`, …; `qiskit` must stay < 2).
* `run_cluster.sh` is the Slurm launcher (1 node, 4 CPUs, 16 GB, partition
  `all`, no time limit). **It alone switches on the real machine** via
  environment variables (`PIASTQ_BACKEND_NAME=piast_q`,
  `PIASTQ_ALLOW_REAL_HARDWARE=1`, `PIASTQ_API_TOKEN` read from
  `~/.piastq_token`, chmod 600). The committed repo files stay on the safe
  offline simulator (`configs/backend.yaml`: `offline_simulator_no_noise`), so
  nothing real can start by accident. The token is never committed; the
  `piastq_vpn/` folder with VPN credentials is git-ignored.

---

## 7. Execution design: how circuits were run on the hardware

### 7.1 Per-circuit protocol

For every subject and every repetition (`iteration_id` 1…10) and every cluster:

1. load the frozen `.qpy` circuit;
2. **raw pass:** the circuit with its measurement, 200 shots (one batch);
3. **TREx passes:** 3 independent copies, each with a fresh random X-twirl mask
   before measurement (§8.4), 200 shots each (600 shots in total);
4. every circuit is transpiled with `optimization_level=3` to the machine's
   native gates, with an explicit `initial_layout` (the physical qubits it was
   assigned), so the execution and its calibration use the same qubits.

Record naming: files are called `<subject>-rep-1-…`. **"rep-1" is the QAOA
depth `p = 1`, not a repetition.** The 10 repetitions are the `iteration_id`
values 1…10 inside the files.

### 7.2 Packing: several circuits per job

Because wall-clock was the bottleneck and the billed time per job does not
depend on qubit count (§6.2), the 20 qubits are split into three fixed
**slots**: `pack_slots: [[0,7],[7,7],[14,6]]` (`configs/backend.yaml`;
first qubit, width). `src/piastq_execution/packing.py`:

* `plan_packs` places circuits with a deterministic **first-fit-decreasing**
  rule: widest first, each into the first job whose slot still has room,
  several circuits can follow one another inside a slot; observed fill
  90–98 % of the 20 qubits.
* `build_packed_circuit` composes the member circuits side by side on disjoint
  qubits of one combined circuit (dropping their own measurements and measuring
  every qubit once), and returns the physical layout.
* After the run, `marginalize_counts` splits the joint counts per circuit by
  marginalizing the classical bits of its window.
* Each circuit keeps **its own 200 shots, its own counts and its own
  `RawCountsRecord`**, with `physical_qubit_mapping` pointing at its real
  window. The job's billed time is **shared among members in proportion to
  their qubit share** (`member_record_from_joint`); consequently
  `execution_time_seconds` of a subject depends slightly on the packing mix
  (TREx/raw ratios are 3.0–3.1×, not exactly 3).

Within one (subject, repetition) the raw circuits of all clusters and all their
TREx copies are packed together, so a job mixes raw and TREx circuits. The
statistical independence of circuits is by construction (disjoint qubits), but
neighbours run simultaneously (possible readout crosstalk; MEM/M3 were
calibrated with all slots active, §9.1, to reflect this).

### 7.3 Concurrency, retries, resume, safety stops

* **Concurrency.** Up to `MAX_JOBS_IN_FLIGHT = 16` jobs are submitted at once
  (thread pool, one sampler per job). They wait in the shared queue together;
  the machine still runs them one at a time. This overlaps queue waits.
* **Retries.** A job that comes back `status: failed` is resubmitted up to
  `MAX_JOB_ATTEMPTS = 3` (`raw_counts._submit_and_wait`); failed jobs bill 0.
  Transport errors while waiting for a result are retried every 5 s for about
  30 minutes without resubmitting (`backend_config._harden_pcss_backend`); a
  *submit* that times out or fails to connect is also retried (this was added
  during the study after a `ReadTimeout`, §11.2; at worst it leaves one orphan
  job of ≈ 18.5 s).
* **Resume.** `python run_all.py --resume` (env `PIASTQ_RESUME=1`) reuses every
  circuit already in the raw-counts files, skips fully finished jobs, skips a
  finished calibration and drift check, and skips the pilot gate. A half-written
  last line of a `.jsonl` file (interrupted write) is repaired before appending.
  Records are keyed by `(circuit_id, iteration_id)`; **this is what let the
  study be extended from 5 to 10 repetitions**: repetitions 1–5 were reused
  and only 6–10 were executed (verified on a simulator copy before the real
  extension: no duplicates, first records unchanged).
* **Budget stop.** Before every batch, `qpu_usage.check_budget` reads the
  account's billed time from the service and refuses to submit if the next
  batch would cross `PIASTQ_QPU_LIMIT_SECONDS` (25 h minus a margin of
  `(16 + 2) × 18.4974 s` for jobs already in flight).
* **Pilot gate.** After calibration, `run_all.py` projects the whole run from
  the billed seconds per calibration job and aborts if the projection exceeds
  25 h (skipped on resume, because nothing is billed then and wall-clock on the
  real machine is queue wait, not QPU time).

### 7.4 Pipeline orchestration (`run_all.py`)

Stages, in order: (1) unit tests; (1b) QUBO backfill; (2) budget plan;
(3) calibration (also saves a backend-properties snapshot); (4) pilot gate;
(5) QAOA-TCS single-objective; (6) QAOA-TCS multi-objective; (7) IGDec-QAOA
(only if planned — not in this study); (8) end-of-session drift check;
(9) evaluation and statistical comparison. A failing stage stops the run;
`--resume` continues. The unit-test stage runs with every `PIASTQ_*` variable
stripped so tests can never reach the real service. Output: console,
`logs/run_all_<time>.log`, `logs/run_summary_<time>.json` (the `logs/` folder
is git-ignored, so those files are not in a clone; the facts they held are
recorded in §11).

### 7.5 Budget planner (`budget_planner.py`)

Converts a time budget into an execution plan using the cost model
`ceil(shots/200) × 18.4974 s` per circuit execution. For this study
(`configs/execution_plan.yaml`: one pool `reduced_scope_run`, the four
subjects, all four methods, 10 repetitions, 200 shots, 3 twirls) its
one-circuit-per-job **upper bound** is ≈ 56.9 h; the cap `total_hours: 80` is
only the planner's own ceiling and must stay above that bound, otherwise the
planner would silently drop TREx or repetitions. The real need, thanks to
packing, was ≈ 13.6 h. (`execution_plan/execution_plan_report.*` is the output
of the planner for the shipped configuration.)

---

## 8. The four methods: raw, MEM, M3, TREx — as implemented

All four are evaluated on the same circuits; the "answer" of a cluster is the
**argmax** of the (corrected) distribution, with a deterministic,
method-independent tie-break (highest probability rounded to 12 digits, then
smallest bitstring) so floating-point noise cannot create a spurious effect.

### 8.1 Raw (baseline)

The normalized counts of the raw pass (200 shots).

### 8.2 MEM — full measurement-error mitigation

`mitigation.build_confusion_matrix` builds the full `2^w × 2^w` assignment
matrix `A[measured, prepared]` from the calibration; the corrected
distribution solves `A x = p_noisy` by **non-negative least squares**
(`scipy.optimize.nnls`) and is renormalized (fallback: pseudo-inverse with
clipping). It uses the **same raw counts as raw** (no extra execution).

### 8.3 M3 — matrix-free measurement mitigation

Per-qubit 2×2 assignment matrices are combined as a tensor product restricted
to the observed outcomes; the correction is computed with the `mthree` package
(iterative solver; a reduced-support NNLS fallback exists if `mthree` is
missing — not used in the study). `mthree` returns a *quasi-probability* distribution (entries can be slightly negative; it is not clipped), which is what the evaluation uses for `probability_of_optimal`; the argmax is taken on it. Again **same raw counts as raw**. Tiny
numerical differences between runs (≈ 1e-8 in probabilities) come from the
iterative solver's tolerance; no argmax changed.

### 8.4 TREx — measurement twirling, **without rescaling**

For each of the 3 passes a random mask `m ∈ {0,1}^w` is drawn
(`generate_random_twirl_mask`, from an **unseeded** `random.Random()`; the mask
is stored in the record, so the analysis is reproducible even though the draw
is not), an `X` gate is applied to the masked qubits right before measurement
(`build_trex_twirled_circuit`), the measured bits are XOR-ed back with the mask
(`undo_trex_twirl`) and the three corrected count dictionaries are **summed**
(`aggregate_trex_instances`; 600 shots). **No calibration and no rescaling step
is applied.** This is the twirling half of TREx only: it converts a systematic
(here, one-sided 1→0) readout bias into a symmetric one but does not invert it,
so it can at best change which bitstring wins the argmax in ways that are not
guaranteed to be improvements. All TREx findings in this study apply to *this*
variant. TREx also receives 3× the shots of raw (a confound, §14).

---

## 9. Calibration design and what it measured

### 9.1 Design (`run_calibration.py`)

Calibration is measured on the same slots the packed jobs use and under the
same conditions (all slots active at once):

* **M3: 2 jobs** — every qubit prepared in |0⟩, then every qubit in |1⟩ — give
  each of the 20 qubits its 2×2 assignment matrix (one record, qubits 0–19).
* **MEM: 128 jobs** (2^widest-slot). In job `j` each slot prepares basis state
  `j mod 2^(slot width)`, so after 128 jobs every slot has seen all of its own
  basis states: **one full confusion matrix per slot** (3 records: qubits
  0–6, 7–13, 14–19).
* 200 shots per calibration job.
* The matrix of any circuit window inside a slot is derived by
  **marginalization** (`marginal_confusion_matrix`,
  `CalibrationStore.load_window`; M3: select the per-qubit matrices). The
  marginalization was verified against exact tensor products.
* Files: `calibration/m3_q0-1-…-19.json`, `calibration/mem_q0-1-2-3-4-5-6.json`,
  `…q7-8-…-13.json`, `…q14-15-…-19.json`, each with timestamp, shots,
  `calibration_billed_seconds` and the number of jobs.

### 9.2 Cost accounting used in the evaluation

A derived window carries the cost of the **stored calibration it comes from**,
not a per-window share: the calibration is a one-off cost shared by every
circuit, subject and repetition, and it is what the study actually spent.
Billed seconds: **M3 36.99 s (2 jobs)**, **MEM 2,367.67 s (128 jobs, shared by
all 3 slots)**. (An earlier version of the evaluator charged each subject only
the cost of calibrating its *first cluster's* window — 37 s/74 s — and did not
report any calibration cost for the multi-objective subjects; both defects were
found and fixed during the final audit, see §20.) The `calibration_wall_clock_seconds`
field is the *sum* of the wall-clock of 16 concurrently submitted jobs and has
no physical meaning on the real machine (≈ 249,000 s for MEM); the evaluation
uses the billed seconds.

### 9.3 What the machine's readout error actually is

From `calibration/m3_q0-…-19.json` (30 Oct 2026, 200 shots per state, so the
resolution is 1 count = 0.5 %):

* P(read 1 | prepared 0): **0 on all 20 qubits**.
* P(read 0 | prepared 1): min 0, **mean 0.005**, max 0.015.
* MEM, slot 0: mean diagonal 0.977; P(all 7 qubits correct | |1…1⟩) = 0.93.

So readout error is small, one-sided (|1⟩ decays to 0) and near the resolution
of the calibration. This is the single most important fact for interpreting
the results.

### 9.4 Drift

`run_calibration.py drift` re-measured M3 (2 jobs) at the end of the first
5-repetition phase (4 Nov 2026 02:57 UTC), 4.7 days after the initial
calibration: the **maximum absolute change of any per-qubit readout
probability is 0.01** (1–2 counts of 200, i.e. within calibration resolution).
Report: `calibration/drift/m3_drift_report.json` (plus
`calibration/drift/m3_q0-…-19.json`, the second calibration). **The drift check
was not repeated after repetitions 6–10** (executed 4–6 Nov; the resumed run
skips it when its report exists), so drift during the extension is not
directly measured.

### 9.5 Backend snapshots

`calibration/backend_snapshots/*.json` hold timestamped copies of the
properties the backend itself reports (taken at the start of every
hardware-execution script). On PIAST-Q **T1, T2 and frequency are `null`** — the
service does not expose them — so the snapshots only document the sessions
(timestamps, 20 qubits), not machine coherence.

---

## 10. Evaluation: metrics, statistics, how RQs are answered

All of this is classical post-processing (`evaluate_all.py`, `evaluation.py`,
`compare_all.py`, `statistics.py`); no backend is touched and it can be re-run
anywhere (§17).

### 10.1 Per-repetition metrics (one value per subject × method × repetition)

Single-objective (`evaluate_single_objective_combo`):

* **`qubo_energy`**: sum over clusters of the QUBO energy of the cluster's
  selected (argmax) bitstring. *Lower is better.* Noiseless optimum: see §4.4.
* **`probability_of_optimal`**: mean over clusters of the (corrected)
  probability of the brute-forced optimal bitstring (≤ 128 states per cluster).
  *Higher is better.*
* **`execution_cost`**: summed dataset cost of the final merged suite (union of
  selected tests).
* **`effectiveness.failure_rate`**: summed failure rate of the final suite.
  *Higher = more faults revealed.*
* **`execution_time_seconds`**: sum over the repetition's circuit records of
  their **billed QPU seconds** (wall-clock only on simulators). For TREx this
  includes the 3 passes.
* **`mitigation_overhead`**: calibration jobs/circuits, total shots and the
  billed calibration seconds (§9.2); zero for raw and TREx.
* **`classical_post_processing_seconds`**: local time of the correction (not a
  QPU quantity).

Multi-objective (`evaluate_multi_objective_combo`): each method's merged
selection is turned into a Pareto front (§2.5). Reference points are 3-tuples
`(−cost, statement coverage, fault coverage)`.

* **`hypervolume`** of the method's front with a reference point strictly worse
  than every observed point (cost margin 1, coverage 0). *Higher is better.*
* **`igd`**: inverted generational distance of the method's front to the
  **a-posteriori reference front = non-dominated union of all four methods'
  fronts for that repetition**. *Lower is better.* (It is relative to the
  methods being compared, not to an external optimum.)
* **`num_non_dominated`**: how many of the method's front points are on that
  reference front.
* execution time and overhead as above.

Note: raw, MEM and M3 are computed from **the same counts** of the same
repetition, so their results are *paired by construction*; TREx comes from its
own passes.

### 10.2 Statistics (`statistics.py`, `compare_all.py`)

Per subject and metric, the four methods (10 samples each) are compared with:
Shapiro–Wilk (α = 0.05) on each group; if all groups are normal, **one-way
ANOVA + Tukey HSD + Cohen's d**, otherwise **Kruskal–Wallis + Dunn's test with
Bonferroni adjustment + Vargha–Delaney Â12**. Reported: omnibus p, adjusted
pairwise p, effect size. There is **no correction across metrics or subjects**
(§14). The tests treat the four groups as independent (they do not exploit the
raw/MEM/M3 pairing).

**Cost (RQ3) is not tested.** Billed QPU time is a deterministic function of the
shots and the packing, not a random variable; a significance test on zero
variance only produces meaningless astronomical effect sizes (an earlier report
showed Cohen's d ≈ 1e15). The report gives mean, standard deviation and ratio to
raw instead.

### 10.3 Outputs

`results/evaluation/<subject>_qaoa_tcs.json` (per-repetition metrics),
`comparison_report.txt/.json` (RQ1, RQ2 tests and RQ3 cost summary per subject),
`ideal_reference.json` (noiseless reference).

---

## 11. What was actually run: configuration and chronology

### 11.1 Final configuration

| Item | Value |
|---|---|
| Subjects | `gsdtsr`, `paintcontrol` (single); `flex`, `gzip` (multi) — QAOA-TCS |
| Circuits per repetition | 79 + 24 + 105 + 61 = 269 raw, + 3 × 269 TREx copies |
| Methods | raw, MEM, M3, TREx |
| Repetitions | **10** (1–5 first, 6–10 added by `--resume`) |
| Shots per circuit | 200 (one batch; hardware cap and floor) |
| TREx passes | 3 (600 shots) |
| Packing | slots `[[0,7],[7,7],[14,6]]`, FFD, ≤ 16 jobs in flight |
| Calibration | M3 2 jobs, MEM 128 jobs, 200 shots each |
| Jobs | ≈ 2,510 execution (≈ 1,255 per 5 repetitions), 130 calibration, 2 drift |
| Billed QPU time | **≈ 13.6 h** by the study (account counter 13.67 h at the end, 205 s were already billed before launch); limit 25 h |
| Unit tests | 235 (all passing) |

### 11.2 Chronology (all times CET)

* **29 Oct** — timing and billing measurements (§6.2) and the decision to pack.
* **Fri 30 Oct, 11:26** — launch (Slurm job 1960) after the PCSS maintenance
  window (9–10). Calibration ran 11:33–16:08 (M3 first, then the 128 MEM jobs).
  Initial plan: 5 repetitions, 3 twirls (≈ 7.1 billed hours projected).
* **30 Oct, 16:40** — the stage died on an `httpx.ReadTimeout` raised by a job
  *submission*. Fix: submit-time timeouts are retried. Relaunched with
  `--resume`.
* **30 Oct, ≈ 17:10** — a second stop: the pilot gate projected 575 h because on
  resume no calibration is billed and the gate fell back to wall-clock (queue
  wait). Fix: skip the gate on resume. Relaunched (job 1969) at 17:11.
* **30 Oct – 1 Nov** — `gsdtsr` completed (5 repetitions) on Sat 31 Oct at 03:57
  and `paintcontrol` at 10:05; `flex` (multi-objective) was in progress when a
  job came back `status: failed` three times in a row and the run stopped at
  **Sun 1 Nov 02:41**, with `flex` at 116 of 525 raw records (cause on the
  service side, not analysed; the billed counter did not move after Sat ≈15:40). The run was
  **not resumed until Mon 2 Nov 20:12** (≈ 41 h lost, ≈ 53 h of idle time since
  the last billed job).
* **Mon 2 Nov – Wed 4 Nov (≈ 04:00)** — `flex` finished (Tue evening), `gzip`
  ran, and the 5-repetition phase completed ("ALL STAGES COMPLETED"), including
  the drift check (§9.4) and a first evaluation. A first analysis (R = 5)
  showed low power.
* **Wed 4 Nov, 20:41** — decision to extend to 10 repetitions
  (`target_repetitions: 10`, `total_hours: 80`, `--resume`; job 1990). Only
  repetitions 6–10 were executed.
* **Fri 6 Nov, ≈ 19:03** — run finished and the final evaluation/comparison
  were generated.
* **Final audit (9 Nov)** — the evaluator was corrected (§20) and the
  evaluation/comparison re-run on the unchanged raw data (quality metrics
  identical up to 1e-8; calibration-overhead and report format changed).

Observed throughput: a job started every ≈ 2–3 minutes on average, with
slow periods of ≈ 7 minutes per job; ≈ 7.3 days wall-clock in total.

---

## 12. Results

All numbers are from `results/evaluation/` (10 repetitions per cell; mean ± sd,
median where noted). "p_adj" is the adjusted pairwise p from the test chosen by
the Shapiro gate (§10.2).

### 12.1 Readout error on the machine (context)

See §9.3: mean P(0|1) = 0.5 %, P(1|0) = 0, max drift 0.01 over 4.7 days.

### 12.2 Single-objective subjects

`gsdtsr` (79 circuits, noiseless optimum energy −8.7432, noiseless mean
P(optimal) 0.3770):

| Metric | raw | MEM | M3 | TREx |
|---|---|---|---|---|
| QUBO energy (↓) | −8.5166 ± 0.020 | −8.5602 ± 0.018 | −8.5617 ± 0.024 | −8.5615 ± 0.019 |
| P(optimal) (↑) | 0.38658 ± 0.0025 | 0.39083 ± 0.0026 | 0.38921 ± 0.0026 | 0.38221 ± 0.0009 |
| suite `execution_cost` | 4.118e7 ± 2.3e6 | 4.252e7 ± 1.3e6 | 4.251e7 ± 1.3e6 | 4.314e7 ± 6.4e5 |
| suite `failure_rate` (= −energy) | 8.5166 | 8.5602 | 8.5617 | 8.5615 |

Tests: energy ANOVA p = 1.5e-5, raw vs MEM / M3 / TREx p_adj = 0.0002 / 0.0001 /
0.0001 (Cohen's d = 2.28 / 2.00 / 2.30), MEM, M3, TREx indistinguishable from
each other (p_adj > 0.99). P(optimal) ANOVA p = 2.2e-9: raw vs MEM p_adj = 0.001
(d = −1.65, MEM higher), raw vs M3 p_adj = 0.065 (n.s.), raw vs TREx p_adj =
0.0007 (d = +2.30, TREx **lower**), TREx vs MEM and vs M3 p_adj < 1e-6.
`execution_cost` ANOVA p = 0.041; only raw vs TREx is significant (p_adj = 0.028).
Effectiveness `failure_rate`: identical to the energy test (α = 0).

`paintcontrol` (24 circuits, noiseless optimum −8.5357, noiseless P(optimal)
0.7884):

| Metric | raw | MEM | M3 | TREx |
|---|---|---|---|---|
| QUBO energy (↓) | −8.5339 ± 0.0031 | −8.5343 ± 0.0021 | −8.5339 ± 0.0031 | −8.5348 ± 0.0021 |
| P(optimal) (↑) | 0.8187 ± 0.0100 | 0.8331 ± 0.0100 | 0.8289 ± 0.0100 | 0.8018 ± 0.0035 |
| suite `execution_cost` | 2.401e6 ± 1.8e4 | 2.405e6 ± 9.7e3 | 2.401e6 ± 1.8e4 | 2.401e6 ± 1.3e4 |
| suite `failure_rate` | 19.646 ± 0.036 | 19.654 ± 0.019 | 19.646 ± 0.036 | 19.648 ± 0.024 |

Tests: energy Kruskal–Wallis p = 0.86 (no difference; raw is already within
0.0018 of the noiseless optimum — a **ceiling effect**), `execution_cost` and
`failure_rate` p = 0.87 (no difference). P(optimal) ANOVA p = 1.2e-8: raw vs MEM
p_adj = 0.0056 (d = −1.40), raw vs M3 p_adj = 0.073 (n.s.), raw vs TREx p_adj =
0.001 (d = +2.23, TREx lower).

Observation not explained: for both subjects the hardware P(optimal) of raw
and MEM is *at or above* the noiseless value (0.387/0.391 vs 0.377;
0.819/0.833 vs 0.788). Correcting readout therefore does not move the
probability towards the ideal one; the deviations from the ideal circuit are
not (only) readout error. This was not investigated further.

### 12.3 Multi-objective subjects

`flex` (105 circuits):

| Metric | raw | MEM | M3 | TREx |
|---|---|---|---|---|
| Hypervolume (↑) | 2.516e14 ± 2.8e12 | 2.510e14 ± 3.4e12 | 2.509e14 ± 3.0e12 | 2.509e14 ± 3.2e12 |
| IGD (↓), mean (median) | 1.80e5 (1.77e4) | 1.78e5 (1.77e4) | 1.79e5 (1.76e4) | 2.01e5 (4.98e4) |
| non-dominated points | 459 ± 180 | 410 ± 230 | 435 ± 180 | 443 ± 220 |

Tests: hypervolume ANOVA p = 0.95; IGD Kruskal–Wallis p = 0.12; non-dominated
p = 0.54. **No significant difference anywhere.** (The IGD distribution is
heavy-tailed: sd ≈ 5e5 against medians of ≈ 2e4.)

`gzip` (61 circuits):

| Metric | raw | MEM | M3 | TREx |
|---|---|---|---|---|
| Hypervolume (↑) | 2.810e13 ± 3.8e12 | 2.818e13 ± 3.8e12 | 2.811e13 ± 3.8e12 | 2.781e13 ± 5.6e12 |
| IGD (↓), mean (median) | 5.75e5 (5.49e5) | 5.86e5 (6.04e5) | 5.91e5 (6.02e5) | **3.18e6 (3.75e6)** |
| non-dominated points | 158 ± 31 | 135 ± 55 | 108 ± 67 | 144 ± 64 |

Tests: hypervolume p = 0.92; non-dominated p = 0.087; **IGD Kruskal–Wallis p =
0.019 and raw vs TREx p_adj = 0.033 (Â12 = 0.16)**: TREx's front is much
further from the union reference front (both mean and median, ≈ 6× raw). The
cause was not investigated (TREx selects different bitstrings in the one
cluster with a quadratic term, or elsewhere; unconfirmed).

### 12.4 Cost (RQ3)

Billed QPU seconds **per repetition** (all 4 methods share the same raw
circuits; TREx adds its passes):

| Subject | raw = MEM = M3 | TREx | TREx / raw |
|---|---|---|---|
| `gsdtsr` | 268.2 s | 804.6 s | 3.00× |
| `paintcontrol` | 85.9 s | 265.5 s | 3.09× |
| `flex` | 585.3 s | 1,763.9 s | 3.01× |
| `gzip` | 215.2 s | 654.2 s | 3.04× |

(The ratios deviate from 3 because a job's billed time is shared by qubit
share inside mixed packs.) One-off calibration cost (shared by all subjects):
**M3 37.0 s (2 jobs); MEM 2,367.7 s (128 jobs).** Relative to a subject's raw
QPU time over the 10 repetitions (`gsdtsr` 2,682 s, `paintcontrol` 859 s,
`flex` 5,853 s, `gzip` 2,152 s):

| Subject | + M3 | + MEM | + TREx |
|---|---|---|---|
| `gsdtsr` | +1.4 % | +88 % | +200 % |
| `paintcontrol` | +4.3 % | +276 % | +209 % |
| `flex` | +0.6 % | +40 % | +201 % |
| `gzip` | +1.7 % | +110 % | +204 % |

(These are amortization views computed in this document from the numbers
above; the report itself lists the cost components separately.)

---

## 13. Answers to the research questions

**RQ1 — Does mitigation improve quality?** *Only marginally, and only on one
of four subjects.* On `gsdtsr` MEM, M3 and TREx all improve the energy (≈ 0.044
lower, p_adj ≤ 0.0002, d ≈ 2), closing ≈ 19 % of the gap between raw
(−8.517) and the noiseless optimum (−8.743); MEM also raises P(optimal) by
0.4 pp. On `paintcontrol` energy is unchanged (ceiling effect) and MEM raises
P(optimal) by 1.4 pp (M3 +1.0 pp, n.s.). On `flex` and `gzip` there is no
benefit on any Pareto metric. TREx never improves P(optimal) and lowers it on
both single-objective subjects; on `gzip` it significantly degrades IGD.
*Interpretation:* readout error on this machine is tiny and one-sided (§9.3),
the circuits are shallow and non-entangling, and the argmax is robust; the
effects are therefore small. Whether mitigation matters more on noisier
hardware or deeper circuits is not answered by this study.

**RQ2 — Does it translate to the selected suite?** *No consistent effect.* The
only significant suite-level result is `gsdtsr`'s effectiveness
(`failure_rate` up by 0.044 for MEM/M3/TREx), which restates RQ1 because α = 0
makes it the same quantity as the energy. `execution_cost` of the `gsdtsr`
suite is higher with mitigation (more tests selected; only raw vs TREx
significant). `paintcontrol`: no difference.

**RQ3 — Improvement per QPU second?** TREx: +200 % QPU time, no gain, some harm
→ dominated. MEM: the same energy gain as M3 (and a slightly larger, significant
P(optimal) gain) at +40 % to +276 % amortized over 10 repetitions per subject
(one-off 2,368 s). M3: the energy gain at +0.6 % to +4.3 % (one-off 37 s).
**M3 gives the best improvement per billed second**; MEM only pays if its
128-job calibration is amortized over many more circuits.

Sentences that can be defended: "mitigation of readout error provided at most
a small benefit for QAOA-TCS on this device"; "M3 achieved MEM's energy gain at
1/64 of the calibration cost"; "twirling-only TREx at 3 passes was not
beneficial". Sentences that cannot be defended from this data: general claims
about the value of mitigation on noisier devices; claims about TREx
as published (rescaled) rather than the twirling-only variant implemented here;
claims about subjects other than the four.

---

## 14. Threats to validity and known limitations

**Internal / construct**

1. **TREx implementation** is twirling-only (no learned scaling, §8.4); it
   gets 3× the shots of raw — the comparison is confounded by shot count.
2. **Paired vs unpaired:** raw/MEM/M3 share counts; the omnibus tests treat the
   groups as independent, so power for those comparisons is understated.
   A paired analysis was not performed.
3. **No multiple-comparison correction across metrics/subjects.** With ≈ 30
   tests, a p_adj of 0.03–0.06 (e.g. `gzip` IGD, `gsdtsr` execution_cost) should
   be read as suggestive only.
4. **Packing:** circuits run next to each other (crosstalk is possible) and not
   only on qubits 0–6; MEM/M3 were calibrated under the same packed conditions
   to mitigate this.
5. **Billed-time sharing:** a circuit's execution time is its qubit share of a
   job's billed time, not an independent measurement.
6. **Calibration resolution:** 200 shots per calibration state → 0.5 %
   quantization of the assignment probabilities, comparable to the errors
   being corrected.
7. **Repetitions are not time-homogeneous:** repetitions 1–5 and 6–10 ran days
   apart with the same calibration; drift was measured only up to 4 Nov (max
   change 0.01, at resolution).
8. **Single machine, one calibration set, one set of trained circuits;**
   QAOA-TCS training is not seeded, so the circuits cannot be regenerated
   bit-for-bit.
9. **Inherited quirks:** multi-objective quadratic terms effectively absent
   (§4.2); cluster counts and α values inherited without derivation;
   `trex_twirl_instances` default 32 in code (the study config sets 3);
   `DEFAULT_TREX_TWIRL_INSTANCES` is used only when the key is missing.
10. **Observed but unexplained:** hardware P(optimal) at/above the noiseless
    value; TREx degradation on `gzip`; `flex` IGD heavy tails.

**External**

11. Four subjects, `p = 1`, ≤ 7 qubits, mostly non-entangling circuits.
12. Readout error on this machine is atypically small and one-sided.

**Conclusion / operational**

13. The 41-hour stall and the two code fixes during the run (§11.2) did not
    alter data already saved (resume reuses records verbatim) but mean that
    repetitions 1–5 were collected under slightly different code revisions
    (retry behaviour only; no change to circuits, shots or analysis).
14. `status: failed` jobs (billed 0) were retried; their rate was not tabulated.

---

## 15. Every parameter and why it has that value

**A. Given by the platform/operators**

| Parameter | Value | Source |
|---|---|---|
| `shots_per_batch_cap` | 200 | operators: "a constraint of 200 shots per run" |
| `seconds_per_batch` | 18.4974 | measured billing of a 200-shot job (29 Oct); operators said ~15 s |
| QPU allowance | 25 h billed | project allocation |

**B. Inherited from the original QAOA-TCS paper / code (not re-derived)**

| Parameter | Value | Note |
|---|---|---|
| Max cluster size | 7 qubits | consistency with prior IGDec-QAOA work |
| QAOA depth `p` | 1 | paper: depth did not significantly affect quality |
| `COBYLA(maxiter=500)` | 500 | engineering default, no cited reason |
| α single-objective | `gsdtsr` 0, `paintcontrol` 0.45 | Optuna search in the paper; values carried over |
| α multi-objective | 0.5 | inherited |
| Ward target clusters | 60 / 16 / 50 | inherited engineering defaults |
| Repetitions "convention" | 10 | paper: stochastic algorithms run ten times |

**C. Chosen for this study**

| Parameter | Value | Reasoning |
|---|---|---|
| Subjects | 4 | fits the allowance and deadline; two per objective mode (§3.3) |
| `target_shots_per_circuit` | 200 | hardware floor and cap: one batch; billed cost per job is constant |
| `target_repetitions` | 10 | paper convention; 5 first, extended once 5 proved low-powered and wall-clock allowed |
| `trex_twirl_instances` | 3 | cheapest setting that still averages several masks; 32 (the old default) was unaffordable (§20) |
| `pack_slots` | `[[0,7],[7,7],[14,6]]` | 7 = widest circuit; 7+7+6 = 20 qubits; slots keep calibration windows fixed |
| `MAX_JOBS_IN_FLIGHT` | 16 | overlap queue waits; margin of the budget stop covers them |
| `MAX_JOB_ATTEMPTS` | 3 | retry transient `failed` jobs (they bill 0) |
| Transport retry | every 5 s, ≈ 30 min | survive VPN/network blips without resubmitting |
| `optimization_level` | 3 | highest standard transpiler optimization for hardware |
| `calibration_shots_per_circuit` | 200 | one batch per calibration state |
| `total_hours` (planner cap) | 80 | only a ceiling above the planner's 56.9 h upper bound |
| `QPU_HOURS_LIMIT` | 25 | allowance; stop margin (16+2)×18.4974 s |
| `physical_qubits` | `[0..6]` | only for single un-packed circuits; not used for the results |

---

## 16. Repository map and data dictionary

```
PiastQ4QAOA/
├── README.md                    # user guide: setup, run, reproduce
├── EXPLAINATION.md              # this document
├── PROJECT_SUMMARY.md           # short project/RQ/limitations summary
├── requirements.txt             # pinned environment (Python 3.12)
├── run_all.py                   # full pipeline, stages in §7.4
├── run_cluster.sh               # Slurm launcher; the only place that enables the real machine
├── configs/
│   ├── backend.yaml             # simulator by default; pack_slots; env vars override
│   └── execution_plan.yaml      # budget planner + evaluation configuration (10 reps, 3 twirls)
├── datasets/                    # input data (unchanged)
├── trained_qaoa_circuits/       # frozen .qpy circuits + *_qubos.json (QUBO coefficients)
├── src/
│   ├── qaoa_tcs/                # single_obj.py, multi_obj.py: clustering, QUBO, train/qubos/execute
│   ├── igdec_qaoa/              # implemented, not run on hardware (§18)
│   └── piastq_execution/        # shared library
│       ├── backend_config.py    # backend/token resolution, PCSS hardening (retries)
│       ├── packing.py           # slots, first-fit-decreasing, joint circuits, marginalization
│       ├── raw_counts.py        # batches, records, concurrency, packed execution, resume
│       ├── qpu_usage.py         # billed-time reading and budget stop
│       ├── run_calibration.py   # M3 (2 jobs) + MEM (128 jobs) + drift check
│       ├── mitigation.py        # MEM, M3, TREx, CalibrationStore (windows by marginalization)
│       ├── evaluation.py, evaluate_all.py, compare_all.py, statistics.py
│       ├── ideal_reference.py   # noiseless reference values (§4.4)
│       ├── budget_planner.py, backend_calibration_snapshot.py, qubit_layout.py,
│       ├── qubo_io.py, sir_metrics.py, seeding.py
├── results/
│   ├── qaoa_tcs/<subject>/      # raw data (below)
│   └── evaluation/              # evaluation + report (below)
├── calibration/                 # MEM/M3 matrices, drift/, backend_snapshots/
├── execution_plan/              # planner report
├── tests/                       # 235 unit tests (no real hardware)
└── logs/                        # git-ignored run logs (not in a clone)
```

**`results/qaoa_tcs/<subject>/`**

* `<subject>-rep-1-raw_counts.jsonl` — one JSON record per circuit execution
  (raw pass): `circuit_id`, `cluster_id`, `iteration_id` (= repetition 1…10),
  `physical_qubit_mapping` (its real window), `batches` (counts, job id,
  billed seconds, timestamp), `aggregated_counts`, `total_billed_seconds`, … .
  Record counts: `gsdtsr` 790, `paintcontrol` 240, `flex` 1,050, `gzip` 610
  (= circuits × 10).
* `<subject>-rep-1-trex-raw_counts.jsonl` — same for the TREx passes, with
  `twirl_mask`; 3× as many records (2,370 / 720 / 3,150 / 1,830).
* `<subject>-rep-1.json`, `<subject>-rep-1-subsuites.json` — per-repetition
  selected suites / cluster assignments from the raw pass (single-objective:
  `selected_test_suite_i`; multi-objective: Pareto fronts).

**`results/evaluation/`** — `<subject>_qaoa_tcs.json` (per method, list of the
10 per-repetition metric records), `comparison_report.txt/.json`,
`ideal_reference.json`.

**`calibration/`** — see §9.1, §9.4, §9.5.

---

## 17. How to reproduce, re-evaluate or extend

**Environment.** `python3.12 -m venv qiskit_env && source qiskit_env/bin/activate
&& pip install -r requirements.txt`.

**Re-evaluate from the stored data (no hardware, ~1 minute).**

```bash
cd src/piastq_execution
python evaluate_all.py      # -> results/evaluation/<subject>_qaoa_tcs.json
python compare_all.py       # -> results/evaluation/comparison_report.{txt,json}
python ideal_reference.py   # -> results/evaluation/ideal_reference.json
```

**Free dry run of the whole pipeline on a simulator** (never touches hardware;
uses the committed defaults): `python run_all.py` from the repository root. To
avoid mixing a dry run with the study's data, run it in a **copy** of the
repository (the writers append, and `run_all.py` moves leftover raw-counts files
to `results/_archive_<time>/` on a fresh start).

**Real machine** (needs account, VPN, token): copy the repo and `~/.piastq_token`
to the cluster, then `sbatch run_cluster.sh` (fresh) or
`sbatch run_cluster.sh --resume`. Do **not** edit `configs/backend.yaml` or
`run_all.py` to switch on hardware. **Do not run it inside this repository
without moving `results/` and `calibration/` away first**: a fresh run
archives/appends over the study's data.

**Extending the repetitions.** Raise `target_repetitions` (and keep
`total_hours` above the planner's upper bound), then `--resume`: finished
repetitions are reused, new ones appended. Extending the number of twirls would
require re-running every TREx pass (the masks are part of each record) and is
wasteful.

**Unit tests.** `python -m unittest discover -s tests` (235 tests, a few
seconds, entirely synthetic).

---

## 18. Out of scope but implemented (IGDec-QAOA, other datasets)

`src/igdec_qaoa/` holds IGDec-QAOA (impact-guided decomposition: a random
initial solution, repeated re-ranking of tests by impact, QAOA on the next 7
highest-impact tests, keep the best solution). Its trained circuits exist for
`iofrol` (10 samplings × 1,230 circuits) and `paintcontrol` (10 × 30); training
code for `gsdtsr`, `elevator_o2`, `elevator_o3` exists but was not run. The
hardware and evaluation scripts support it (with a replay of the training-fixed
subproblem sequence for the final suite), and it is covered by unit tests, but
it was **not executed on PIAST-Q** and `results/igdec_qaoa/` is empty. Other
QAOA-TCS datasets (`iofrol` 443 circuits, `elevator` 881, `elevator2` 56 — not
evaluable, `grep` 139, `sed` 77) are defined in the config but excluded from the
plan. The full study design (14 dataset × algorithm combinations) is described
in `README.md` §11 ("Out-of-scope combinations").

---

## 19. Glossary

* **Argmax** — the most frequent bitstring; the cluster's answer.
* **Billed time** — QPU seconds counted against the allowance; independent of
  queue wait and (here) of the qubit count.
* **Confusion/assignment matrix** — `P(measured | prepared)`; MEM uses the full
  one, M3 per-qubit ones.
* **Hypervolume (HV) / IGD** — scores of a Pareto front (higher HV better; lower
  IGD better; IGD here is relative to the union of the compared methods' fronts).
* **Packing / slot** — several independent circuits on disjoint qubits of one
  job / the fixed qubit ranges used for it.
* **p** — QAOA depth (`rep_1` in file names = p = 1).
* **Repetition** — one full re-execution of all circuits (`iteration_id`).
* **TREx** — here: measurement twirling with classical un-flipping, no rescaling.
* **Twirl pass / mask** — one extra execution with a random X pattern before
  measurement.

---

## 20. Decision log

| # | Decision | Why |
|---|---|---|
| 1 | QAOA-TCS only, 4 subjects | allowance and deadline; two per objective mode |
| 2 | Billed time as the budget, wall-clock as the real constraint | measured 18.4974 s/job independent of qubits; queue overhead 2–5 min/job |
| 3 | Pack circuits into 20-qubit jobs | 5× fewer jobs; same shots and independence per circuit |
| 4 | Calibrate MEM per slot (128 jobs) and M3 for all qubits (2 jobs) | consistent with packed execution; windows by marginalization |
| 5 | Up to 16 jobs in flight | overlap queue waits; the machine still runs them serially |
| 6 | 3 TREx twirls (not 32) | affordability: each twirl is a full extra pass over all circuits |
| 7 | 5 repetitions first, then 10 via `--resume` | first results had low power; time allowed; resume makes extension exact |
| 8 | Real-machine switches only in `run_cluster.sh` | repo stays on the simulator; no accidental hardware use |
| 9 | Retry submit-time timeouts (during the run) | a `ReadTimeout` killed the run once; cost of an orphan job ≈ 18 s |
| 10 | Skip pilot gate on resume (during the run) | gate used wall-clock (queue wait) and falsely projected 575 h |
| 11 | RQ3a merged into RQ1; RQ3b → RQ3; RQ4 dropped | same comparison; IGDec-QAOA not run |
| 12 | Final audit fixes (9 Nov): (a) calibration overhead = cost of the stored shared calibration, also reported for multi-objective subjects; (b) `compare_all` no longer treats its own report as a subject; (c) RQ3 cost reported descriptively (zero variance) with RQ1/RQ2/RQ3 labels; (d) QUBO coefficient files added to the package; (e) docs and stale comments refreshed | found while auditing code and results; the quality metrics were unchanged by the re-evaluation (max difference 1.6e-8, no argmax changed) |
