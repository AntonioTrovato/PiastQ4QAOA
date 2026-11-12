# PIAST-Q4QAOA — Real-Hardware Replication Package

Replication package of an empirical study that runs **QAOA-based Regression Test
Case Selection (QAOA-TCS)** on the **real PIAST-Q 20-qubit trapped-ion quantum
computer** and measures how much **readout-error mitigation** (full MEM, M3,
TREx) improves it, and at what cost in billed QPU time.

> **Status.** The experiments are finished (30 Oct – 6 Nov 2026): QAOA-TCS on
> four subjects (`gsdtsr`, `paintcontrol` single-objective; `flex`, `gzip`
> multi-objective), methods raw / MEM / M3 / TREx, **10 repetitions**, 200 shots
> per circuit, 3 TREx passes, **13.6 billed QPU hours of the 25 available**.
> All raw data, calibrations, evaluation files and the statistical report are
> committed. **The detailed account — design, every decision, final results,
> limitations — is in [`EXPLAINATION.md`](EXPLAINATION.md); a short summary is in
> [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).** This README is the practical
> guide: layout, data, how to re-run and reproduce.

## Index

- [1. The study in brief](#1-the-study-in-brief)
- [2. Directory layout](#2-directory-layout)
- [3. Requirements](#3-requirements)
- [4. The data and how to read it](#4-the-data-and-how-to-read-it)
- [5. Reproduce, re-evaluate, re-run](#5-reproduce-re-evaluate-re-run)
- [6. The pipeline (`run_all.py`) and its safety mechanisms](#6-the-pipeline-run_allpy-and-its-safety-mechanisms)
- [7. Packing: several circuits per hardware job](#7-packing-several-circuits-per-hardware-job)
- [8. Mitigation and calibration](#8-mitigation-and-calibration)
- [9. Evaluation and statistics](#9-evaluation-and-statistics)
- [10. Environment variables](#10-environment-variables)
- [11. Training and scope](#11-training-and-scope)
- [12. Tests](#12-tests)
- [13. Parameters](#13-parameters)

---

## 1. The study in brief

| | |
|---|---|
| Algorithm | QAOA-TCS: deterministic Ward clustering into ≤ 7-test clusters, one pre-trained `p = 1` QAOA circuit per cluster, results merged (single-objective: union; multi-objective: Pareto front by Additional-Greedy prefixes) |
| Subjects | `gsdtsr` (79 circuits), `paintcontrol` (24) — single-objective; `flex` (105), `gzip` (61) — multi-objective. 269 circuits, 1–7 qubits |
| Machine | PIAST-Q (20 qubits) through the PCSS service (`pcss-qapi`); ≤ 200 shots per run; 25 QPU h billed |
| Methods | raw, **MEM** (full confusion matrix, NNLS), **M3** (`mthree`), **TREx** (measurement twirling, no rescaling, 3 passes) |
| Repetitions | 10 (the first 5 were run, then extended with `--resume`) |
| Research questions | **RQ1** quality (energy, P(optimal), HV/IGD) · **RQ2** suite-level effectiveness · **RQ3** improvement per billed QPU second |
| Result in one line | Readout error on the machine is tiny (mean 0.5 %, one-sided); mitigation gave at most a small gain on one single-objective subject; **M3 matches MEM's energy gain at 1/64 of the calibration cost**; twirling-only TREx costs 3× and does not help |

Numbers, tables and the discussion of validity: `EXPLAINATION.md` §12–§14.

---

## 2. Directory layout

```
PiastQ4QAOA/
├── README.md  EXPLAINATION.md  PROJECT_SUMMARY.md
├── requirements.txt              # pinned environment (Python 3.12)
├── run_all.py                    # the one-click pipeline (§6)
├── run_cluster.sh                # Slurm launcher; the only place enabling the real machine
├── configs/
│   ├── backend.yaml              # simulator by default; pack_slots; env vars override
│   └── execution_plan.yaml       # budget planner + evaluation configuration (10 reps, 3 twirls)
├── datasets/                     # input data (unchanged from the original study)
├── trained_qaoa_circuits/
│   └── qaoa_tcs/<subject>/rep_1/ # frozen .qpy circuits + <subject>_rep1_qubos.json
├── src/
│   ├── qaoa_tcs/                 # single_obj.py, multi_obj.py: clustering, QUBO, train / qubos / execute
│   └── piastq_execution/         # shared library
│       ├── backend_config.py     #   backend/token resolution; PCSS retry hardening
│       ├── packing.py            #   slots, first-fit-decreasing, joint circuits, marginalization
│       ├── raw_counts.py         #   batches, records, concurrency, packed execution, resume
│       ├── qpu_usage.py          #   billed-time reading and budget stop
│       ├── run_calibration.py    #   M3 (2 jobs) + MEM (128 jobs) + `drift`
│       ├── mitigation.py         #   MEM, M3, TREx, CalibrationStore (windows by marginalization)
│       ├── evaluation.py  evaluate_all.py  compare_all.py  statistics.py
│       ├── ideal_reference.py    #   noiseless reference values of the trained circuits
│       ├── budget_planner.py  backend_calibration_snapshot.py  qubit_layout.py
│       └── qubo_io.py  sir_metrics.py  seeding.py
├── results/
│   ├── qaoa_tcs/<subject>/       # raw data and per-repetition selections (§4)
│   └── evaluation/               # evaluation, statistical report, ideal reference (§4)
├── calibration/                  # MEM/M3 matrices, drift/, backend_snapshots/
├── execution_plan/               # budget planner report
├── tests/                        # 229 unit tests (never touch real hardware)
└── logs/                         # run logs; git-ignored, so absent from a clone
```

Every script in `src/` is run from its own folder (`cd src/qaoa_tcs`, …): the
relative paths `../../datasets`, `../../results` depend on it.

---

## 3. Requirements

* **Python 3.12** (≥ 3.11 is required by `pcss-qapi`). One environment serves
  tests, dry runs, evaluation and the real run.

```bash
python3.12 -m venv qiskit_env
source qiskit_env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the tested set (`qiskit` 1.4.6, `qiskit-aqt-provider`
1.15.0, `pcss-qapi` 0.3.2, `qiskit-aer`, `mthree` 3.0.0, `scipy`, …). `qiskit`
must stay below 2.x. The trained circuits are `.qpy` files and the results are
plain JSON/JSONL, so nothing stored depends on library versions.

---

## 4. The data and how to read it

| Where | What | How to read it |
|---|---|---|
| `results/qaoa_tcs/<subject>/<subject>-rep-1-raw_counts.jsonl` | One line per circuit execution (raw pass): `circuit_id`, `cluster_id`, `iteration_id` (**= repetition 1…10**), `physical_qubit_mapping` (the circuit's real qubits), `batches` (full bitstring counts, job id, timestamp, billed seconds), `aggregated_counts`, `total_billed_seconds`, `transpiled_stats` | Source of truth for all post-hoc mitigation. Counts: `gsdtsr` 790, `paintcontrol` 240, `flex` 1,050, `gzip` 610 |
| `…-rep-1-trex-raw_counts.jsonl` | Same, one line per TREx pass, with `twirl_mask` | Undo the mask, sum the 3 passes (`mitigation.aggregate_trex_records`). 2,370 / 720 / 3,150 / 1,830 lines |
| `…-rep-1.json`, `…-rep-1-subsuites.json` | Per-repetition selections / cluster assignments (single-objective suites, multi-objective Pareto fronts) from the raw pass | Per-repetition outputs of the execution scripts |
| `calibration/m3_q0-…-19.json` | M3 per-qubit 2×2 assignment matrices for qubits 0–19 (2 jobs) | Start-of-run readout calibration |
| `calibration/mem_q0-…-6.json`, `mem_q7-…-13.json`, `mem_q14-…-19.json` | Full MEM confusion matrix per slot (128 jobs, shared) | Same |
| `calibration/drift/` | Second M3 calibration at the end of the first 5-repetition phase and `m3_drift_report.json` (max change 0.01) | Readout stability; not repeated after repetitions 6–10 |
| `calibration/backend_snapshots/` | Timestamped backend-reported properties per session; PIAST-Q reports `null` for T1/T2/frequency | Documents the sessions only |
| `trained_qaoa_circuits/qaoa_tcs/<subject>/rep_1/` | `*_clusterN.qpy` frozen circuits and `*_qubos.json` QUBO coefficients | Needed by the evaluation |
| `results/evaluation/<subject>_qaoa_tcs.json` | `{method: [per-repetition metric record, …10]}` | Input of the statistics |
| `results/evaluation/comparison_report.{txt,json}` | RQ-mapped statistics (below) | |
| `results/evaluation/ideal_reference.json` | Noiseless P(optimal), argmax correctness and energies of the trained circuits | Reference for how far hardware results are from ideal |
| `execution_plan/execution_plan_report.{txt,json}` | Budget planner output for the shipped configuration | Upper bound ≈ 56.9 h (one circuit per job); real use 13.6 h |

**File names:** `rep-1` in file names is the **QAOA depth `p = 1`**, not a
repetition. Repetitions are the `iteration_id` values 1…10 inside the files.

**Reading `comparison_report.txt`.** One block per subject: **RQ1** (quality)
and **RQ2** (effectiveness; single-objective only) list, per metric, the Shapiro
gate result (`normal=`), the omnibus test (ANOVA or Kruskal–Wallis) with p, and
one line per method pair: `p_adj` (Tukey, or Dunn + Bonferroni) and the effect
size (Cohen's d, or Vargha–Delaney `a12`: 0.5 = no effect). **RQ3** (cost) lists,
per cost component and method, mean, standard deviation and ratio to raw of the
billed QPU seconds. Cost is deterministic (a function of shots and packing), so
it is described rather than tested. For `gsdtsr` (QUBO weight α = 0) the RQ2
`failure_rate` is the energy with the opposite sign — see `EXPLAINATION.md`
§4.2. There is no correction for multiple comparisons.

---

## 5. Reproduce, re-evaluate, re-run

**Never run the pipeline against the real machine from inside this repository
copy without first moving `results/` and `calibration/` away**: the writers
append to existing files, and a fresh start archives them.

### A. Unit tests (no backend, a few seconds)

```bash
python -m unittest discover -s tests          # 229 tests
```

### B. Re-evaluate from the stored data (no hardware, about a minute)

```bash
cd src/piastq_execution
python evaluate_all.py        # -> results/evaluation/<subject>_qaoa_tcs.json
python compare_all.py         # -> results/evaluation/comparison_report.{txt,json}
python ideal_reference.py     # -> results/evaluation/ideal_reference.json
```

The QUBO coefficient files (`*_qubos.json`) are part of the package. If they are
ever missing (circuits without them), rebuild them without retraining:
`cd src/qaoa_tcs && python single_obj.py qubos && python multi_obj.py qubos`
(each is verified against the saved circuits).

### C. Free dry run of the whole pipeline on a simulator

Work in a **copy** of the repository. With the shipped defaults
(`backend_name: offline_simulator_no_noise`, no `PIASTQ_*` variables set):

```bash
python run_all.py                              # prints ALL STAGES COMPLETED
```

### D. Real PIAST-Q run (needs account, VPN to reach the login node, token)

The study ran on the PCSS Slurm cluster (partition `all`, no time limit).

1. Copy the repo to the cluster (`rsync -az --exclude qiskit_env --exclude .git
   ./ <user>@<login-node>:PiastQ4QAOA/`), build the environment there (§3;
   Python 3.12 via `uv python install 3.12`, `uv venv --seed --python 3.12
   qiskit_env`, `pip install -r requirements.txt`), and put the token in
   `~/.piastq_token` (`chmod 600`; never commit it).
2. `cd PiastQ4QAOA && sbatch run_cluster.sh`. After any stop, continue with
   `sbatch run_cluster.sh --resume`. `run_cluster.sh` exports the real-machine
   switches (§10); **the repo files are never edited to enable hardware.**
3. Watch with `squeue -u $USER`, the files in `results/` and `calibration/` and
   the service dashboard. The Slurm log (`logs/slurm_<id>.out`) is buffered and
   can lag. Cancel with `scancel <id>`.

Locally instead: `PIASTQ_BACKEND_NAME=piast_q PIASTQ_ALLOW_REAL_HARDWARE=1
PIASTQ_API_TOKEN=<token> python run_all.py` from the Python 3.12 environment
with the VPN on.

### E. Extending the repetitions

Raise `target_repetitions` in `configs/execution_plan.yaml` (keep
`total_hours` above the planner's upper bound, otherwise it drops TREx or
repetitions), then `--resume`: finished repetitions are reused verbatim and only
the new ones are executed (this is how the study went from 5 to 10). Changing
the number of TREx passes later is wasteful: every pass would have to be re-run.

---

## 6. The pipeline (`run_all.py`) and its safety mechanisms

Stages, in order (a failing stage stops the run): (1) unit tests, run with all
`PIASTQ_*` variables stripped; (1b) **QUBO backfill**; (2) budget plan;
(3) **calibration** (+ backend snapshot); (4) **pilot gate**; (5) QAOA-TCS
single-objective; (6) QAOA-TCS multi-objective; (7) **drift check** (2 M3 jobs); (8) evaluation and
statistical comparison; (9) noiseless reference simulations.

* **Which backend:** decided by `configs/backend.yaml` or the `PIASTQ_*`
  variables (which win). Without `PIASTQ_ALLOW_REAL_HARDWARE=1` the script
  refuses to start on anything but an offline simulator.
* **Billed-time budget stop:** before every batch the account's billed time is
  read from the service (`qpu_usage.check_budget`); a batch that would cross the
  limit (25 h minus a margin of `(16 + 2) × 18.4974 s`) is not submitted.
* **Pilot gate:** after calibration the billed seconds per job are projected to
  the whole run and the run aborts if the projection exceeds 25 h. Skipped on
  `--resume` (nothing is billed then).
* **Retries:** a job returning `status: failed` is resubmitted up to 3 times
  (failed jobs bill 0); transport errors while waiting for a result are retried
  every 5 s for ≈ 30 min without resubmitting; submit timeouts / connect errors
  are retried too (at worst one orphan job of ≈ 18.5 s).
* **Concurrency:** up to `MAX_JOBS_IN_FLIGHT = 16` jobs queued at once; the
  machine runs them one at a time.
* **`--resume`:** reuses every circuit already in the raw-counts files (keyed by
  `(circuit_id, iteration_id)`), repairs a half-written last line, skips finished
  jobs, finished calibration and an existing drift report, and skips the pilot
  gate. Without `--resume` a start archives leftover raw-counts files to
  `results/_archive_<time>/`.
* **Output:** console, `logs/run_all_<time>.log`, `logs/run_summary_<time>.json`.

---

## 7. Packing: several circuits per hardware job

On PIAST-Q a job is billed **18.4974 s per 200 shots whatever the number of
qubits** (measured 29 Oct 2026), yet each job occupies the machine for minutes
of unbilled overhead and waits in a shared queue; one circuit per job (≈ 4,400
jobs) would have taken 9–15 days. `packing.py` therefore places several
independent circuits side by side on **disjoint qubits** of one 20-qubit job,
inside three fixed slots (`pack_slots: [[0,7],[7,7],[14,6]]`), using a
deterministic first-fit-decreasing rule (90–98 % fill). After the run the joint
counts are split per circuit by marginalizing the classical bits. Each circuit
keeps **its own 200 shots, counts and record** (`physical_qubit_mapping` gives its
real window); a job's billed time is shared by qubit share. Costs of the design,
to state in the paper: neighbours run simultaneously (possible readout
crosstalk), circuits are not confined to qubits 0–6, and per-circuit execution
time is a share of the job's billed time. Nearly all trained circuits contain no
two-qubit gate, so packed circuits do not interact computationally.

---

## 8. Mitigation and calibration

* **MEM:** full `2^w × 2^w` confusion matrix, solved by non-negative least
  squares. **M3:** per-qubit matrices combined on the observed outcomes
  (`mthree`; quasi-probabilities, not clipped). Both reuse **the raw counts**.
* **TREx:** random `X` mask before measurement, un-flipped classically, 3 passes
  summed; **no calibration or rescaling step** — the twirling half of TREx only.
  Three times the shots of raw.
* **Calibration** (on the packed slots, all active at once): **M3 = 2 jobs**
  (all |0⟩, all |1⟩), **MEM = 128 jobs** (each slot cycles its own basis
  states), 200 shots each. A circuit's window inside a slot gets its matrix by
  **marginalization** (`CalibrationStore.load_window`). Cost: M3 37.0 s billed,
  MEM 2,367.7 s billed, shared by every subject and repetition.
* **Measured readout error:** P(1|0) = 0 on all qubits, P(0|1) mean 0.5 %,
  max 1.5 %; resolution 0.5 % (200 shots).

---

## 9. Evaluation and statistics

Per subject × method × repetition: single-objective — `qubo_energy` (sum of the
selected bitstrings' energies), `probability_of_optimal` (mean over clusters,
optimum brute-forced), suite `execution_cost` and `effectiveness.failure_rate`;
multi-objective — `hypervolume`, `igd` (to the union reference front of the
four methods), `num_non_dominated`; all — `execution_time_seconds` (billed
seconds), `mitigation_overhead`. The cluster's answer is the **argmax** of the
(corrected) distribution with a deterministic tie-break. Tests: Shapiro–Wilk
(0.05) → ANOVA + Tukey + Cohen's d, otherwise Kruskal–Wallis + Dunn
(Bonferroni) + Vargha–Delaney Â12. Cost is described, not tested. Full
definitions: `EXPLAINATION.md` §10.

---

## 10. Environment variables

| Variable | Set by | Effect |
|---|---|---|
| `PIASTQ_BACKEND_NAME` | `run_cluster.sh` | overrides `backend_name` (`piast_q` = real machine) |
| `PIASTQ_ALLOW_REAL_HARDWARE=1` | `run_cluster.sh` | allows `run_all.py` to start on a non-simulator |
| `PIASTQ_API_TOKEN` | `run_cluster.sh` (from `~/.piastq_token`) | PCSS token; never stored in the repo |
| `PIASTQ_QPU_LIMIT_SECONDS` | `run_all.py` | billed-time stop for every batch (limit minus margin) |
| `PIASTQ_MAX_IN_FLIGHT` | `run_all.py` (real backend) | jobs queued at once (16) |
| `PIASTQ_RESUME=1` | `run_all.py --resume` | reuse finished circuits |

---

## 11. Training and scope

Only the four subjects of the study are kept in the repository; everything else
of the original package (IGDec-QAOA, the datasets `iofrol` and `elevator`, the
SIR programs `grep` and `sed`) was removed because it was never executed on
PIAST-Q (it needed several times the available machine time).

| Algorithm | Mode | Subjects | Circuits |
|---|---|---|---|
| QAOA-TCS | single-objective | `gsdtsr` 79, `paintcontrol` 24 | 103 |
| QAOA-TCS | multi-objective | `flex` 105, `gzip` 61 | 166 |

**Training** (not part of the hardware study; ideal simulator,
`COBYLA(maxiter=500)`, `p = 1`; the training code sets no seed, the
`gsdtsr`/`paintcontrol` circuits of the study were retrained on 9 Nov 2026 with
`algorithm_globals.random_seed = 42` set from outside):

```bash
cd src/qaoa_tcs && python single_obj.py train && python multi_obj.py train
```

Plain `python <script>.py` (no argument) **executes** trained circuits on the
configured backend; `qubos` rebuilds the QUBO coefficient files.

---

## 12. Tests

`python -m unittest discover -s tests` — **229 tests**, a few seconds, entirely
synthetic (toy 1–3 qubit circuits and QUBOs, fake samplers and a fake noisy
machine, tiny Pareto fronts); none touches the real service. Test modules that
touch execution code import `tests/hermetic_env.py` first, which scrubs every
`PIASTQ_*` variable.

---

## 13. Parameters

The full table (what is inherited from the original study, what was decided
here, and why) is `EXPLAINATION.md` §15. The ones that matter for reproduction:
`target_repetitions: 10`, `target_shots_per_circuit: 200`,
`trex_twirl_instances: 3`, `seconds_per_batch: 18.4974`, `pack_slots`
`[[0,7],[7,7],[14,6]]`, `MAX_JOBS_IN_FLIGHT: 16`, `QPU_HOURS_LIMIT: 25`,
`optimization_level=3`, `COBYLA(maxiter=500)`, `p = 1`, cluster size ≤ 7.
