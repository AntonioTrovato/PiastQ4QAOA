# PIAST-Q4QAOA: Project Summary

## 1. Context

Regression test case selection (TCS) chooses a subset of a test suite that is cheap to run yet still effective at catching faults. It can be written as a QUBO (a binary quadratic optimisation problem) and approximately solved with the Quantum Approximate Optimization Algorithm (QAOA). QAOA-TCS makes large suites fit on small quantum circuits (at most 7 qubits each) by clustering the suite once, solving each cluster independently with a pre-trained `p = 1` circuit and merging the results. It has a single-objective variant (cost and failure rate) and a multi-objective variant (cost, statement coverage, fault coverage), the latter producing a Pareto front of trade-offs.

QAOA-TCS had so far been evaluated mainly on simulators. On real hardware the circuit output is distorted by noise. For shallow circuits such as these (the trained circuits are essentially products of single-qubit gates: only 2 of 269 contain a two-qubit gate), a dominant and well-understood component is **readout error**. Three mitigation techniques are compared with the unmitigated baseline (raw):

- **MEM (full measurement-error mitigation)**: every n-qubit basis state is prepared and measured (2^n calibration circuits) to build the full confusion matrix, which is inverted (non-negative least squares) on any measured distribution. Calibration cost grows exponentially.
- **M3**: the same idea with per-qubit assignment probabilities and a correction restricted to the observed outcomes. Far cheaper (2 calibration jobs for all 20 qubits).
- **TREx (measurement twirling)**: a random subset of qubits is flipped with X gates before measurement and the flip is undone classically; several random patterns are summed. It needs no calibration but every pattern is an additional hardware run. The variant implemented here is the twirling half only (no learned rescaling).

The target machine is PIAST-Q, a 20-qubit trapped-ion quantum computer (reached through the PCSS service), limited to 200 shots per run, with 25 hours of billed QPU time available.

## 2. Goal and research questions

**Goal:** to evaluate QAOA-based test case selection on real quantum hardware and to quantify how much readout-error mitigation improves it, and at what cost.

| RQ | Question | Why it matters |
|---|---|---|
| **RQ1** | Does mitigation (MEM, M3, TREx) improve solution quality (QUBO energy, probability of the optimal bitstring; hypervolume / IGD / non-dominated count for multi-objective) over raw hardware output? | Establishes whether mitigation is needed at all for these circuits. |
| **RQ2** | Does better quality translate into a better selected test suite (suite cost and effectiveness for single-objective programs)? | Links quantum-level improvements to the practical test-selection outcome. |
| **RQ3** | Which technique gives the best improvement per second of billed QPU time? | QPU time is the scarce resource. A technique can win RQ1 and still lose here. |

Subjects: two multi-objective programs (`gzip`, `flex`) and two single-objective datasets (`paintcontrol`, `gsdtsr`), so that conclusions are not tied to one problem type.

## 3. What was run

QAOA-TCS on the four subjects (269 circuits of 1–7 qubits), all four methods, **10 repetitions** (5 first, extended to 10 with `--resume`), 200 shots per circuit, 3 TREx passes, on the real PIAST-Q between 30 Oct and 6 Nov 2026: about 2,510 execution jobs, 130 calibration jobs and 2 drift jobs, **13.6 billed QPU hours of 25** (about 7.3 days wall-clock including a 41-hour stall caused by a service-side job failure). Several independent circuits were packed side by side on disjoint qubits of each 20-qubit job to make the wall-clock fit the deadline.

## 4. Results in brief

- Readout error on the machine is **very small and one-sided**: P(read 1 | prepared 0) = 0 on all qubits, P(read 0 | prepared 1) mean 0.5 %, max 1.5 %; the maximum drift over 4.7 days was 0.01 (at calibration resolution).
- **RQ1.** No general improvement. On `gsdtsr` all three methods lower the QUBO energy from −8.517 to ≈ −8.561 (p_adj ≤ 0.0002, d ≈ 2; noiseless optimum −8.743), and MEM raises the optimum probability by 0.4 pp (`gsdtsr`) and 1.4 pp (`paintcontrol`). On `paintcontrol` the energy is unchanged (already within 0.002 of the optimum). On `flex` and `gzip` there is no benefit; TREx is significantly worse on `gzip` (IGD, p_adj = 0.033) and lowers the optimum probability on both single-objective subjects.
- **RQ2.** No consistent suite-level effect; the one significant indicator (`gsdtsr` failure rate) restates RQ1 because the QUBO weight α = 0 makes it the energy with opposite sign.
- **RQ3.** TREx costs ≈ 3× the QPU time of raw and never helps. M3 costs 37 s once (+0.6 % to +4.3 % of a subject's 10-repetition QPU time) and gives the same energy gain as MEM (MEM's P(optimal) gain is larger and significant, M3's is not), which costs 2,368 s once (+40 % to +276 %). **M3 has the best improvement per billed second.**

## 5. Soundness of the methodology

- **Paired comparison.** Raw, MEM and M3 are all computed from the same stored raw counts, so differences come from the correction alone. TREx uses dedicated twirled runs of the same circuits.
- **Controlled hardware conditions.** The same trained circuits run for every method; each circuit's physical qubits are recorded and its calibration window is derived from a calibration measured on the same slots under the same (packed, all-slots-active) conditions. Backend properties are snapshotted per session and readout drift was measured with a second M3 calibration.
- **Statistics.** Shapiro–Wilk gating, then ANOVA with Tukey or Kruskal–Wallis with Dunn (Bonferroni-corrected); effect sizes (Cohen's d or Vargha–Delaney Â12) reported. Billed cost is deterministic and is described, not tested.
- **Budget control.** Every batch is checked against the account's billed QPU time and refused if it would cross the limit; failed jobs are retried; an interrupted run resumes without repeating finished work (this is how the study was extended from 5 to 10 repetitions).
- **Deterministic and tested.** The clustering is deterministic; the optimum of each subproblem is found by brute force (≤ 128 states); a noiseless reference (exact statevector) is provided; 235 unit tests pass.
- **Raw data preserved.** Counts, per-batch billed seconds, job ids, transpiled-circuit statistics and the qubit mapping are stored for every execution, so any further analysis needs no new hardware time.

## 6. Scope and limitations (stated explicitly)

- The 25-hour allowance and the deadline limit the study to a feasibility-driven set: QAOA-TCS on four subjects. IGDec-QAOA, the second algorithm of the original study, is implemented but was not run, so no algorithm comparison is made.
- Circuits are packed side by side (possible readout crosstalk) and not only on qubits 0–6; a circuit's execution time is its qubit share of the job's billed time.
- TREx is measurement twirling only (no rescaling) with 3 passes, and receives 3× the shots of raw (a confound); conclusions apply to that variant.
- Ten repetitions give moderate statistical power; there is no correction for multiple comparisons across metrics, and the raw/MEM/M3 pairing is not exploited by the tests.
- Calibration uses 200 shots per state, so assignment probabilities have 0.5 % resolution, comparable to the errors being corrected. Drift was measured only up to 4 Nov; repetitions 6–10 ran 4–6 Nov with the same calibration.
- One machine, one calibration, one set of trained circuits (QAOA-TCS training is not seeded). Hardware P(optimal) of raw and MEM is at or above the noiseless value on both single-objective subjects, which is unexplained; the TREx degradation on `gzip` was not investigated; the multi-objective QUBOs are effectively linear because of an inherited indexing quirk (quadratic terms survive in 1 of 105 `flex` and 1 of 61 `gzip` clusters).
