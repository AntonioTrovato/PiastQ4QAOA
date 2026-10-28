# PIAST-Q4QAOA: Project Summary

## 1. Context

Regression test case selection (TCS) chooses a subset of a test suite that is cheap to run yet still effective at catching faults. It can be written as a QUBO (a binary quadratic optimisation problem) and approximately solved with the Quantum Approximate Optimization Algorithm (QAOA). QAOA-TCS makes large suites fit on small quantum circuits (at most 7 qubits each) by clustering the suite once, solving each cluster independently and merging the results. It has a single-objective variant (cost and failure rate or diversity) and a multi-objective variant (cost, statement coverage, fault coverage), the latter producing a Pareto front of trade-offs.

QAOA-TCS has so far been evaluated mainly on simulators. On real hardware the circuit output is distorted by noise. For shallow circuits such as these, a dominant and well-understood component is **readout error**: a qubit measured as 0 when it was in state 1, or vice versa. Mitigation techniques try to undo this distortion. Three are compared:

- **MEM (full measurement-error mitigation)**: every possible n-qubit basis state is prepared and measured (2^n calibration circuits) to build the full confusion matrix. Its inverse is then applied to any measured distribution. It is exact in the readout model, but its calibration cost grows exponentially.
- **M3**: the same idea with per-qubit assignment probabilities (2n calibration circuits) and a correction restricted to the observed outcomes. It is far cheaper and scales to wider circuits.
- **TREx (measurement twirling)**: before measurement, a random subset of qubits is flipped with X gates and the flip is undone classically. Averaging over several random patterns turns systematic readout bias into random noise that averages out. It needs no calibration step, but every twirl pattern is an additional hardware run.

The target machine is PIAST-Q, a 20-qubit trapped-ion quantum computer (accessed through the AQT provider), limited to 200 shots per run, with 25 hours of QPU time available.

## 2. Goal and research questions

**Goal:** to evaluate QAOA-based test case selection on real quantum hardware and to quantify how much readout-error mitigation improves it, and at what cost.

| RQ | Question | Why it matters |
|---|---|---|
| **RQ1** | Does mitigation (MEM, M3, TREx) improve solution quality (QUBO energy, probability of the optimal bitstring) over raw hardware output? | Establishes whether mitigation is needed at all for these circuits. |
| **RQ2** | Does better quality translate into a better selected test suite (suite cost and effectiveness, or Pareto hypervolume / IGD in the multi-objective case)? | Links quantum-level improvements to the practical test-selection outcome. |
| **RQ3a** | Which technique improves quality the most, ignoring cost? | If MEM and M3 (which target readout error directly) match TREx (a broader technique), readout error is the dominant noise source for these circuits. |
| **RQ3b** | Which technique gives the best improvement per second of measured QPU time? | QPU time is the scarce resource. A technique can win RQ3a and still lose here. |

These are answered on two multi-objective programs (`gzip`, `flex`) and two single-objective datasets (`paintcontrol`, `gsdtsr`), so that conclusions are not tied to one problem type.

## 3. Novelty

- Moves QAOA-based TCS from simulation to a real trapped-ion device.
- Compares three complementary readout-mitigation techniques (full calibration, scalable calibration, calibration-free twirling) on the same circuits and the same physical qubits, for both single- and multi-objective problems.
- Reports mitigation benefit together with measured QPU cost (RQ3b), which matters for budget-limited quantum experiments.
- Provides a replication package that stores the full raw measurement data (not only the best bitstring), so any further analysis can be performed without new hardware time.

## 4. Soundness of the methodology

- **Paired comparison.** Raw, MEM and M3 are all computed from the same stored raw counts, so differences come from the correction alone. TREx uses dedicated twirled runs of the same circuits.
- **Controlled hardware conditions.** The same trained circuits run for every method. A pinned physical-qubit layout is shared by execution and calibration circuits, so each calibration is measured on the qubits it is applied to. Backend calibration data (T1/T2) is snapshotted per session, and readout drift is measured with a second calibration at the end of the run.
- **Statistics.** Repeated runs are tested with Shapiro-Wilk normality gating, then ANOVA with Tukey or Kruskal-Wallis with Dunn (Bonferroni-corrected). Effect sizes are reported (Cohen's d or Vargha-Delaney Â12).
- **Budget control.** A cost model (200 shots per 15 s batch) plans the run, and a pilot check measures the real time per batch after calibration and aborts if the projection exceeds the QPU limit.
- **Deterministic and tested.** The clustering is deterministic. The ground-truth optimum of each subproblem is found by brute force (at most 128 states). The pipeline has 187 unit tests and was run end to end on a noiseless simulator before any hardware use.
- **Raw data preserved.** Counts, per-batch timings, job metadata, transpiled circuit statistics and the qubit mapping are stored for every execution.

## 5. Scope and limitations (stated explicitly)

- The 25-hour budget limits the study to a reduced, feasibility-driven set: QAOA-TCS on `gzip` and `flex` (multi-objective) and `paintcontrol` and `gsdtsr` (single-objective). All four methods (raw, MEM, M3, TREx) are run with 6 repetitions, 200 shots per circuit and 2 TREx twirls (about 21.7 QPU hours including calibration and a drift check).
- IGDec-QAOA, the second algorithm of the original study, is implemented in the package but excluded from the hardware run (only one dataset is affordable for it), so no algorithm comparison is made.
- Six repetitions give limited statistical power, so effect sizes are reported next to p-values.
- TREx is implemented as measurement twirling only (random pre-measurement bit flips undone classically); gate-level twirling and calibration-based rescaling are not included. With two twirls it is a weak test of that technique, and it receives more shots (twirls x 200) than raw, which is accounted for in the cost analysis of RQ3b.
