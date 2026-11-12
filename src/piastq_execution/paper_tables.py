"""Tables for the Q-SE 2027 paper (multi-objective subjects: flex, gzip).

Six configurations are compared: the noiseless simulator with 200 and 600 shots
(S200, S600) and the four techniques on PIAST-Q (raw, MEM, M3, TREx). The
exact (statevector) simulator is NOT used here.

For every program and repetition the script re-evaluates the six
configurations with ``ideal_simulation.evaluate_combo_against_ideal`` (same
code as the rest of the pipeline), so that HV/IGD/ND are computed against the
non-dominated union of the six fronts of that repetition, and writes Markdown
tables to ``results/evaluation/paper/``:

  * ``per_repetition.json``   -- the re-evaluated per-repetition metrics;
  * ``means.md``              -- mean (std) of Energy, HV, IGD, ND, Best;
  * ``shot_matched_tests.md`` -- raw/MEM/M3 vs S200 and TREx vs S600, plus
                                 Best (raw vs S200): Shapiro-Wilk per group,
                                 Welch t + Cohen's d or Mann-Whitney U + A12;
  * ``technique_tests.md``    -- raw vs MEM vs M3 vs TREx: ANOVA + Tukey HSD +
                                 Cohen's d, or Kruskal-Wallis + Dunn
                                 (Bonferroni) + A12.

"Best": for every repetition and cluster, the configuration(s) between raw and
S200 whose selected bitstring has the lower QUBO energy get +1 (a tie gives +1
to both); each of the two configurations thus yields 10 counts per program.

Effect-size magnitudes: |d| < 0.2 negligible (N), < 0.5 small (S), < 0.8
medium (M), otherwise large (L); |A12 - 0.5| < 0.06 N, < 0.14 S, < 0.21 M,
otherwise L (Vargha and Delaney).

    cd src/piastq_execution
    python paper_tables.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import warnings
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, shapiro

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from piastq_execution.evaluate_all import load_combo_definitions, repo_root  # noqa: E402
from piastq_execution.evaluation import qubo_energy  # noqa: E402
from piastq_execution.ideal_simulation import (  # noqa: E402
    evaluate_combo_against_ideal,
    load_cluster_assignments,
)
from piastq_execution.mitigation import CalibrationStore  # noqa: E402
from piastq_execution.qubo_io import json_dict_to_cluster_qubo  # noqa: E402
from piastq_execution.statistics import (  # noqa: E402
    compare_groups,
    compare_two_groups,
    dunn_test,
    vargha_delaney_a12,
)

PROGRAMS = ["flex", "gzip"]
SIMULATED = ["ideal_shots200", "ideal_shots600"]
HARDWARE = ["raw", "mem", "m3", "trex"]
CONFIGS = ["ideal_shots200", "ideal_shots600"] + HARDWARE
LABEL = {"ideal_shots200": "S200", "ideal_shots600": "S600",
         "raw": "Raw", "mem": "MEM", "m3": "M3", "trex": "TREx"}
METRICS = [("qubo_energy", "Energy"), ("hypervolume", "HV"), ("igd", "IGD"), ("num_non_dominated", "ND")]
SHOT_MATCHED = [("raw", "ideal_shots200"), ("mem", "ideal_shots200"), ("m3", "ideal_shots200"),
                ("trex", "ideal_shots600")]
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shapiro_p(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    if np.ptp(values) == 0:
        return float("nan")  # constant sample: not testable, treated as non-normal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(shapiro(values)[1])


def is_normal(values: Sequence[float]) -> bool:
    p = shapiro_p(values)
    return p == p and p > ALPHA


def magnitude(name: str, value: float) -> str:
    value = round(value, 2)  # classify the value as printed, so tables never disagree with their labels
    if name == "cohens_d":
        v = abs(value)
        return "N" if v < 0.2 else "S" if v < 0.5 else "M" if v < 0.8 else "L"
    v = round(abs(value - 0.5), 2)
    return "N" if v < 0.06 else "S" if v < 0.14 else "M" if v < 0.21 else "L"


def sw_cell(p: float) -> str:
    if p != p:
        return "const."
    return f"{p:.3f} ({'normal' if p > ALPHA else 'non-normal'})"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_eff(name: str, value: float) -> str:
    return f"{'d' if name == 'cohens_d' else 'A12'}={value:.2f} ({magnitude(name, value)})"


def two_groups(x, y):
    """Welch t + d if both groups are normal, else Mann-Whitney U + A12."""
    if is_normal(x) and is_normal(y):
        r = compare_two_groups(x, y, ALPHA)
        return "Welch t", r.p_value, r.effect_size_name, r.effect_size
    return "Mann-Whitney U", float(mannwhitneyu(x, y, alternative="two-sided")[1]), "a12", vargha_delaney_a12(x, y)


def many_groups(groups: Dict[str, List[float]]):
    if all(is_normal(v) for v in groups.values()):
        res = compare_groups(groups, ALPHA)
        pairs = [(p.group_a, p.group_b, p.p_adjusted, p.effect_size_name, p.effect_size) for p in res.pairwise]
        return "ANOVA", res.omnibus_p_value, "Tukey HSD", pairs
    p = float(kruskal(*groups.values())[1])
    pairs = [(d.group_a, d.group_b, d.p_adjusted, "a12",
              vargha_delaney_a12(groups[d.group_a], groups[d.group_b])) for d in dunn_test(groups)]
    return "Kruskal-Wallis", p, "Dunn (Bonferroni)", pairs


# ---------------------------------------------------------------------------
# Re-evaluation without the exact simulator
# ---------------------------------------------------------------------------

def _load_simulated(root, circuits_dir, n_clusters):
    simulated = {}
    for label in SIMULATED:
        path = os.path.join(root, "results", "ideal_simulation", circuits_dir,
                            f"{circuits_dir}-rep-1-{label}-counts.jsonl")
        per_it: Dict[int, Dict[int, Dict[str, float]]] = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                per_it.setdefault(r["iteration_id"], {})[r["cluster_idx"]] = r["counts"]
        simulated[label] = {it: [d[i] for i in range(n_clusters)] for it, d in per_it.items()}
    return simulated


def evaluate_program(root: str, program: str) -> Dict[str, List[Dict]]:
    combo_name = f"{program}_qaoa_tcs"
    cfg = load_combo_definitions(os.path.join(root, "configs", "execution_plan.yaml"))[combo_name]
    reps_dir = os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", cfg["circuits_dir"], "rep_1")
    qubos_file = [f for f in os.listdir(reps_dir) if f.endswith("_qubos.json")][0]
    with open(os.path.join(reps_dir, qubos_file)) as f:
        qubos_json = json.load(f)
    qubos = [json_dict_to_cluster_qubo(q) for q in qubos_json]
    assignments = load_cluster_assignments(root, cfg, qubos_json)
    simulated = _load_simulated(root, cfg["circuits_dir"], len(qubos))
    store = CalibrationStore(base_dir=os.path.join(root, "calibration"))

    # Capture the selected bitstrings to compute "Best" (raw vs S200).
    import piastq_execution.ideal_simulation as ideal_simulation
    captured: List[List[str]] = []
    original = ideal_simulation._argmax_optimal_fraction

    def capture(selected, qubos_arg):
        captured.append(list(selected))
        return original(selected, qubos_arg)

    ideal_simulation._argmax_optimal_fraction = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results = evaluate_combo_against_ideal(root, combo_name, cfg, simulated, assignments, qubos_json, store)
    finally:
        ideal_simulation._argmax_optimal_fraction = original

    # evaluate_combo_against_ideal visits, per iteration, the hardware methods
    # then the simulated labels, in this order.
    order = [m for m in cfg["methods"]] + list(simulated)
    iterations = [r["iteration_id"] for r in results["raw"]]
    assert len(captured) == len(order) * len(iterations)
    selected = {(it, src): captured[i * len(order) + j]
                for i, it in enumerate(iterations) for j, src in enumerate(order)}
    for i, it in enumerate(iterations):
        energies = {s: [qubo_energy([int(b) for b in bits[::-1]], q[0], q[1])
                        for bits, q in zip(selected[(it, s)], qubos)] for s in ("raw", "ideal_shots200")}
        best = {s: 0 for s in energies}
        for k in range(len(qubos)):
            low = min(energies[s][k] for s in energies)
            for s in energies:
                best[s] += energies[s][k] <= low + 1e-9
        for s in energies:
            results[s][i]["best"] = best[s]
        # Sanity check: the captured bitstrings reproduce the stored energy.
        assert abs(sum(energies["raw"]) - results["raw"][i]["qubo_energy"]) < 1e-6
    return results


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def main() -> None:
    root = repo_root()
    out_dir = os.path.join(root, "results", "evaluation", "paper")
    os.makedirs(out_dir, exist_ok=True)
    data = {p: evaluate_program(root, p) for p in PROGRAMS}
    with open(os.path.join(out_dir, "per_repetition.json"), "w") as f:
        json.dump(data, f, indent=1)

    note = ("Configurations: S200, S600 (noiseless simulator) and Raw, MEM, M3, TREx (PIAST-Q), 10 repetitions each. "
            "HV, IGD and ND are computed, per repetition, against the non-dominated union of the six fronts. "
            "Best: number of clusters in which the configuration selects the lower-energy bitstring "
            "between Raw and S200 (ties count for both). Effect sizes: |d| <0.2 N, <0.5 S, <0.8 M, else L; "
            "|A12-0.5| <0.06 N, <0.14 S, <0.21 M, else L.\n")

    lines = ["# Mean (std) over the 10 repetitions\n", note]
    for prog in PROGRAMS:
        d = data[prog]
        lines += [f"\n## {prog.capitalize()}\n", "| Metric | " + " | ".join(LABEL[c] for c in CONFIGS) + " |",
                  "|---|" + "---|" * len(CONFIGS)]
        for key, name in METRICS + [("best", "Best")]:
            cells = []
            for c in CONFIGS:
                if key not in d[c][0]:
                    cells.append("-")
                    continue
                v = np.array([r[key] for r in d[c]], dtype=float)
                cells.append(f"{v.mean():.6g} ({v.std(ddof=1):.3g})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    open(os.path.join(out_dir, "means.md"), "w").write("\n".join(lines) + "\n")

    lines = ["# PIAST-Q technique vs. noiseless reference with the same shots\n", note,
             "Shapiro-Wilk (alpha=0.05) on each group; both normal: Welch's t-test + Cohen's d; "
             "otherwise Mann-Whitney U + Vargha-Delaney A12 (two-sided).\n"]
    for prog in PROGRAMS:
        d = data[prog]
        lines += [f"\n## {prog.capitalize()}\n",
                  "| Comparison | Metric | SW technique | SW reference | Test | p | Effect size |",
                  "|---|---|---|---|---|---|---|"]
        for a, b in SHOT_MATCHED:
            metrics = METRICS + ([("best", "Best")] if a == "raw" else [])
            for key, name in metrics:
                x = [r[key] for r in d[a]]
                y = [r[key] for r in d[b]]
                test, p, en, ev = two_groups(x, y)
                lines.append(f"| {LABEL[a]} vs {LABEL[b]} | {name} | {sw_cell(shapiro_p(x))} | "
                             f"{sw_cell(shapiro_p(y))} | {test} | {fmt_p(p)} | {fmt_eff(en, ev)} |")
    open(os.path.join(out_dir, "shot_matched_tests.md"), "w").write("\n".join(lines) + "\n")

    lines = ["# Raw vs MEM vs M3 vs TREx on PIAST-Q\n", note,
             "Shapiro-Wilk (alpha=0.05) on each group; all normal: one-way ANOVA + Tukey HSD + Cohen's d; "
             "otherwise Kruskal-Wallis + Dunn's test (Bonferroni) + Vargha-Delaney A12.\n"]
    for prog in PROGRAMS:
        d = data[prog]
        lines.append(f"\n## {prog.capitalize()}\n")
        for key, name in METRICS:
            groups = {m: [r[key] for r in d[m]] for m in HARDWARE}
            sws = ", ".join(f"{LABEL[m]} {sw_cell(shapiro_p(v))}" for m, v in groups.items())
            omnibus, p, posthoc, pairs = many_groups(groups)
            lines += [f"\n### {name}\n", f"Shapiro-Wilk p: {sws}.  ",
                      f"{omnibus}: p = {fmt_p(p)}; post-hoc: {posthoc}.\n",
                      "| Pair | p (adjusted) | Effect size |", "|---|---|---|"]
            for ga, gb, padj, en, ev in pairs:
                lines.append(f"| {LABEL[ga]} vs {LABEL[gb]} | {fmt_p(padj)} | {fmt_eff(en, ev)} |")
    open(os.path.join(out_dir, "technique_tests.md"), "w").write("\n".join(lines) + "\n")

    stale = os.path.join(out_dir, "merge_wins.md")
    if os.path.exists(stale):
        os.remove(stale)  # superseded: Best is now part of means.md / shot_matched_tests.md
    print(f"Saved Markdown tables to {out_dir}")


if __name__ == "__main__":
    main()
