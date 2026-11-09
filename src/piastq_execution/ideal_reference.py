"""Noiseless reference values for the trained QAOA-TCS circuits.

For every cluster circuit of every planned QAOA-TCS combo this computes, from
the exact statevector of the saved circuit (no sampling, no noise):

  * the probability of the brute-forced optimal bitstring of the cluster's QUBO,
  * whether the circuit's most likely bitstring IS that optimum,
  * the QUBO energy of the most likely bitstrings summed over the clusters,
    next to the sum of the brute-forced optima.

These are the numbers a perfect machine would give with infinitely many shots,
so they are the reference for how far the hardware results (raw and mitigated)
are from the ideal one. Written to results/evaluation/ideal_reference.json.
Pure local computation: no backend, no network.

    cd src/piastq_execution
    python ideal_reference.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from piastq_execution.evaluate_all import load_combo_definitions, planned_combo_names, repo_root
from piastq_execution.evaluation import brute_force_optimal, qubo_energy, select_argmax_bitstring
from piastq_execution.qubo_io import json_dict_to_cluster_qubo


def ideal_reference_for_combo(root: str, circuits_dir: str) -> dict:
    from qiskit import qpy
    from qiskit.quantum_info import Statevector

    reps_dir = os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", circuits_dir, "rep_1")
    with open(os.path.join(reps_dir, f"{circuits_dir}_rep1_qubos.json")) as f:
        qubos = json.load(f)

    p_optimal, argmax_is_optimal = [], 0
    energy_of_argmax = energy_of_optimum = 0.0
    for index, entry in enumerate(qubos):
        linear, quadratic, num_qubits = json_dict_to_cluster_qubo(entry)
        with open(os.path.join(reps_dir, f"{circuits_dir}_rep1_cluster{index}.qpy"), "rb") as f:
            circuit = qpy.load(f)[0].remove_final_measurements(inplace=False)
        distribution = Statevector(circuit).probabilities_dict()
        optimal_bitstring, optimal_energy = brute_force_optimal(linear, quadratic, num_qubits)
        selected = select_argmax_bitstring(distribution)
        p_optimal.append(distribution.get(optimal_bitstring, 0.0))
        argmax_is_optimal += selected == optimal_bitstring
        energy_of_argmax += qubo_energy([int(b) for b in selected[::-1]], linear, quadratic)
        energy_of_optimum += optimal_energy

    return {
        "clusters": len(qubos),
        "mean_probability_of_optimal": sum(p_optimal) / len(p_optimal),
        "min_probability_of_optimal": min(p_optimal),
        "clusters_where_argmax_is_optimal": int(argmax_is_optimal),
        "qubo_energy_of_argmax_bitstrings": energy_of_argmax,
        "qubo_energy_of_brute_force_optima": energy_of_optimum,
    }


if __name__ == "__main__":
    root = repo_root()
    plan_path = os.path.join(root, "configs", "execution_plan.yaml")
    combos = load_combo_definitions(plan_path)
    planned = planned_combo_names(plan_path)
    result = {}
    for name, cfg in combos.items():
        if cfg["algorithm"] != "qaoa_tcs" or (planned is not None and name not in planned):
            continue
        result[name] = ideal_reference_for_combo(root, cfg["circuits_dir"])
        print(name, result[name])
    out = os.path.join(root, "results", "evaluation", "ideal_reference.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out}")
