"""Persisting/loading QUBO (linear + quadratic objective) coefficients
alongside trained circuits.

Every training script (qaoa_tcs/single_obj.py, qaoa_tcs/multi_obj.py, the
three igdec_qaoa/loch_qaoa_*_extract_circuits.py scripts) already builds a
qiskit_optimization QuadraticProgram per circuit before training it -- this
module just extracts and reconstructs those same coefficients so
piastq_execution.evaluation's qubo_energy()/brute_force_optimal() can be
recomputed at evaluation time against the exact QUBO a circuit was actually
trained to solve, without reloading datasets or re-deriving anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def qubo_to_json_dict(qubo) -> Dict[str, Any]:
    """Extracts a QuadraticProgram's already-materialized linear + quadratic
    objective coefficients into a JSON-serializable dict:
    `{"linear": [...], "quadratic": [[i, j, coeff], ...], "num_qubits": n}`.

    Works for any QuadraticProgram regardless of how it was built
    (create_linear_qubo(), create_QUBO_problem(),
    TestCaseOptimization().to_quadratic_program(), ...) -- only reads
    coefficients, never re-derives them.
    """
    num_vars = qubo.get_num_binary_vars()
    linear_dict = qubo.objective.linear.to_dict()
    linear = [float(linear_dict.get(i, 0.0)) for i in range(num_vars)]
    quadratic_dict = qubo.objective.quadratic.to_dict()
    quadratic = [[int(i), int(j), float(coeff)] for (i, j), coeff in quadratic_dict.items()]
    return {"linear": linear, "quadratic": quadratic, "num_qubits": num_vars}


def json_dict_to_cluster_qubo(d: Dict[str, Any]) -> Tuple[List[float], Dict[Tuple[int, int], float], int]:
    """Inverse of qubo_to_json_dict(): reconstructs the `(linear, quadratic,
    num_qubits)` tuple `piastq_execution.evaluation`'s
    `evaluate_single_objective_combo(cluster_qubos=...)` expects."""
    linear = [float(x) for x in d["linear"]]
    quadratic = {(int(i), int(j)): float(coeff) for i, j, coeff in d["quadratic"]}
    num_qubits = int(d["num_qubits"])
    return linear, quadratic, num_qubits
