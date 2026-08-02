"""SIR (Software-artifact Infrastructure Repository) program static test
data and Pareto-front construction, shared by qaoa_tcs/multi_obj.py (at
circuit-training/execution time) and piastq_execution.evaluate_all (at
evaluation time) so both read the exact same cost/fault/coverage data and
build fronts identically, without either reloading datasets differently or
importing the other as a module.

datasets/sir_programs/{test_cases_costs,faults_dictionary,
test_coverage_line_by_line}.json are static, hand-curated per-program
per-test-case data -- not derived from any hardware execution or random
process -- so loading them directly here is exactly equivalent to what
multi_obj.py's own top-level code does.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

from piastq_execution.statistics import pareto_front_indices


def _json_keys_to_int(d: Any) -> Any:
    """Recursively converts digit-string dict keys to int. Matches
    qaoa_tcs/multi_obj.py's json_keys_to_int() exactly: top-level program
    name keys (e.g. "flex") are left as strings, nested test-id keys become
    int."""
    if isinstance(d, dict):
        return {int(k) if k.isdigit() else k: _json_keys_to_int(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_json_keys_to_int(i) for i in d]
    return d


def load_sir_program_data(sir_program: str, datasets_dir: str) -> Dict[str, Any]:
    """Loads one SIR program's static cost/fault/coverage data from
    `datasets_dir` (normally `datasets/sir_programs/`). Returns
    `{"test_cases_costs": {test_id: cost}, "faults_dictionary": [fault_cov, ...],
    "test_coverage_line_by_line": {test_id: [line, ...]}}`.
    """
    def _load(filename: str) -> Any:
        with open(os.path.join(datasets_dir, filename)) as f:
            return json.load(f)[sir_program]

    return {
        "test_cases_costs": _json_keys_to_int(_load("test_cases_costs.json")),
        "faults_dictionary": _load("faults_dictionary.json"),
        "test_coverage_line_by_line": _json_keys_to_int(_load("test_coverage_line_by_line.json")),
    }


def covered_lines(test_coverage_line_by_line: Dict[int, Sequence[int]], test_cases_list: Sequence[int]) -> int:
    """Number of distinct covered lines (no redundancy) across
    `test_cases_list`. Matches qaoa_tcs/multi_obj.py's covered_lines()."""
    lines = set()
    for test_case in test_cases_list:
        try:
            for line in test_coverage_line_by_line[test_case]:
                lines.add(line)
        except KeyError:
            continue
    return len(lines)


def build_pareto_candidates(
    selected_tests: Sequence[int],
    test_cases_costs: Dict[int, float],
    faults_dictionary: Sequence[float],
    test_coverage_line_by_line: Dict[int, Sequence[int]],
) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    """Returns `(points, solutions)`: for each growing prefix of
    `selected_tests` (the paper's "Additional-Greedy" method -- only
    prefixes of the given order are considered, not every subset),
    `points[i]` is the `(-cost, statement_coverage, fault_coverage)`
    3-tuple and `solutions[i]` is that prefix's test-id list. Matches
    qaoa_tcs/multi_obj.py's build_pareto_front() candidate generation
    exactly (that function then non-domination-filters both in lockstep via
    pareto_front_indices() -- see build_pareto_front_from_selection() here
    for the same, or pass `points` straight into
    piastq_execution.evaluation.evaluate_multi_objective_combo(), which
    filters internally).
    """
    solutions: List[List[int]] = []
    points: List[Tuple[float, float, float]] = []
    cost = 0.0
    faults = 0.0
    for test in selected_tests:
        solutions.append(list(selected_tests[: len(solutions) + 1]))
        cost += test_cases_costs[test]
        faults += faults_dictionary[test]
        stmt_coverage = covered_lines(test_coverage_line_by_line, solutions[-1])
        points.append((-cost, float(stmt_coverage), faults))
    return points, solutions


def build_pareto_front_from_selection(
    selected_tests: Sequence[int],
    test_cases_costs: Dict[int, float],
    faults_dictionary: Sequence[float],
    test_coverage_line_by_line: Dict[int, Sequence[int]],
) -> List[List[int]]:
    """Matches qaoa_tcs/multi_obj.py's build_pareto_front() exactly, but
    takes the SIR program's static data as arguments instead of reading it
    off module-level globals -- usable standalone at evaluation time without
    importing multi_obj.py."""
    points, solutions = build_pareto_candidates(
        selected_tests, test_cases_costs, faults_dictionary, test_coverage_line_by_line
    )
    non_dominated_indices = pareto_front_indices(points)
    return [solutions[i] for i in non_dominated_indices]
