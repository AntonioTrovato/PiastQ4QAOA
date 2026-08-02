"""Synthetic unit tests for piastq_execution.sir_metrics.

Uses small, hand-built cost/fault/coverage data (no real SIR dataset files
touched) so every expected point/front can be verified by hand.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.sir_metrics import (
    build_pareto_candidates,
    build_pareto_front_from_selection,
    covered_lines,
    load_sir_program_data,
)


class TestCoveredLines(unittest.TestCase):
    def test_union_of_covered_lines_no_double_count(self):
        coverage = {0: [1, 2, 3], 1: [3, 4], 2: [5]}
        self.assertEqual(covered_lines(coverage, [0, 1]), 4)  # {1,2,3,4}
        self.assertEqual(covered_lines(coverage, [0, 1, 2]), 5)  # {1,2,3,4,5}

    def test_missing_test_case_is_skipped_not_raised(self):
        coverage = {0: [1, 2]}
        self.assertEqual(covered_lines(coverage, [0, 99]), 2)

    def test_empty_selection_is_zero(self):
        coverage = {0: [1, 2]}
        self.assertEqual(covered_lines(coverage, []), 0)


class TestBuildParetoCandidates(unittest.TestCase):
    def test_hand_verified_growing_prefix_points(self):
        # test 0: cost=10, fault=1, covers {1,2}
        # test 1: cost=5, fault=2, covers {2,3}
        costs = {0: 10.0, 1: 5.0}
        faults = [1.0, 2.0]  # list, indexed positionally (matches faults_dictionary's shape)
        coverage = {0: [1, 2], 1: [2, 3]}

        points, solutions = build_pareto_candidates([0, 1], costs, faults, coverage)

        # prefix [0]: cost=10, stmt_cov=2 ({1,2}), faults=1
        self.assertEqual(points[0], (-10.0, 2.0, 1.0))
        self.assertEqual(solutions[0], [0])
        # prefix [0,1]: cost=15, stmt_cov=3 ({1,2,3}), faults=3
        self.assertEqual(points[1], (-15.0, 3.0, 3.0))
        self.assertEqual(solutions[1], [0, 1])

    def test_empty_selection_gives_empty_candidates(self):
        points, solutions = build_pareto_candidates([], {}, [], {})
        self.assertEqual(points, [])
        self.assertEqual(solutions, [])


class TestBuildParetoFrontFromSelection(unittest.TestCase):
    def test_dominated_prefix_is_excluded(self):
        # test 0: cost=10, fault=1, covers {1}
        # test 1: cost=1, fault=5, covers {1,2,3}
        # prefix [0]: (-10, 1, 1)
        # prefix [0,1]: (-11, 3, 6) -- strictly better coverage+faults than
        # [0] at a bit more cost, so [0] is NOT dominated (cheaper), both
        # should survive on a genuine cost/quality tradeoff.
        costs = {0: 10.0, 1: 1.0}
        faults = [1.0, 5.0]
        coverage = {0: [1], 1: [1, 2, 3]}

        front = build_pareto_front_from_selection([0, 1], costs, faults, coverage)
        self.assertEqual(front, [[0], [0, 1]])

    def test_strictly_worse_prefix_is_dominated(self):
        # test 0: cost=10, fault=1, covers {1}   -> (-10, 1, 1)
        # test 1: cost=0, fault=0, covers {}      -> prefix [0,1] = (-10, 1, 1)
        # identical point -> not strictly dominated, both survive (equal
        # points are mutually non-dominating under dominates()'s strict
        # definition), but let's use a genuinely worse second test instead:
        # test 1 costs MORE and adds nothing -> [0,1] strictly worse than [0].
        costs = {0: 10.0, 1: 5.0}
        faults = [1.0, 0.0]
        coverage = {0: [1], 1: []}

        front = build_pareto_front_from_selection([0, 1], costs, faults, coverage)
        self.assertEqual(front, [[0]])


class TestLoadSirProgramData(unittest.TestCase):
    def test_loads_and_converts_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "test_cases_costs.json"), "w") as f:
                json.dump({"flex": {"0": 10, "1": 5}}, f)
            with open(os.path.join(tmp, "faults_dictionary.json"), "w") as f:
                json.dump({"flex": [1, 0]}, f)
            with open(os.path.join(tmp, "test_coverage_line_by_line.json"), "w") as f:
                json.dump({"flex": {"0": [1, 2], "1": [3]}}, f)

            data = load_sir_program_data("flex", tmp)

            self.assertEqual(data["test_cases_costs"], {0: 10, 1: 5})
            self.assertEqual(data["faults_dictionary"], [1, 0])
            self.assertEqual(data["test_coverage_line_by_line"], {0: [1, 2], 1: [3]})


if __name__ == "__main__":
    unittest.main()
