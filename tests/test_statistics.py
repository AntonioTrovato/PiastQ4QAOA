"""Synthetic unit tests for piastq_execution.statistics.

All samples here are small, hand-built arrays with known properties (exact
Cohen's d/A12 by construction, clearly normal vs clearly skewed data, hand-
solvable Pareto fronts) -- no real result data exists yet for this study.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from piastq_execution.statistics import (
    build_reference_front,
    cohens_d,
    compare_groups,
    compare_two_groups,
    dominates,
    dunn_test,
    hypervolume_2d,
    inverted_generational_distance,
    pareto_front_indices,
    shapiro_is_normal,
    vargha_delaney_a12,
)


class TestEffectSizes(unittest.TestCase):
    def test_cohens_d_zero_for_identical_distributions(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(cohens_d(x, y), 0.0)

    def test_cohens_d_sign_matches_direction(self):
        x = [10, 11, 12]
        y = [1, 2, 3]
        self.assertGreater(cohens_d(x, y), 0)
        self.assertLess(cohens_d(y, x), 0)

    def test_a12_all_x_greater_is_one(self):
        x = [10, 11, 12]
        y = [1, 2, 3]
        self.assertAlmostEqual(vargha_delaney_a12(x, y), 1.0)
        self.assertAlmostEqual(vargha_delaney_a12(y, x), 0.0)

    def test_a12_identical_distributions_is_half(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        self.assertAlmostEqual(vargha_delaney_a12(x, y), 0.5)

    def test_a12_hand_computed_partial_overlap(self):
        # x=[1,3], y=[2,2]: pairs (1,2)->0, (1,2)->0, (3,2)->1, (3,2)->1
        # A12 = (2 + 0*ties)/(2*2) = 0.5
        x = [1, 3]
        y = [2, 2]
        self.assertAlmostEqual(vargha_delaney_a12(x, y), 0.5)


class TestShapiro(unittest.TestCase):
    def test_too_few_samples_is_not_normal(self):
        is_normal, p = shapiro_is_normal([1, 2])
        self.assertFalse(is_normal)
        self.assertTrue(math.isnan(p))

    def test_clearly_nonnormal_sample_rejected(self):
        # extreme outlier-laden sample: very unlikely to pass Shapiro-Wilk
        skewed = [1, 1, 1, 1, 1, 1, 1, 1, 100]
        is_normal, p = shapiro_is_normal(skewed)
        self.assertFalse(is_normal)

    def test_symmetric_evenly_spaced_sample_is_normal_enough(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(loc=0, scale=1, size=200)
        is_normal, p = shapiro_is_normal(sample)
        self.assertTrue(is_normal)


class TestDunnTest(unittest.TestCase):
    def test_hand_computed_z_for_three_separated_groups_no_ties(self):
        groups = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        results = dunn_test(groups, p_adjust="none")
        by_pair = {(r.group_a, r.group_b): r for r in results}
        r = by_pair[("A", "B")]
        # ranks: A={1,2,3}, B={4,5,6}, C={7,8,9}; mean ranks 2,5,8; N=9, no ties
        # SE = sqrt((9*10/12)*(1/3+1/3)) = sqrt(7.5*0.6667) = sqrt(5.0)
        expected_se = math.sqrt((9 * 10 / 12.0) * (1.0 / 3 + 1.0 / 3))
        expected_z = (2.0 - 5.0) / expected_se
        self.assertAlmostEqual(r.z, expected_z, places=6)

    def test_bonferroni_inflates_p_by_number_of_comparisons(self):
        groups = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        raw = dunn_test(groups, p_adjust="none")
        adjusted = dunn_test(groups, p_adjust="bonferroni")
        m = len(raw)  # 3 pairs for 3 groups
        self.assertEqual(m, 3)
        for r_raw, r_adj in zip(raw, adjusted):
            self.assertAlmostEqual(r_adj.p_adjusted, min(1.0, r_raw.p_value * m), places=6)


class TestCompareGroups(unittest.TestCase):
    def test_normal_data_routes_to_anova(self):
        rng = np.random.default_rng(1)
        groups = {
            "A": rng.normal(0, 1, 200),
            "B": rng.normal(0.1, 1, 200),
            "C": rng.normal(-0.1, 1, 200),
        }
        result = compare_groups(groups)
        self.assertTrue(result.normal)
        self.assertEqual(result.omnibus_test, "anova")
        self.assertEqual(len(result.pairwise), 3)
        for p in result.pairwise:
            self.assertEqual(p.effect_size_name, "cohens_d")

    def test_nonnormal_data_routes_to_kruskal_wallis(self):
        groups = {
            "A": [1, 1, 1, 1, 1, 1, 1, 1, 50],
            "B": [2, 2, 2, 2, 2, 2, 2, 2, 60],
            "C": [3, 3, 3, 3, 3, 3, 3, 3, 70],
        }
        result = compare_groups(groups)
        self.assertFalse(result.normal)
        self.assertEqual(result.omnibus_test, "kruskal_wallis")
        for p in result.pairwise:
            self.assertEqual(p.effect_size_name, "a12")

    def test_requires_at_least_two_groups(self):
        with self.assertRaises(ValueError):
            compare_groups({"A": [1, 2, 3]})


class TestCompareTwoGroups(unittest.TestCase):
    def test_nonnormal_pair_uses_mannwhitney_and_a12(self):
        x = [1, 1, 1, 1, 1, 1, 1, 1, 50]
        y = [2, 2, 2, 2, 2, 2, 2, 2, 60]
        result = compare_two_groups(x, y)
        self.assertEqual(result.effect_size_name, "a12")


class TestParetoAndHV(unittest.TestCase):
    def test_dominates_basic(self):
        self.assertTrue(dominates((2, 2), (1, 1)))
        self.assertFalse(dominates((1, 1), (2, 2)))
        self.assertFalse(dominates((1, 2), (2, 1)))  # neither dominates
        self.assertFalse(dominates((1, 1), (1, 1)))  # equal doesn't dominate

    def test_pareto_front_indices_hand_verified(self):
        points = [(1, 5), (5, 1), (3, 3), (2, 2), (4, 4)]
        # (2,2) is dominated by (3,3) and (4,4); (4,4) is dominated by nothing among these
        front = pareto_front_indices(points)
        front_points = {points[i] for i in front}
        self.assertEqual(front_points, {(1, 5), (5, 1), (4, 4)})

    def test_hypervolume_2d_hand_computed_single_point(self):
        # single point (3,3), reference (0,0) -> area = 3*3 = 9
        hv = hypervolume_2d([(3, 3)], reference_point=(0, 0))
        self.assertAlmostEqual(hv, 9.0)

    def test_hypervolume_2d_hand_computed_two_points(self):
        # front (1,4),(3,2), reference (0,0), sorted by x: (1,4) then (3,2)
        # rect1: (1-0)*(4-0)=4; rect2: (3-1)*(2-0)=4; total=8
        hv = hypervolume_2d([(1, 4), (3, 2)], reference_point=(0, 0))
        self.assertAlmostEqual(hv, 8.0)

    def test_hypervolume_2d_empty_front_is_zero(self):
        self.assertEqual(hypervolume_2d([], reference_point=(0, 0)), 0.0)

    def test_igd_zero_when_front_equals_reference(self):
        front = [(1, 1), (2, 2)]
        self.assertAlmostEqual(inverted_generational_distance(front, front), 0.0)

    def test_igd_hand_computed(self):
        # reference has one point (0,0); front has (3,4) -> distance 5
        igd = inverted_generational_distance([(3, 4)], [(0, 0)])
        self.assertAlmostEqual(igd, 5.0)

    def test_build_reference_front_is_union_pareto(self):
        runs = {
            "raw": [(1, 5), (2, 2)],
            "mem": [(5, 1), (3, 3)],
        }
        ref = build_reference_front(runs)
        self.assertEqual(set(ref), {(1, 5), (5, 1), (3, 3)})


if __name__ == "__main__":
    unittest.main()
