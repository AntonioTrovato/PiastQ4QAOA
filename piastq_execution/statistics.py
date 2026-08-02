"""Statistical comparison utilities, following the same approach used
throughout the paper/repo family's R scripts (see SelectQAOA/stat_tests/*.R):
Shapiro-Wilk normality check first, then either ANOVA + Tukey HSD + Cohen's d
(normal data) or Kruskal-Wallis + Dunn's test (Bonferroni-adjusted) +
Vargha-Delaney A12 (non-normal data). Also includes the Hypervolume (HV) and
Inverted Generational Distance (IGD) metrics used for the multi-objective
QAOA-TCS combos' Pareto frontiers.

Nothing here touches a backend or real result data -- these are pure
statistics over numeric samples supplied by the caller (see
piastq_execution/evaluation.py for where the samples come from).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import f_oneway, kruskal, norm, rankdata, shapiro, ttest_ind


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Pooled-variance Cohen's d, matching the cohen_d() used in
    SelectQAOA/stat_tests/*.R."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s_pooled = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    if s_pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / s_pooled)


def vargha_delaney_a12(x: Sequence[float], y: Sequence[float]) -> float:
    """Vargha-Delaney A12 effect size, matching the a12() used in
    SelectQAOA/stat_tests/*.R: P(a random x > a random y) + 0.5*P(tie)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    greater = (x[:, None] > y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(x) * len(y)))


# ---------------------------------------------------------------------------
# Normality gate
# ---------------------------------------------------------------------------

def shapiro_is_normal(samples: Sequence[float], alpha: float = 0.05) -> Tuple[bool, float]:
    """Returns (is_normal, p_value). Shapiro-Wilk needs at least 3 samples;
    fewer than that is treated as non-normal (can't establish normality)."""
    if len(samples) < 3:
        return False, float("nan")
    _, p_value = shapiro(samples)
    return p_value > alpha, float(p_value)


def all_groups_normal(groups: Dict[str, Sequence[float]], alpha: float = 0.05) -> bool:
    return all(shapiro_is_normal(values, alpha)[0] for values in groups.values())


# ---------------------------------------------------------------------------
# Dunn's test (Bonferroni-adjusted), matching FSA::dunnTest's formula
# ---------------------------------------------------------------------------

@dataclass
class DunnComparison:
    group_a: str
    group_b: str
    z: float
    p_value: float
    p_adjusted: float


def dunn_test(groups: Dict[str, Sequence[float]], p_adjust: str = "bonferroni") -> List[DunnComparison]:
    """Pairwise post-hoc test following Kruskal-Wallis, with the standard
    tie-corrected z-statistic (Dunn, 1964) and Bonferroni p-value adjustment.
    """
    names = list(groups.keys())
    all_values: List[float] = []
    for name in names:
        all_values.extend(groups[name])
    all_values_arr = np.asarray(all_values, dtype=float)
    ranks = rankdata(all_values_arr)
    n_total = len(all_values_arr)

    _, tie_counts = np.unique(all_values_arr, return_counts=True)
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts))

    mean_ranks: Dict[str, float] = {}
    sizes: Dict[str, int] = {}
    idx = 0
    for name in names:
        n = len(groups[name])
        mean_ranks[name] = float(np.mean(ranks[idx: idx + n]))
        sizes[name] = n
        idx += n

    pairs = list(itertools.combinations(names, 2))
    raw: List[Tuple[str, str, float, float]] = []
    for g1, g2 in pairs:
        n1, n2 = sizes[g1], sizes[g2]
        variance_term = (n_total * (n_total + 1) / 12.0) - (tie_term / (12.0 * (n_total - 1)))
        se = math.sqrt(variance_term * (1.0 / n1 + 1.0 / n2))
        z = (mean_ranks[g1] - mean_ranks[g2]) / se if se > 0 else 0.0
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
        raw.append((g1, g2, z, p))

    if p_adjust == "bonferroni":
        m = len(raw)
        adjusted = [min(1.0, p * m) for (_, _, _, p) in raw]
    elif p_adjust == "none":
        adjusted = [p for (_, _, _, p) in raw]
    else:
        raise ValueError(f"Unsupported p_adjust method '{p_adjust}'")

    return [
        DunnComparison(group_a=g1, group_b=g2, z=z, p_value=p, p_adjusted=p_adj)
        for (g1, g2, z, p), p_adj in zip(raw, adjusted)
    ]


# ---------------------------------------------------------------------------
# Top-level dispatcher: Shapiro-Wilk -> (ANOVA+Tukey+Cohen's d) or
# (Kruskal-Wallis+Dunn+Bonferroni+A12)
# ---------------------------------------------------------------------------

@dataclass
class PairwiseComparison:
    group_a: str
    group_b: str
    p_value: float
    p_adjusted: float
    effect_size: float
    effect_size_name: str  # "cohens_d" | "a12"


@dataclass
class ComparisonResult:
    normal: bool
    omnibus_test: str  # "anova" | "kruskal_wallis"
    omnibus_statistic: float
    omnibus_p_value: float
    pairwise: List[PairwiseComparison]


def _tukey_hsd_pairwise(groups: Dict[str, Sequence[float]]) -> List[PairwiseComparison]:
    from scipy.stats import tukey_hsd

    names = list(groups.keys())
    samples = [np.asarray(groups[n], dtype=float) for n in names]
    result = tukey_hsd(*samples)

    pairwise = []
    for i, j in itertools.combinations(range(len(names)), 2):
        pairwise.append(
            PairwiseComparison(
                group_a=names[i],
                group_b=names[j],
                p_value=float(result.pvalue[i, j]),
                p_adjusted=float(result.pvalue[i, j]),  # Tukey HSD's p-value is already family-wise adjusted
                effect_size=cohens_d(groups[names[i]], groups[names[j]]),
                effect_size_name="cohens_d",
            )
        )
    return pairwise


def compare_groups(groups: Dict[str, Sequence[float]], alpha: float = 0.05) -> ComparisonResult:
    """Compares 2+ independent groups following the paper's established
    approach: Shapiro-Wilk on every group first; if all are normal, one-way
    ANOVA + Tukey HSD post-hoc + Cohen's d; otherwise Kruskal-Wallis +
    Bonferroni-adjusted Dunn's test + Vargha-Delaney A12.
    """
    if len(groups) < 2:
        raise ValueError("compare_groups needs at least 2 groups")

    normal = all_groups_normal(groups, alpha)
    samples = [np.asarray(v, dtype=float) for v in groups.values()]

    if normal:
        statistic, p_value = f_oneway(*samples)
        pairwise = _tukey_hsd_pairwise(groups)
        return ComparisonResult(
            normal=True, omnibus_test="anova",
            omnibus_statistic=float(statistic), omnibus_p_value=float(p_value),
            pairwise=pairwise,
        )

    statistic, p_value = kruskal(*samples)
    dunn_results = dunn_test(groups, p_adjust="bonferroni")
    pairwise = [
        PairwiseComparison(
            group_a=d.group_a, group_b=d.group_b,
            p_value=d.p_value, p_adjusted=d.p_adjusted,
            effect_size=vargha_delaney_a12(groups[d.group_a], groups[d.group_b]),
            effect_size_name="a12",
        )
        for d in dunn_results
    ]
    return ComparisonResult(
        normal=False, omnibus_test="kruskal_wallis",
        omnibus_statistic=float(statistic), omnibus_p_value=float(p_value),
        pairwise=pairwise,
    )


def compare_two_groups(x: Sequence[float], y: Sequence[float], alpha: float = 0.05) -> PairwiseComparison:
    """Two-group special case: Shapiro-Wilk on both; if both normal, Welch's
    t-test + Cohen's d; otherwise Mann-Whitney U + A12."""
    x_normal, _ = shapiro_is_normal(x, alpha)
    y_normal, _ = shapiro_is_normal(y, alpha)

    if x_normal and y_normal:
        _, p_value = ttest_ind(x, y, equal_var=False)
        return PairwiseComparison("x", "y", float(p_value), float(p_value), cohens_d(x, y), "cohens_d")

    from scipy.stats import mannwhitneyu

    _, p_value = mannwhitneyu(x, y, alternative="two-sided")
    return PairwiseComparison("x", "y", float(p_value), float(p_value), vargha_delaney_a12(x, y), "a12")


# ---------------------------------------------------------------------------
# Multi-objective: Pareto dominance, Hypervolume (2D), Inverted Generational
# Distance -- used for the flex/grep/gzip/sed multi-objective QAOA-TCS combos.
# ---------------------------------------------------------------------------

def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if point `a` dominates `b` under maximization on every objective."""
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))


def pareto_front_indices(points: Sequence[Sequence[float]]) -> List[int]:
    """Indices of the non-dominated points in `points` (maximize every objective)."""
    non_dominated = []
    for i, p in enumerate(points):
        if not any(dominates(points[j], p) for j in range(len(points)) if j != i):
            non_dominated.append(i)
    return non_dominated


def hypervolume_2d(front: Sequence[Tuple[float, float]], reference_point: Tuple[float, float]) -> float:
    """Hypervolume dominated by a 2D Pareto front (maximization), relative to
    a reference point that every front point dominates. Standard rectangle-
    sweep algorithm: sort by the first objective ascending, sweep accumulating
    rectangle areas against the reference point.
    """
    if not front:
        return 0.0
    rx, ry = reference_point
    pts = sorted(front, key=lambda p: p[0])
    volume = 0.0
    prev_x = rx
    for x, y in pts:
        if x <= rx or y <= ry:
            continue
        volume += (x - prev_x) * (y - ry)
        prev_x = x
    return volume


def inverted_generational_distance(front: Sequence[Sequence[float]], reference_front: Sequence[Sequence[float]]) -> float:
    """IGD: average, over every point in the reference front, of the Euclidean
    distance to the nearest point in `front`. Lower is better (front covers
    the reference front well)."""
    if not reference_front:
        return 0.0
    if not front:
        return float("inf")
    front_arr = np.asarray(front, dtype=float)
    total = 0.0
    for ref_point in reference_front:
        ref_arr = np.asarray(ref_point, dtype=float)
        distances = np.linalg.norm(front_arr - ref_arr, axis=1)
        total += distances.min()
    return float(total / len(reference_front))


def build_reference_front(runs: Dict[str, Sequence[Sequence[float]]]) -> List[Sequence[float]]:
    """A posteriori reference frontier: the non-dominated union of every
    compared run/method's points for a dataset, matching the paper's approach."""
    all_points: List[Sequence[float]] = []
    for points in runs.values():
        all_points.extend(points)
    if not all_points:
        return []
    indices = pareto_front_indices(all_points)
    return [all_points[i] for i in indices]
