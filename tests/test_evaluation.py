"""Synthetic unit tests for piastq_execution.evaluation.

Every QUBO/circuit here is a tiny 1-2 qubit hand-built toy so expected values
can be verified by hand. No real result data, no AQT/backend calls.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.evaluation import (
    ClusterAssignment,
    MitigationOverhead,
    SingleObjectiveEvaluation,
    brute_force_optimal,
    compute_execution_time_seconds,
    compute_mitigation_overhead,
    compute_single_objective_metrics,
    evaluate_multi_objective_combo,
    evaluate_single_objective_combo,
    extract_metric_samples,
    merge_selected_tests,
    probability_of_optimal,
    qubo_energy,
    select_argmax_bitstring,
)
from piastq_execution.statistics import compare_groups
from piastq_execution.raw_counts import RawCountsRecord, BatchRecord


def make_raw_record(total_shots_returned, wall_clock_seconds):
    return RawCountsRecord(
        algorithm="qaoa_tcs", objective_mode="single_objective", dataset="toy",
        circuit_id="c", cluster_id=0, iteration_id=None, subproblem_id=None,
        backend_name="b", backend_version="v", physical_qubit_mapping={0: 0},
        batches=[BatchRecord(0, total_shots_returned, total_shots_returned, {}, "t", wall_clock_seconds)],
        total_shots_requested=total_shots_returned, total_shots_returned=total_shots_returned,
        total_wall_clock_seconds=wall_clock_seconds, aggregated_counts={},
    )


class TestQuboEnergy(unittest.TestCase):
    def test_hand_computed_energy(self):
        linear = [1, -2]
        quadratic = {(0, 1): 3}
        self.assertAlmostEqual(qubo_energy([1, 1], linear, quadratic), 2.0)
        self.assertAlmostEqual(qubo_energy([1, 0], linear, quadratic), 1.0)
        self.assertAlmostEqual(qubo_energy([0, 1], linear, quadratic), -2.0)
        self.assertAlmostEqual(qubo_energy([0, 0], linear, quadratic), 0.0)


class TestBruteForceOptimal(unittest.TestCase):
    def test_hand_verified_2_qubit_minimum(self):
        linear = [1, -2]
        quadratic = {(0, 1): 3}
        bitstring, energy = brute_force_optimal(linear, quadratic, num_qubits=2)
        self.assertEqual(bitstring, "10")
        self.assertAlmostEqual(energy, -2.0)

    def test_probability_of_optimal_lookup(self):
        distribution = {"10": 0.7, "00": 0.3}
        self.assertAlmostEqual(probability_of_optimal(distribution, "10"), 0.7)
        self.assertAlmostEqual(probability_of_optimal(distribution, "11"), 0.0)


class TestSelectArgmax(unittest.TestCase):
    def test_picks_highest_probability(self):
        distribution = {"00": 0.2, "10": 0.7, "11": 0.1}
        self.assertEqual(select_argmax_bitstring(distribution), "10")

    def test_empty_distribution_raises(self):
        with self.assertRaises(ValueError):
            select_argmax_bitstring({})


class TestMergeSelectedTests(unittest.TestCase):
    def test_hand_verified_merge_across_two_clusters(self):
        assignments = [
            ClusterAssignment(cluster_id=0, cluster_test_cases=[5, 6]),
            ClusterAssignment(cluster_id=1, cluster_test_cases=[7, 8, 9]),
        ]
        # cluster0 bitstring "10" -> bits(little-endian)=[0,1] -> selects test_cases[1]=6
        # cluster1 bitstring "101" -> bits=[1,0,1] -> selects test_cases[0]=7, test_cases[2]=9
        bitstrings = ["10", "101"]
        merged = merge_selected_tests(assignments, bitstrings)
        self.assertEqual(merged, [6, 7, 9])

    def test_deduplicates_repeated_test_ids(self):
        assignments = [
            ClusterAssignment(cluster_id=0, cluster_test_cases=[1, 2]),
            ClusterAssignment(cluster_id=1, cluster_test_cases=[2, 3]),
        ]
        bitstrings = ["11", "10"]  # cluster0 selects [1,2]; cluster1 bits=[0,1]->selects test_cases[1]=3
        merged = merge_selected_tests(assignments, bitstrings)
        self.assertEqual(merged, [1, 2, 3])

    def test_length_mismatch_raises(self):
        assignments = [ClusterAssignment(cluster_id=0, cluster_test_cases=[1])]
        with self.assertRaises(ValueError):
            merge_selected_tests(assignments, ["0", "1"])


class TestSingleObjectiveMetrics(unittest.TestCase):
    def test_hand_computed_cost_and_failure_rate(self):
        test_case_data = {"cost": [1, 2, 3], "failure_rate": [0.1, 0.2, 0.3]}
        metrics = compute_single_objective_metrics("iofrol", [0, 2], test_case_data)
        self.assertAlmostEqual(metrics["execution_cost"], 4.0)
        self.assertAlmostEqual(metrics["failure_rate"], 0.4)

    def test_elevator_o3_sums_two_effectiveness_columns(self):
        test_case_data = {
            "cost": [1, 1], "passenger_count": [10, 20], "travel_distance": [5, 15],
        }
        metrics = compute_single_objective_metrics("elevator_o3", [0, 1], test_case_data)
        self.assertAlmostEqual(metrics["passenger_count"], 30)
        self.assertAlmostEqual(metrics["travel_distance"], 20)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            compute_single_objective_metrics("nope", [0], {"cost": [1]})


class TestExecutionTimeSeconds(unittest.TestCase):
    def test_sums_wall_clock_across_subproblem_records(self):
        # one repetition = one RawCountsRecord per subproblem/cluster circuit
        records = [make_raw_record(80, 1.0), make_raw_record(80, 1.5), make_raw_record(80, 0.25)]
        self.assertAlmostEqual(compute_execution_time_seconds(records), 2.75)

    def test_empty_records_is_zero(self):
        self.assertAlmostEqual(compute_execution_time_seconds([]), 0.0)

    def test_trex_twirl_instances_all_count_since_each_is_a_fresh_execution(self):
        # TREx: several twirl-instance executions per subproblem, each its
        # own RawCountsRecord -- all of them count towards execution time,
        # unlike raw/MEM/M3 which have one record per subproblem.
        twirl_instances = [make_raw_record(80, 0.5) for _ in range(4)]
        self.assertAlmostEqual(compute_execution_time_seconds(twirl_instances), 2.0)


class TestMitigationOverhead(unittest.TestCase):
    def test_raw_only_sums_shots(self):
        records = [make_raw_record(80, 1.0), make_raw_record(80, 1.5)]
        overhead = compute_mitigation_overhead(records)
        self.assertEqual(overhead.calibration_circuits, 0)
        self.assertEqual(overhead.total_shots, 160)
        self.assertEqual(overhead.calibration_wall_clock_seconds, 0.0)

    def test_mem_calibration_adds_2n_circuits_worth_of_shots(self):
        from piastq_execution.mitigation import make_mem_calibration_record
        import numpy as np

        cal = make_mem_calibration_record(
            np.eye(4), physical_qubits=[0, 1], backend_name="b", backend_version="v",
            shots_per_calibration_circuit=200, calibration_wall_clock_seconds=12.5,
        )
        records = [make_raw_record(80, 1.0)]
        overhead = compute_mitigation_overhead(records, calibration_record=cal, calibration_shots_per_circuit=200)
        self.assertEqual(overhead.calibration_circuits, 4)  # 2**2
        self.assertEqual(overhead.total_shots, 80 + 4 * 200)
        self.assertEqual(overhead.calibration_wall_clock_seconds, 12.5)


class TestEvaluateSingleObjectiveCombo(unittest.TestCase):
    def test_end_to_end_raw_method_one_cluster(self):
        linear = [1, -2]
        quadratic = {(0, 1): 3}
        # raw counts already concentrated on the optimum "10" (energy=-2)
        raw_counts = {"10": 100}

        # cluster_test_cases hold 0-based indices into the dataset's full
        # per-test-case lists (test_case_data), matching how the real
        # execution scripts build bootqa_clusters/clusters_dictionary.
        assignments = [ClusterAssignment(cluster_id=0, cluster_test_cases=[0, 1])]
        test_case_data = {"cost": [5.0, 7.0], "failure_rate": [0.1, 0.9]}
        records = [make_raw_record(100, 0.5)]

        result = evaluate_single_objective_combo(
            combo="toy_qaoa_tcs",
            method="raw",
            dataset="iofrol",
            cluster_assignments=assignments,
            cluster_raw_counts=[raw_counts],
            cluster_qubos=[(linear, quadratic, 2)],
            test_case_data=test_case_data,
            raw_records=records,
        )

        self.assertAlmostEqual(result.execution_time_seconds, 0.5)
        self.assertAlmostEqual(result.probability_of_optimal, 1.0)
        self.assertAlmostEqual(result.qubo_energy, -2.0)
        # "10" -> bits=[0,1] -> selects cluster_test_cases[1]=43
        self.assertAlmostEqual(result.execution_cost, 7.0)
        self.assertAlmostEqual(result.effectiveness["failure_rate"], 0.9)
        self.assertEqual(result.mitigation_overhead.total_shots, 100)
        self.assertGreaterEqual(result.classical_post_processing_seconds, 0.0)


class TestEvaluateMultiObjectiveCombo(unittest.TestCase):
    def test_hand_verified_hv_and_igd_three_objectives(self):
        # Points are (-cost, statement_coverage, fault_coverage), matching
        # SelectQAOA/MOQ-Pipeline.ipynb's actual 3-objective HV/IGD
        # evaluation (total_cost negated, total_coverage, total_faults).
        raw_points = [(-1, 5, 2), (-2, 2, 4)]
        mem_points = [(-1, 6, 3), (-3, 3, 5)]
        all_points = {"raw": raw_points, "mem": mem_points}
        records = [make_raw_record(80, 1.0)]

        result = evaluate_multi_objective_combo(
            combo="toy_qaoa_tcs_multi",
            method="raw",
            pareto_points=raw_points,
            all_methods_points=all_points,
            reference_point=(-10, 0, 0),
            raw_records=records,
        )
        # raw's OWN front (dominance within its own 2 points only): neither
        # (-1,5,2) nor (-2,2,4) dominates the other -> both stay.
        # HV vs ref(-10,0,0), verified two ways (recursive slicing and
        # inclusion-exclusion of the two boxes): box1=(8*2*4)=64,
        # box2=(9*5*2)=90, intersection=(8*2*2)=32, union=64+90-32=122.
        self.assertAlmostEqual(result.hypervolume, 122.0)
        # reference front (union pareto across raw+mem) excludes (-1,5,2)
        # (dominated by mem's (-1,6,3): same cost, strictly more coverage,
        # more faults) but keeps (-2,2,4) -> only 1 of raw's own 2 points
        # survives into the reference frontier.
        self.assertEqual(result.num_non_dominated, 1)
        self.assertGreater(result.igd, 0.0)  # raw's front doesn't cover (-1,6,3) or (-3,3,5)
        self.assertAlmostEqual(result.execution_time_seconds, 1.0)


def _make_single_objective_result(execution_time, total_shots, combo="toy", method="raw"):
    overhead = MitigationOverhead(calibration_circuits=0, total_shots=total_shots, calibration_wall_clock_seconds=0.0)
    return SingleObjectiveEvaluation(
        combo=combo, method=method, qubo_energy=0.0, optimal_bitstring="0",
        probability_of_optimal=1.0, execution_cost=0.0, effectiveness={},
        execution_time_seconds=execution_time, mitigation_overhead=overhead,
        classical_post_processing_seconds=0.0,
    )


class TestExtractMetricSamples(unittest.TestCase):
    def test_extracts_top_level_field(self):
        results = [_make_single_objective_result(1.0, 80), _make_single_objective_result(2.0, 80)]
        samples = extract_metric_samples(results, "execution_time_seconds")
        self.assertEqual(samples, [1.0, 2.0])

    def test_extracts_nested_dotted_path(self):
        results = [_make_single_objective_result(1.0, 80), _make_single_objective_result(2.0, 160)]
        samples = extract_metric_samples(results, "mitigation_overhead.total_shots")
        self.assertEqual(samples, [80.0, 160.0])

    def test_end_to_end_compare_algorithms_on_execution_time(self):
        # Demonstrates the exact workflow the README documents for comparing
        # two algorithms' quantum-hardware execution time with statistical
        # soundness: extract samples per group, then compare_groups().
        qaoa_tcs_results = [_make_single_objective_result(t, 80) for t in [1.0, 1.1, 0.9, 1.05, 0.95]]
        igdec_results = [_make_single_objective_result(t, 80) for t in [5.0, 5.2, 4.8, 5.1, 4.9]]

        groups = {
            "qaoa_tcs": extract_metric_samples(qaoa_tcs_results, "execution_time_seconds"),
            "igdec_qaoa": extract_metric_samples(igdec_results, "execution_time_seconds"),
        }
        comparison = compare_groups(groups)

        self.assertIn(comparison.omnibus_test, ("anova", "kruskal_wallis"))
        self.assertEqual(len(comparison.pairwise), 1)
        pair = comparison.pairwise[0]
        # igdec_qaoa's execution times are all clearly larger than qaoa_tcs's
        if pair.effect_size_name == "a12":
            self.assertLess(pair.effect_size, 0.5)
        else:
            self.assertLess(pair.effect_size, 0.0)


if __name__ == "__main__":
    unittest.main()
