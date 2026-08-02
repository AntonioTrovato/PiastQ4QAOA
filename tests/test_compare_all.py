"""Synthetic unit tests for piastq_execution.compare_all.

Uses small, hand-built results/evaluation/*.json-shaped dicts (as
evaluate_all.py would produce) -- no real evaluation pipeline or hardware
data involved.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.compare_all import (
    compare_algorithms,
    compare_all,
    compare_combo,
    comparison_to_json_dict,
    extract_samples,
    load_evaluation_results,
    pair_combos_by_dataset,
    render_report,
)


def _single_obj_record(qubo_energy, execution_cost, execution_time_seconds, calibration_seconds=0.0):
    return {
        "combo": "toy", "method": "x", "qubo_energy": qubo_energy, "optimal_bitstring": "0",
        "probability_of_optimal": 0.9, "execution_cost": execution_cost, "effectiveness": {"failure_rate": 1.0},
        "execution_time_seconds": execution_time_seconds,
        "mitigation_overhead": {"calibration_circuits": 4, "total_shots": 100, "calibration_wall_clock_seconds": calibration_seconds},
        "classical_post_processing_seconds": 0.01,
    }


def _multi_obj_record(hypervolume, execution_time_seconds):
    return {
        "combo": "toy", "method": "x", "num_non_dominated": 2, "hypervolume": hypervolume, "igd": 0.1,
        "execution_time_seconds": execution_time_seconds,
        "mitigation_overhead": {"calibration_circuits": 0, "total_shots": 0, "calibration_wall_clock_seconds": 0.0},
        "classical_post_processing_seconds": 0.01,
    }


class TestExtractSamples(unittest.TestCase):
    def test_extracts_top_level_metric(self):
        records = [_single_obj_record(1.0, 5.0, 0.5), _single_obj_record(2.0, 6.0, 0.6)]
        self.assertEqual(extract_samples(records, "qubo_energy"), [1.0, 2.0])

    def test_extracts_dotted_path(self):
        records = [_single_obj_record(1.0, 5.0, 0.5, calibration_seconds=3.0)]
        self.assertEqual(extract_samples(records, "mitigation_overhead.calibration_wall_clock_seconds"), [3.0])

    def test_skips_none_instead_of_raising(self):
        records = [_single_obj_record(1.0, 5.0, 0.5)]
        records[0]["execution_cost"] = None
        self.assertEqual(extract_samples(records, "execution_cost"), [])

    def test_skips_missing_key(self):
        records = [{"qubo_energy": 1.0}]
        self.assertEqual(extract_samples(records, "hypervolume"), [])


class TestCompareCombo(unittest.TestCase):
    def test_single_objective_combo_produces_quality_and_effectiveness_sections(self):
        combo_results = {
            "raw": [_single_obj_record(5.0, 10.0, 1.0), _single_obj_record(5.2, 10.5, 1.1)],
            "mem": [_single_obj_record(1.0, 8.0, 1.5), _single_obj_record(1.1, 8.2, 1.4)],
        }
        report = compare_combo("toy_combo", combo_results)
        self.assertIn("qubo_energy", report["rq1_rq3a_quality"])
        self.assertIn("execution_cost", report["rq2_effectiveness"])
        self.assertIn("execution_time_seconds", report["rq3b_cost"])
        pair = report["rq1_rq3a_quality"]["qubo_energy"].pairwise[0]
        self.assertEqual({pair.group_a, pair.group_b}, {"raw", "mem"})

    def test_multi_objective_combo_uses_hv_igd_not_execution_cost(self):
        combo_results = {
            "raw": [_multi_obj_record(10.0, 1.0), _multi_obj_record(10.5, 1.1)],
            "mem": [_multi_obj_record(15.0, 1.5), _multi_obj_record(15.5, 1.4)],
        }
        report = compare_combo("toy_multi", combo_results)
        self.assertIn("hypervolume", report["rq1_rq3a_quality"])
        self.assertEqual(report["rq2_effectiveness"], {})

    def test_single_method_produces_no_comparisons(self):
        combo_results = {"raw": [_single_obj_record(5.0, 10.0, 1.0)]}
        report = compare_combo("toy_combo", combo_results)
        self.assertEqual(report["rq1_rq3a_quality"], {})


class TestPairCombosByDataset(unittest.TestCase):
    def test_pairs_matching_dataset_across_algorithms(self):
        combo_defs = {
            "gsdtsr_qaoa_tcs": {"algorithm": "qaoa_tcs", "dataset": "gsdtsr"},
            "gsdtsr_igdec_qaoa": {"algorithm": "igdec_qaoa", "dataset": "gsdtsr"},
            "flex_qaoa_tcs": {"algorithm": "qaoa_tcs", "dataset": "flex"},  # no igdec counterpart
        }
        pairs = pair_combos_by_dataset(combo_defs)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["dataset"], "gsdtsr")
        self.assertEqual(pairs[0]["qaoa_tcs"], "gsdtsr_qaoa_tcs")
        self.assertEqual(pairs[0]["igdec_qaoa"], "gsdtsr_igdec_qaoa")


class TestCompareAlgorithms(unittest.TestCase):
    def test_compares_shared_methods_across_algorithms(self):
        all_results = {
            "gsdtsr_qaoa_tcs": {
                "raw": [_single_obj_record(1.0, 1.0, 0.5), _single_obj_record(1.0, 1.0, 0.6)],
            },
            "gsdtsr_igdec_qaoa": {
                "raw": [_single_obj_record(1.0, 1.0, 5.0), _single_obj_record(1.0, 1.0, 5.5)],
            },
        }
        combo_defs = {
            "gsdtsr_qaoa_tcs": {"algorithm": "qaoa_tcs", "dataset": "gsdtsr"},
            "gsdtsr_igdec_qaoa": {"algorithm": "igdec_qaoa", "dataset": "gsdtsr"},
        }
        reports = compare_algorithms(all_results, combo_defs)
        self.assertEqual(len(reports), 1)
        self.assertIn("raw", reports[0]["by_method"])


class TestEndToEnd(unittest.TestCase):
    def test_compare_all_and_render_report(self):
        with tempfile.TemporaryDirectory() as root:
            eval_dir = os.path.join(root, "results", "evaluation")
            os.makedirs(eval_dir)
            with open(os.path.join(eval_dir, "gsdtsr_qaoa_tcs.json"), "w") as f:
                json.dump(
                    {
                        "raw": [_single_obj_record(5.0, 10.0, 0.5), _single_obj_record(5.1, 10.1, 0.6)],
                        "mem": [_single_obj_record(1.0, 8.0, 1.5), _single_obj_record(1.1, 8.1, 1.4)],
                    },
                    f,
                )
            with open(os.path.join(eval_dir, "gsdtsr_igdec_qaoa.json"), "w") as f:
                json.dump(
                    {"raw": [_single_obj_record(2.0, 3.0, 4.0), _single_obj_record(2.1, 3.1, 4.2)]}, f
                )

            plan_path = os.path.join(root, "configs", "execution_plan.yaml")
            os.makedirs(os.path.dirname(plan_path))
            with open(plan_path, "w") as f:
                f.write(
                    "combos:\n"
                    "  gsdtsr_qaoa_tcs:\n"
                    "    algorithm: qaoa_tcs\n"
                    "    dataset: gsdtsr\n"
                    "  gsdtsr_igdec_qaoa:\n"
                    "    algorithm: igdec_qaoa\n"
                    "    dataset: gsdtsr\n"
                    "pools: []\n"
                )

            comparison = compare_all(eval_dir, plan_path)
            self.assertEqual(len(comparison["per_combo"]), 2)
            self.assertEqual(len(comparison["rq4_algorithm_comparison"]), 1)

            report_text = render_report(comparison)
            self.assertIn("RQ4", report_text)
            self.assertIn("gsdtsr_qaoa_tcs", report_text)

            json_dict = comparison_to_json_dict(comparison)
            json.dumps(json_dict)  # must be JSON-serializable
            self.assertEqual(len(json_dict["per_combo"]), 2)

    def test_missing_evaluation_dir_returns_empty(self):
        results = load_evaluation_results("/tmp/piastq_does_not_exist_evaluation_dir")
        self.assertEqual(results, {})


if __name__ == "__main__":
    unittest.main()
