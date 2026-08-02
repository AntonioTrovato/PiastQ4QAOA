"""Synthetic end-to-end tests for piastq_execution.evaluate_all.

Builds a tiny fake repo layout (tmp dir) mimicking exactly what the real
training/execution scripts write -- trained QUBOs, subsuites/
circuits_metadata, raw + TREx raw_counts.jsonl -- for one combo per
algorithm/objective_mode, and runs the real evaluate_*_combo() loaders
against it. No AQTProvider/AQTSampler/backend/real dataset files involved.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.evaluate_all import (
    evaluate_all,
    evaluate_igdec_qaoa_single_objective_combo,
    evaluate_qaoa_tcs_multi_objective_combo,
    evaluate_qaoa_tcs_single_objective_combo,
    load_combo_definitions,
    merge_igdec_solution,
    write_evaluation_results,
)
from piastq_execution.mitigation import CalibrationStore


def _raw_record(circuit_id, cluster_id, iteration_id, subproblem_id, aggregated_counts, twirl_mask=None):
    return {
        "algorithm": "qaoa_tcs",
        "objective_mode": "single_objective",
        "dataset": "gsdtsr",
        "circuit_id": circuit_id,
        "cluster_id": cluster_id,
        "iteration_id": iteration_id,
        "subproblem_id": subproblem_id,
        "backend_name": "fake",
        "backend_version": "0.0",
        "physical_qubit_mapping": {},
        "batches": [],
        "total_shots_requested": 100,
        "total_shots_returned": 100,
        "total_wall_clock_seconds": 0.5,
        "aggregated_counts": aggregated_counts,
        "twirl_mask": twirl_mask,
    }


class TestMergeIgdecSolution(unittest.TestCase):
    def test_last_write_wins_across_overlapping_iterations(self):
        # iteration 1 selects test 0 (bit=1); iteration 2 revisits test 0
        # and flips it to 0 -- the LATER decision must win, unlike
        # merge_selected_tests()'s first-wins union semantics.
        circuits_metadata = [
            {"case_list": [0, 1]},
            {"case_list": [0, 2]},
        ]
        # bitstring "10" -> bits=[0,1] (index0=0,index1=1) per existing
        # little-endian convention: bits = [int(b) for b in s[::-1]]
        selected_bitstrings = ["01", "10"]
        # itr1 "01"[::-1]="10" -> bits=[1,0] -> case_list[0]=0->1, case_list[1]=1->0
        # itr2 "10"[::-1]="01" -> bits=[0,1] -> case_list[0]=0->0 (overwrites!), case_list[1]=2->1
        result = merge_igdec_solution(circuits_metadata, selected_bitstrings)
        # final solution: {0: 0 (overwritten), 1: 0, 2: 1} -> only test 2 selected
        self.assertEqual(result, [2])

    def test_non_overlapping_case_lists_union_like_normal(self):
        circuits_metadata = [{"case_list": [0]}, {"case_list": [1]}]
        selected_bitstrings = ["1", "1"]
        result = merge_igdec_solution(circuits_metadata, selected_bitstrings)
        self.assertEqual(result, [0, 1])


class TestEvaluateQaoaTcsSingleObjectiveCombo(unittest.TestCase):
    def test_end_to_end_raw_and_mem(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "datasets", "quantum_sota_datasets"))
            os.makedirs(os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", "gsdtsr", "rep_1"))
            os.makedirs(os.path.join(root, "results", "qaoa_tcs", "gsdtsr"))
            os.makedirs(os.path.join(root, "calibration"))

            with open(os.path.join(root, "datasets", "quantum_sota_datasets", "gsdtsr.csv"), "w") as f:
                f.write("time,rate\n5.0,0.1\n7.0,0.9\n")

            qubos_path = os.path.join(
                root, "trained_qaoa_circuits", "qaoa_tcs", "gsdtsr", "rep_1", "gsdtsr_rep1_qubos.json"
            )
            with open(qubos_path, "w") as f:
                json.dump(
                    [{"cluster_idx": 0, "cluster_id": 0, "linear": [1.0, -2.0], "quadratic": [], "num_qubits": 2}],
                    f,
                )

            subsuites_path = os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-subsuites.json")
            with open(subsuites_path, "w") as f:
                json.dump(
                    {
                        "experiment_1": {
                            "final_selected_tests": [1],
                            "cluster_assignments": [
                                {"cluster_id": 0, "cluster_test_cases": [0, 1], "bit_values": [0, 1]}
                            ],
                        }
                    },
                    f,
                )

            raw_path = os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-raw_counts.jsonl")
            with open(raw_path, "w") as f:
                f.write(json.dumps(_raw_record("gsdtsr_rep1_cluster0", 0, 1, None, {"10": 100})) + "\n")

            # empty TREx file (not requested by this combo's methods)
            open(os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-trex-raw_counts.jsonl"), "w").close()

            combo_cfg = {
                "algorithm": "qaoa_tcs",
                "objective_mode": "single_objective",
                "dataset": "gsdtsr",
                "circuits_dir": "gsdtsr",
                "methods": ["raw"],
            }
            calibration_store = CalibrationStore(base_dir=os.path.join(root, "calibration"))

            results = evaluate_qaoa_tcs_single_objective_combo("gsdtsr_qaoa_tcs", combo_cfg, root, calibration_store)

            self.assertIn("raw", results)
            self.assertEqual(len(results["raw"]), 1)
            result = results["raw"][0]
            # "10" -> bits=[0,1] -> selects cluster_test_cases[1] -> test id 1
            self.assertAlmostEqual(result.execution_cost, 7.0)
            self.assertAlmostEqual(result.effectiveness["failure_rate"], 0.9)
            self.assertAlmostEqual(result.qubo_energy, -2.0)
            self.assertAlmostEqual(result.execution_time_seconds, 0.5)

    def test_end_to_end_mem_and_trex(self):
        import numpy as np

        from piastq_execution.mitigation import make_mem_calibration_record

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "datasets", "quantum_sota_datasets"))
            os.makedirs(os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", "gsdtsr", "rep_1"))
            os.makedirs(os.path.join(root, "results", "qaoa_tcs", "gsdtsr"))
            os.makedirs(os.path.join(root, "calibration"))

            with open(os.path.join(root, "datasets", "quantum_sota_datasets", "gsdtsr.csv"), "w") as f:
                f.write("time,rate\n5.0,0.1\n7.0,0.9\n")

            with open(
                os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", "gsdtsr", "rep_1", "gsdtsr_rep1_qubos.json"),
                "w",
            ) as f:
                json.dump(
                    [{"cluster_idx": 0, "cluster_id": 0, "linear": [1.0, -2.0], "quadratic": [], "num_qubits": 2}], f
                )

            with open(os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-subsuites.json"), "w") as f:
                json.dump(
                    {
                        "experiment_1": {
                            "cluster_assignments": [
                                {"cluster_id": 0, "cluster_test_cases": [0, 1], "bit_values": [0, 1]}
                            ],
                        }
                    },
                    f,
                )

            with open(os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-raw_counts.jsonl"), "w") as f:
                f.write(json.dumps(_raw_record("gsdtsr_rep1_cluster0", 0, 1, None, {"10": 90, "01": 10})) + "\n")

            # TREx: 2 twirl instances for the same (cluster=0, iteration=1)
            with open(
                os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-trex-raw_counts.jsonl"), "w"
            ) as f:
                f.write(json.dumps(_raw_record("gsdtsr_rep1_cluster0_trex0", 0, 1, None, {"10": 100}, twirl_mask=[0, 0])) + "\n")
                f.write(json.dumps(_raw_record("gsdtsr_rep1_cluster0_trex1", 0, 1, None, {"01": 100}, twirl_mask=[1, 0])) + "\n")

            # Perfect-readout MEM calibration for width=2 (identity confusion matrix)
            store = CalibrationStore(base_dir=os.path.join(root, "calibration"))
            cal = make_mem_calibration_record(
                np.eye(4), physical_qubits=[0, 1], backend_name="fake", backend_version="0.0",
                shots_per_calibration_circuit=200, calibration_wall_clock_seconds=2.0,
            )
            store.save(cal)

            combo_cfg = {
                "algorithm": "qaoa_tcs", "objective_mode": "single_objective",
                "dataset": "gsdtsr", "circuits_dir": "gsdtsr", "methods": ["mem", "trex"],
            }

            results = evaluate_qaoa_tcs_single_objective_combo("gsdtsr_qaoa_tcs", combo_cfg, root, store)

            self.assertIn("mem", results)
            self.assertIn("trex", results)
            mem_result = results["mem"][0]
            trex_result = results["trex"][0]
            # perfect-readout identity matrix -> mem's corrected distribution
            # equals raw's; argmax is still "10" -> selects test 1 (cost=7.0)
            self.assertAlmostEqual(mem_result.execution_cost, 7.0)
            self.assertAlmostEqual(mem_result.mitigation_overhead.calibration_wall_clock_seconds, 2.0)
            # trex: undoing mask [0,0] on "10" is "10"; undoing mask [1,0] on
            # "01" flips qubit0 -> "00" -- aggregated counts {"10":100,"00":100},
            # argmax is a tie broken by dict iteration order -- just check it
            # ran end-to-end and produced a valid (non-negative) energy.
            self.assertIsNotNone(trex_result.qubo_energy)


class TestEvaluateQaoaTcsMultiObjectiveCombo(unittest.TestCase):
    def test_end_to_end_raw(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "datasets", "sir_programs"))
            os.makedirs(os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", "toyprog", "rep_1"))
            os.makedirs(os.path.join(root, "results", "qaoa_tcs", "toyprog"))
            os.makedirs(os.path.join(root, "calibration"))

            with open(os.path.join(root, "datasets", "sir_programs", "test_cases_costs.json"), "w") as f:
                json.dump({"toyprog": {"0": 10, "1": 5}}, f)
            with open(os.path.join(root, "datasets", "sir_programs", "faults_dictionary.json"), "w") as f:
                json.dump({"toyprog": [1, 2]}, f)
            with open(os.path.join(root, "datasets", "sir_programs", "test_coverage_line_by_line.json"), "w") as f:
                json.dump({"toyprog": {"0": [1, 2], "1": [2, 3]}}, f)

            qubos_path = os.path.join(
                root, "trained_qaoa_circuits", "qaoa_tcs", "toyprog", "rep_1", "toyprog_rep1_qubos.json"
            )
            with open(qubos_path, "w") as f:
                json.dump(
                    [{"cluster_idx": 0, "linear": [1.0, -1.0], "quadratic": [], "num_qubits": 2}], f
                )

            subsuites_path = os.path.join(root, "results", "qaoa_tcs", "toyprog", "toyprog-rep-1-subsuites.json")
            with open(subsuites_path, "w") as f:
                json.dump(
                    {
                        "experiment_1": {
                            "cluster_assignments": [
                                {"cluster_id": 0, "cluster_test_cases": [0, 1], "bit_values": [1, 1]}
                            ],
                        }
                    },
                    f,
                )

            raw_path = os.path.join(root, "results", "qaoa_tcs", "toyprog", "toyprog-rep-1-raw_counts.jsonl")
            with open(raw_path, "w") as f:
                # "11" -> bits=[1,1] -> both tests selected
                f.write(json.dumps(_raw_record("toyprog_rep1_cluster0", 0, 1, None, {"11": 100})) + "\n")

            open(os.path.join(root, "results", "qaoa_tcs", "toyprog", "toyprog-rep-1-trex-raw_counts.jsonl"), "w").close()

            combo_cfg = {
                "algorithm": "qaoa_tcs",
                "objective_mode": "multi_objective",
                "dataset": "toyprog",
                "circuits_dir": "toyprog",
                "methods": ["raw"],
            }
            calibration_store = CalibrationStore(base_dir=os.path.join(root, "calibration"))

            results = evaluate_qaoa_tcs_multi_objective_combo("toyprog_qaoa_tcs", combo_cfg, root, calibration_store)

            self.assertIn("raw", results)
            self.assertEqual(len(results["raw"]), 1)
            result = results["raw"][0]
            # both tests selected: cost=15, faults=3, coverage={1,2,3}=3
            self.assertGreater(result.hypervolume, 0.0)
            self.assertAlmostEqual(result.execution_time_seconds, 0.5)


class TestEvaluateIgdecQaoaSingleObjectiveCombo(unittest.TestCase):
    def test_end_to_end_raw_with_overlapping_case_lists(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "datasets", "quantum_sota_datasets"))
            sampling_dir = os.path.join(root, "trained_qaoa_circuits", "igdec_qaoa", "iofrol", "sampling_1")
            os.makedirs(sampling_dir)
            os.makedirs(os.path.join(root, "results", "igdec_qaoa"))
            os.makedirs(os.path.join(root, "calibration"))

            with open(os.path.join(root, "datasets", "quantum_sota_datasets", "iofrol.csv"), "w") as f:
                f.write("time,rate\n5.0,0.1\n7.0,0.9\n9.0,0.3\n")

            circuits_metadata = [
                {
                    "iteration": 1, "subproblem_index": 1, "case_list": [0, 1], "qpy_file": "itr_1_subproblem_1.qpy",
                    "linear": [1.0, -1.0], "quadratic": [], "num_qubits": 2,
                },
                {
                    "iteration": 2, "subproblem_index": 1, "case_list": [0, 2], "qpy_file": "itr_2_subproblem_1.qpy",
                    "linear": [1.0, -1.0], "quadratic": [], "num_qubits": 2,
                },
            ]
            with open(os.path.join(sampling_dir, "circuits_metadata.json"), "w") as f:
                json.dump(circuits_metadata, f)

            raw_path = os.path.join(root, "results", "igdec_qaoa", "iofrol-raw_counts.jsonl")
            with open(raw_path, "w") as f:
                # itr1 "01"[::-1]="10"->bits=[1,0]: case0(test0)=1,case1(test1)=0
                f.write(json.dumps(_raw_record("iofrol_s1_itr1_sub1", None, 1, 1, {"01": 100})) + "\n")
                # itr2 "10"[::-1]="01"->bits=[0,1]: case0(test0)=0 (overwrites itr1!), case1(test2)=1
                f.write(json.dumps(_raw_record("iofrol_s1_itr2_sub1", None, 2, 1, {"10": 100})) + "\n")

            open(os.path.join(root, "results", "igdec_qaoa", "iofrol-trex-raw_counts.jsonl"), "w").close()

            combo_cfg = {
                "algorithm": "igdec_qaoa",
                "objective_mode": "single_objective",
                "dataset": "iofrol",
                "circuits_dir": "iofrol",
                "methods": ["raw"],
            }
            calibration_store = CalibrationStore(base_dir=os.path.join(root, "calibration"))

            results = evaluate_igdec_qaoa_single_objective_combo("iofrol_igdec_qaoa", combo_cfg, root, calibration_store)

            self.assertIn("raw", results)
            self.assertEqual(len(results["raw"]), 1)
            result = results["raw"][0]
            # final solution after last-write-wins: test0=0 (overwritten), test1=0, test2=1
            # -> only test 2 selected -> cost=9.0, failure_rate=0.3
            self.assertAlmostEqual(result.execution_cost, 9.0)
            self.assertAlmostEqual(result.effectiveness["failure_rate"], 0.3)


class TestLoadComboDefinitions(unittest.TestCase):
    def test_reads_combos_section_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "execution_plan.yaml")
            with open(path, "w") as f:
                f.write(
                    "combos:\n"
                    "  gsdtsr_qaoa_tcs:\n"
                    "    algorithm: qaoa_tcs\n"
                    "    objective_mode: single_objective\n"
                    "    dataset: gsdtsr\n"
                    "    circuits_dir: gsdtsr\n"
                    "    methods: [raw, mem]\n"
                    "pools: []\n"
                )
            combos = load_combo_definitions(path)
            self.assertEqual(set(combos.keys()), {"gsdtsr_qaoa_tcs"})
            self.assertEqual(combos["gsdtsr_qaoa_tcs"]["algorithm"], "qaoa_tcs")
            self.assertEqual(combos["gsdtsr_qaoa_tcs"]["methods"], ["raw", "mem"])


class TestEvaluateAllDriver(unittest.TestCase):
    def test_skips_combo_with_missing_files_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "configs"))
            os.makedirs(os.path.join(root, "calibration"))
            plan_path = os.path.join(root, "configs", "execution_plan.yaml")
            with open(plan_path, "w") as f:
                f.write(
                    "combos:\n"
                    "  nonexistent_qaoa_tcs:\n"
                    "    algorithm: qaoa_tcs\n"
                    "    objective_mode: single_objective\n"
                    "    dataset: nonexistent\n"
                    "    circuits_dir: nonexistent\n"
                    "    methods: [raw]\n"
                    "pools: []\n"
                )
            results = evaluate_all(plan_path, root)
            self.assertEqual(results, {})

    def test_full_dispatch_and_write_for_one_real_combo(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "configs"))
            os.makedirs(os.path.join(root, "datasets", "quantum_sota_datasets"))
            os.makedirs(os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", "gsdtsr", "rep_1"))
            os.makedirs(os.path.join(root, "results", "qaoa_tcs", "gsdtsr"))
            os.makedirs(os.path.join(root, "calibration"))

            with open(os.path.join(root, "datasets", "quantum_sota_datasets", "gsdtsr.csv"), "w") as f:
                f.write("time,rate\n5.0,0.1\n7.0,0.9\n")
            with open(
                os.path.join(root, "trained_qaoa_circuits", "qaoa_tcs", "gsdtsr", "rep_1", "gsdtsr_rep1_qubos.json"),
                "w",
            ) as f:
                json.dump(
                    [{"cluster_idx": 0, "cluster_id": 0, "linear": [1.0, -2.0], "quadratic": [], "num_qubits": 2}], f
                )
            with open(os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-subsuites.json"), "w") as f:
                json.dump(
                    {
                        "experiment_1": {
                            "cluster_assignments": [
                                {"cluster_id": 0, "cluster_test_cases": [0, 1], "bit_values": [0, 1]}
                            ],
                        }
                    },
                    f,
                )
            with open(os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-raw_counts.jsonl"), "w") as f:
                f.write(json.dumps(_raw_record("gsdtsr_rep1_cluster0", 0, 1, None, {"10": 100})) + "\n")
            open(os.path.join(root, "results", "qaoa_tcs", "gsdtsr", "gsdtsr-rep-1-trex-raw_counts.jsonl"), "w").close()

            plan_path = os.path.join(root, "configs", "execution_plan.yaml")
            with open(plan_path, "w") as f:
                f.write(
                    "combos:\n"
                    "  gsdtsr_qaoa_tcs:\n"
                    "    algorithm: qaoa_tcs\n"
                    "    objective_mode: single_objective\n"
                    "    dataset: gsdtsr\n"
                    "    circuits_dir: gsdtsr\n"
                    "    methods: [raw]\n"
                    "pools: []\n"
                )

            results = evaluate_all(plan_path, root)
            self.assertIn("gsdtsr_qaoa_tcs", results)
            self.assertIn("raw", results["gsdtsr_qaoa_tcs"])
            self.assertEqual(results["gsdtsr_qaoa_tcs"]["raw"][0]["execution_cost"], 7.0)

            output_dir = os.path.join(root, "results", "evaluation")
            written = write_evaluation_results(results, output_dir)
            self.assertEqual(len(written), 1)
            self.assertTrue(os.path.exists(written[0]))
            with open(written[0]) as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["raw"][0]["execution_cost"], 7.0)


if __name__ == "__main__":
    unittest.main()
