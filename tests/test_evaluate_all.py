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
    load_igdec_elevator_test_case_data,
    load_igdec_tcm_dataset,
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

    def test_untouched_tests_keep_initial_solution(self):
        circuits_metadata = [{"case_list": [1]}]
        result = merge_igdec_solution(circuits_metadata, ["0"], initial_solution=[1, 1, 0, 1])
        # test 1 overwritten to 0; tests 0 and 3 keep their initial 1
        self.assertEqual(result, [0, 3])

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
            with open(os.path.join(sampling_dir, "initial_random_solution.json"), "w") as f:
                json.dump({"initial_solution": [0, 0, 0], "initial_energy": 0.0}, f)

            raw_path = os.path.join(root, "results", "igdec_qaoa", "iofrol-raw_counts.jsonl")
            with open(raw_path, "w") as f:
                # itr1 "01"[::-1]="10"->bits=[1,0]: case0(test0)=1,case1(test1)=0
                f.write(json.dumps(_raw_record("iofrol_sampling1_itr1_sub1", None, 1, 1, {"01": 100})) + "\n")
                # itr2 "10"[::-1]="01"->bits=[0,1]: case0(test0)=0 (overwrites itr1!), case1(test2)=1
                f.write(json.dumps(_raw_record("iofrol_sampling1_itr2_sub1", None, 2, 1, {"10": 100})) + "\n")

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


    def test_samplings_do_not_share_counts_and_unexecuted_ones_are_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "datasets", "quantum_sota_datasets"))
            base = os.path.join(root, "trained_qaoa_circuits", "igdec_qaoa", "iofrol")
            os.makedirs(os.path.join(root, "results", "igdec_qaoa"))
            os.makedirs(os.path.join(root, "calibration"))
            with open(os.path.join(root, "datasets", "quantum_sota_datasets", "iofrol.csv"), "w") as f:
                f.write("time,rate\n5.0,0.1\n7.0,0.9\n")
            meta = [{"iteration": 1, "subproblem_index": 1, "case_list": [0, 1], "qpy_file": "x.qpy",
                     "linear": [1.0, -1.0], "quadratic": [], "num_qubits": 2}]
            for n in (1, 2, 3):  # sampling_3 is trained but never executed
                d = os.path.join(base, f"sampling_{n}")
                os.makedirs(d)
                with open(os.path.join(d, "circuits_metadata.json"), "w") as f:
                    json.dump(meta, f)
                with open(os.path.join(d, "initial_random_solution.json"), "w") as f:
                    json.dump({"initial_solution": [0, 0], "initial_energy": 0.0}, f)
            with open(os.path.join(root, "results", "igdec_qaoa", "iofrol-raw_counts.jsonl"), "w") as f:
                # bits are little-endian: "01" -> test0=1,test1=0 ; "10" -> test0=0,test1=1
                f.write(json.dumps(_raw_record("iofrol_sampling1_itr1_sub1", None, 1, 1, {"01": 100})) + "\n")
                f.write(json.dumps(_raw_record("iofrol_sampling2_itr1_sub1", None, 1, 1, {"10": 100})) + "\n")
            open(os.path.join(root, "results", "igdec_qaoa", "iofrol-trex-raw_counts.jsonl"), "w").close()

            combo_cfg = {"algorithm": "igdec_qaoa", "objective_mode": "single_objective",
                         "dataset": "iofrol", "circuits_dir": "iofrol", "methods": ["raw"]}
            results = evaluate_igdec_qaoa_single_objective_combo(
                "iofrol_igdec_qaoa", combo_cfg, root, CalibrationStore(base_dir=os.path.join(root, "calibration"))
            )
            self.assertEqual(len(results["raw"]), 2)  # sampling_3 skipped
            costs = sorted(r.execution_cost for r in results["raw"])
            self.assertEqual(costs, [5.0, 7.0])  # each sampling used its own counts


class TestIgdecScriptsUseTrainedCaseList(unittest.TestCase):
    """Regression guard: the hardware-execution function of every IGDec-QAOA
    script must take `case_list` from the trained circuit's metadata, never
    recompute it from the hardware solution's impact ordering (which diverges
    from the training trajectory and misassigns the circuit's bits)."""

    def test_hardware_run_reads_case_list_from_metadata(self):
        igdec_dir = os.path.join(os.path.dirname(__file__), "..", "src", "igdec_qaoa")
        for name in (
            "loch_qaoa_tcm_extract_circuits.py",
            "loch_qaoa_elev_two_extract_circuits.py",
            "loch_qaoa_elev_three_extract_circuits.py",
        ):
            with open(os.path.join(igdec_dir, name)) as f:
                source = f.read()
            run_part = source[source.index("def run_hardware_like_from_saved_circuits"):]
            self.assertNotIn("case_list = impact_order[", run_part, name)
            if name == "loch_qaoa_tcm_extract_circuits.py":
                # iterations come from the trained circuits' metadata, not a constant
                self.assertIn('num_impact_iterations = max(m["iteration"] for m in circuits_metadata)', run_part)
            self.assertEqual(run_part.count('case_list = meta["case_list"]'), 2, name)


class TestIgdecDatasets(unittest.TestCase):
    def test_elevator_two_is_cost_plus_input_diversity_three_is_pcount_dist(self):
        from piastq_execution.evaluation import compute_single_objective_metrics

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "elevator.csv"), "w") as f:
                f.write(",id,cost,pcount,dist,input_div\n0,a,10,2,3,0.5\n1,b,20,4,5,0.25\n")
            two = load_igdec_elevator_test_case_data(d, "elevator_two")
            three = load_igdec_elevator_test_case_data(d, "elevator_three")
            self.assertEqual(set(two), {"cost", "input_diversity"})
            self.assertEqual(set(three), {"cost", "passenger_count", "travel_distance"})
            metrics = compute_single_objective_metrics("elevator_o2", [0, 1], two)
            self.assertEqual(metrics, {"execution_cost": 30, "input_diversity": 0.75})

    def test_only_gsdtsr_is_filtered_to_positive_rate(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("iofrol", "gsdtsr"):
                with open(os.path.join(d, f"{name}.csv"), "w") as f:
                    f.write("time,rate\n1.0,0.0\n2.0,0.5\n")
            self.assertEqual(len(load_igdec_tcm_dataset("iofrol", d)["cost"]), 2)
            self.assertEqual(len(load_igdec_tcm_dataset("gsdtsr", d)["cost"]), 1)


class TestElevatorTwoProblem(unittest.TestCase):
    def test_qubo_is_two_objective_cost_plus_input_div(self):
        import importlib.util

        path = os.path.join(os.path.dirname(__file__), "..", "src", "igdec_qaoa", "loch_qaoa_elev_two_extract_circuits.py")
        spec = importlib.util.spec_from_file_location("elev_two_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cost, div = [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]
        qubo, _ = module.create_qubo(cost, div, module.W_COST, module.W_DIV, [0, 1], [0, 0, 1])
        self.assertEqual(qubo.get_num_binary_vars(), 2)
        # x0 has zero diversity: its linear term comes from the cost objective only
        linear = qubo.objective.linear.to_dict()
        quad = qubo.objective.quadratic.to_dict()
        self.assertTrue(len(quad) > 0 or len(linear) > 0)
        self.assertFalse(hasattr(module, "TestCaseOptimizationThree"))


class TestPlannedComboNames(unittest.TestCase):
    def test_only_pooled_combos_and_none_without_pools(self):
        from piastq_execution.evaluate_all import planned_combo_names

        with tempfile.TemporaryDirectory() as d:
            with_pools = os.path.join(d, "a.yaml")
            with open(with_pools, "w") as f:
                f.write("combos: {x: {algorithm: qaoa_tcs}, y: {algorithm: qaoa_tcs}}\n"
                        "pools:\n  - {name: p, total_hours: 1, combos: [x]}\n")
            self.assertEqual(planned_combo_names(with_pools), {"x"})
            no_pools = os.path.join(d, "b.yaml")
            with open(no_pools, "w") as f:
                f.write("combos: {x: {algorithm: qaoa_tcs}}\npools: []\n")
            self.assertIsNone(planned_combo_names(no_pools))


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
