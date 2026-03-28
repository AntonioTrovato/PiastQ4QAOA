import statistics
import json
import os
import time
import random
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from docplex.mp.model import Model

from qiskit import qpy
from qiskit.result import QuasiDistribution
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler


# ============================================================
# CONFIG
# ============================================================

RUN_LABEL = "elevator_three"
DATASET_FILENAME = "elevator"   # cambia qui solo se serve un csv diverso
NUM_EXPERIMENT = 30
REPS = 1
PROBLEM_SIZE = 7
NUM_ITERATIONS = 30

# ============================================================
# PROBLEM DEFINITION
# ============================================================

class TestCaseOptimizationThree(OptimizationApplication):
    def __init__(
        self,
        cost: List[float],
        pcount: List[float],
        dist: List[float],
        w1: float,
        w2: float,
        w3: float,
        sample: List[int],
        solution: List[int]
    ) -> None:
        self._cost = cost
        self._pcount = pcount
        self._dist = dist
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3
        self._sample = sample
        self._solution = solution

    def to_quadratic_program(self) -> QuadraticProgram:
        mdl = Model(name="Knapsack")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in self._sample}

        obj_cost = 0
        obj_pcount = 0
        obj_dist = 0

        for i in range(len(self._solution)):
            if i in self._sample:
                obj_cost += self._cost[i] * x[i]
                obj_pcount += self._pcount[i] * x[i]
                obj_dist += self._dist[i] * x[i]
            else:
                obj_cost += self._cost[i] * self._solution[i]
                obj_pcount += self._pcount[i] * self._solution[i]
                obj_dist += self._dist[i] * self._solution[i]

        cost_sum = sum(self._cost)
        pcount_sum = sum(self._pcount)
        dist_sum = sum(self._dist)

        obj_cost = pow(obj_cost / cost_sum, 2)
        obj_pcount = pow((obj_pcount - pcount_sum) / pcount_sum, 2)
        obj_dist = pow((obj_dist - dist_sum) / dist_sum, 2)

        obj = self._w1 * obj_cost + self._w2 * obj_pcount + self._w3 * obj_dist
        mdl.minimize(obj)

        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        x = self._result_to_x(result)
        return [i for i, value in enumerate(x) if value]


def create_qubo(cost, pcount, dist, w1, w2, w3, sample, solution):
    testcase = TestCaseOptimizationThree(cost, pcount, dist, w1, w2, w3, sample, solution)
    prob = testcase.to_quadratic_program()
    return prob, testcase


def get_data(data):
    cost = data["cost"].values.tolist()
    p_count = data["pcount"].values.tolist()
    dist = data["dist"].values.tolist()
    return cost, p_count, dist


def print_diet(sample, data):
    cost_list = []
    pcount_list = []
    dist_list = []

    for t in range(len(sample)):
        if sample[t] == 1:
            cost_list.append(data.iloc[t]["cost"])
            pcount_list.append(data.iloc[t]["pcount"])
            dist_list.append(data.iloc[t]["dist"])

    fval = (
        (1 / 3) * pow(sum(cost_list) / sum(data["cost"]), 2)
        + (1 / 3) * pow((sum(pcount_list) - sum(data["pcount"])) / sum(data["pcount"]), 2)
        + (1 / 3) * pow((sum(dist_list) - sum(data["dist"])) / sum(data["dist"]), 2)
    )
    return fval


def OrderByImpact(best_solution, df, best_energy):
    impact_values = {}
    for case in range(len(best_solution)):
        if best_solution[case] == 1:
            temp = best_solution.copy()
            temp[case] = 0
            impact_values[case] = print_diet(temp, df) - best_energy
        else:
            temp = best_solution.copy()
            temp[case] = 1
            impact_values[case] = print_diet(temp, df) - best_energy

    impact_values = sorted(impact_values.items(), key=lambda kv: (kv[1], kv[0]))
    impact_list = [case[0] for case in impact_values]
    return impact_list


def OrderByImpactNum(best_solution, df, best_energy):
    num = len(best_solution)
    cost_array = list(df["cost"])
    pcount_array = list(df["pcount"])
    dist_array = list(df["dist"])

    cost_matrix = np.array(cost_array).reshape(-1, 1)
    pcount_matrix = np.array(pcount_array).reshape(-1, 1)
    dist_matrix = np.array(dist_array).reshape(-1, 1)

    matrix = np.array([best_solution] * len(best_solution))
    for i in range(num):
        matrix[i][i] = 1 - matrix[i][i]

    cost_sum = sum(cost_array)
    pcount_sum = sum(pcount_array)
    dist_sum = sum(dist_array)

    cost_obj = matrix.dot(cost_matrix)
    pcount_obj = matrix.dot(pcount_matrix) - pcount_sum
    dist_obj = matrix.dot(dist_matrix) - dist_sum

    obj = (
        (1 / 3) * (cost_obj / cost_sum) ** 2
        + (1 / 3) * (pcount_obj / pcount_sum) ** 2
        + (1 / 3) * (dist_obj / dist_sum) ** 2
        - best_energy
    )

    sorted_indices = np.argsort(obj, axis=0).flatten()
    return sorted_indices


def run_alg(qubo, reps):
    seed = random.randint(1, 9999999)
    algorithm_globals.random_seed = seed

    optimizer = COBYLA(500)
    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None

    qaoa = QAOA(sampler=ideal_sampler, optimizer=optimizer, reps=reps)
    operator, offset = qubo.to_ising()

    begin = time.time()
    qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
    end = time.time()

    exe_time = end - begin
    return qaoa_result, exe_time


def get_initial_fval(length, df):
    initial_values = [random.choice([0, 1]) for _ in range(length)]
    fval = print_diet(initial_values, df)
    best_solution = initial_values
    best_energy = fval
    return best_solution, best_energy


# ============================================================
# BATCHED EXECUTION
# ============================================================

def run_circuit_with_batching(circuit, sampler):
    from collections import Counter

    total_counts = Counter()

    for _ in range(307):
        sampler.options.shots = 200
        result = sampler.run([circuit]).result()
        counts = result.quasi_dists[0].binary_probabilities()
        for k, v in counts.items():
            total_counts[k] += v * 200

    sampler.options.shots = 40
    result = sampler.run([circuit]).result()
    counts = result.quasi_dists[0].binary_probabilities()
    for k, v in counts.items():
        total_counts[k] += v * 40

    return total_counts


# ============================================================
# STEP 1 - TRAIN AND SAVE CIRCUITS
# ============================================================

def save_trained_circuits_and_initial_solutions():
    base_output_dir = os.path.join("..","trained_qaoa_circuits", "igdec_qaoa", RUN_LABEL)
    os.makedirs(base_output_dir, exist_ok=True)

    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None

    df = pd.read_csv(
        f"../datasets/quantum_sota_datasets/{DATASET_FILENAME}.csv",
        dtype={"cost": float, "pcount": int, "dist": int}
    )

    length = len(df)
    cost, pcount, dist = get_data(df)

    for sampling_id in range(1, NUM_EXPERIMENT + 1):
        print(f"\n----- {RUN_LABEL} | RANDOM INITIAL SAMPLING #{sampling_id} -----")

        sampling_dir = os.path.join(base_output_dir, f"sampling_{sampling_id}")
        os.makedirs(sampling_dir, exist_ok=True)

        best_solution, best_energy = get_initial_fval(length, df)
        solution = best_solution.copy()

        with open(os.path.join(sampling_dir, "initial_random_solution.json"), "w") as f:
            json.dump(
                {
                    "initial_solution": best_solution,
                    "initial_energy": best_energy
                },
                f,
                indent=2
            )

        best_itr = 0
        start_impact = time.time()
        impact_order = OrderByImpactNum(best_solution, df, best_energy)
        end_impact = time.time()
        impact_time = end_impact - start_impact

        index_end = PROBLEM_SIZE
        index_begin = 0
        count = 0
        itr_num = 0

        total_qaoa = 0
        total_exe = 0
        total_impact = 0
        execution_times = []
        best_itr_costs = []
        best_itr_pcounts = []
        best_itr_dists = []

        circuits_metadata = []

        while count < NUM_ITERATIONS:
            df_time = 0
            qaoa_time_total = 0
            itr_num += 1
            total_start = time.time()

            if PROBLEM_SIZE > 0.15 * len(df):
                case_list = impact_order[index_begin:index_end]
                qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)

                qaoa = QAOA(
                    sampler=ideal_sampler,
                    optimizer=COBYLA(500),
                    reps=REPS
                )

                operator, offset = qubo.to_ising()

                start_qaoa = time.time()
                qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
                end_qaoa = time.time()
                qaoa_time = end_qaoa - start_qaoa
                qaoa_time_total += qaoa_time

                optimal_params = qaoa_result.optimal_point
                ansatz = qaoa.ansatz
                bound_circuit = ansatz.assign_parameters(optimal_params)

                circuit_filename = f"itr_{itr_num}_subproblem_1.qpy"
                with open(os.path.join(sampling_dir, circuit_filename), "wb") as f:
                    qpy.dump(bound_circuit, f)

                circuits_metadata.append(
                    {
                        "iteration": itr_num,
                        "subproblem_index": 1,
                        "case_list": [int(x) for x in case_list],
                        "qpy_file": circuit_filename
                    }
                )

                eigenstate = qaoa_result.eigenstate
                most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

                if isinstance(most_likely, int):
                    n = qubo.get_num_binary_vars()
                    bitstring = [int(b) for b in format(most_likely, f"0{n}b")[::-1]]
                elif isinstance(most_likely, str):
                    bitstring = [int(b) for b in most_likely[::-1]]
                else:
                    raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

                start_df = time.time()
                for case_index in range(len(case_list)):
                    solution[case_list[case_index]] = bitstring[case_index]
                result_fval = qubo.objective.evaluate(bitstring)
                end_df = time.time()
                df_time += end_df - start_df

            else:
                subproblem_idx = 0
                result_fval = None

                while index_end <= 0.15 * len(df):
                    subproblem_idx += 1
                    case_list = impact_order[index_begin:index_end]
                    qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)

                    qaoa = QAOA(
                        sampler=ideal_sampler,
                        optimizer=COBYLA(500),
                        reps=REPS
                    )

                    operator, offset = qubo.to_ising()

                    start_qaoa = time.time()
                    qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
                    end_qaoa = time.time()
                    qaoa_time = end_qaoa - start_qaoa
                    qaoa_time_total += qaoa_time

                    optimal_params = qaoa_result.optimal_point
                    ansatz = qaoa.ansatz
                    bound_circuit = ansatz.assign_parameters(optimal_params)

                    circuit_filename = f"itr_{itr_num}_subproblem_{subproblem_idx}.qpy"
                    with open(os.path.join(sampling_dir, circuit_filename), "wb") as f:
                        qpy.dump(bound_circuit, f)

                    circuits_metadata.append(
                        {
                            "iteration": itr_num,
                            "subproblem_index": subproblem_idx,
                            "case_list": [int(x) for x in case_list],
                            "qpy_file": circuit_filename
                        }
                    )

                    eigenstate = qaoa_result.eigenstate
                    most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

                    if isinstance(most_likely, int):
                        n = qubo.get_num_binary_vars()
                        bitstring = [int(b) for b in format(most_likely, f"0{n}b")[::-1]]
                    elif isinstance(most_likely, str):
                        bitstring = [int(b) for b in most_likely[::-1]]
                    else:
                        raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

                    start_df = time.time()
                    for case_index in range(len(case_list)):
                        solution[case_list[case_index]] = bitstring[case_index]
                    result_fval = qubo.objective.evaluate(bitstring)
                    index_begin += PROBLEM_SIZE
                    index_end += PROBLEM_SIZE
                    end_df = time.time()
                    df_time += end_df - start_df

            energy = result_fval

            if energy < best_energy:
                best_itr = itr_num
                best_solution = solution.copy()
                best_energy = energy

            total_end = time.time()
            total_itr_time = total_end - total_start - df_time + impact_time

            execution_times.append(impact_time + qaoa_time_total)
            total_qaoa += qaoa_time_total
            total_exe += total_itr_time
            total_impact += impact_time

            best_itr_costs.append(float(df.loc[np.array(best_solution) == 1, "cost"].sum()))
            best_itr_pcounts.append(float(df.loc[np.array(best_solution) == 1, "pcount"].sum()))
            best_itr_dists.append(float(df.loc[np.array(best_solution) == 1, "dist"].sum()))

            start_impact = time.time()
            impact_order = OrderByImpactNum(solution, df, energy)
            end_impact = time.time()
            impact_time = end_impact - start_impact

            print("best:" + str(best_energy))
            count += 1
            index_begin = 0
            index_end = PROBLEM_SIZE

        with open(os.path.join(sampling_dir, "circuits_metadata.json"), "w") as f:
            json.dump(circuits_metadata, f, indent=2)

        with open(os.path.join(sampling_dir, "training_summary.json"), "w") as f:
            json.dump(
                {
                    "best_itr": best_itr,
                    "best_fval": best_energy,
                    "best_solution": best_solution,
                    "total_qaoa": total_qaoa,
                    "total_impact": total_impact,
                    "total_exe": total_exe,
                    "execution_times": execution_times,
                    "final_test_suite_costs": best_itr_costs,
                    "final_suite_pcounts": best_itr_pcounts,
                    "final_suite_dists": best_itr_dists
                },
                f,
                indent=2
            )

        print(f"Saved trained circuits and metadata in: {sampling_dir}")


# ============================================================
# STEP 2 - HARDWARE-LIKE EXECUTION
# ============================================================

def run_hardware_like_from_saved_circuits():
    results_dir = os.path.join("results", "igdec_qaoa", "piastq")
    os.makedirs(results_dir, exist_ok=True)

    provider = AQTProvider("ACCESS_TOKEN")
    backend = provider.get_backend("offline_simulator_no_noise")

    sampling_sampler = AQTSampler(backend)

    sampling_sampler.set_transpile_options(optimization_level=3)

    df = pd.read_csv(
        f"../datasets/quantum_sota_datasets/{DATASET_FILENAME}.csv",
        dtype={"cost": float, "pcount": int, "dist": int}
    )

    cost, pcount, dist = get_data(df)
    base_input_dir = os.path.join("..","trained_qaoa_circuits", "igdec_qaoa", RUN_LABEL)

    program_results = {}

    for sampling_id in range(1, NUM_EXPERIMENT + 1):
        print(f"\n----- HARDWARE-LIKE {RUN_LABEL} | SAMPLING #{sampling_id} -----")

        sampling_dir = os.path.join(base_input_dir, f"sampling_{sampling_id}")

        with open(os.path.join(sampling_dir, "initial_random_solution.json"), "r") as f:
            init_data = json.load(f)

        with open(os.path.join(sampling_dir, "circuits_metadata.json"), "r") as f:
            circuits_metadata = json.load(f)

        solution = init_data["initial_solution"].copy()
        best_solution = init_data["initial_solution"].copy()
        best_energy = init_data["initial_energy"]

        metadata_index = 0
        qpu_run_times = []
        impact_times = []
        execution_times = []

        count = 0
        itr_num = 0

        start_impact = time.time()
        impact_order = OrderByImpactNum(solution, df, best_energy)
        end_impact = time.time()
        impact_time = end_impact - start_impact

        index_begin = 0
        index_end = PROBLEM_SIZE

        while count < NUM_ITERATIONS:
            qaoa_time_total = 0
            itr_num += 1

            if PROBLEM_SIZE > 0.15 * len(df):
                case_list = impact_order[index_begin:index_end]

                meta = circuits_metadata[metadata_index]
                metadata_index += 1

                circuit_path = os.path.join(sampling_dir, meta["qpy_file"])
                with open(circuit_path, "rb") as f:
                    circuits = qpy.load(f)
                circuit = circuits[0]

                start_qpu = time.time()
                counts = run_circuit_with_batching(circuit, sampling_sampler)
                end_qpu = time.time()

                qpu_time = end_qpu - start_qpu
                qaoa_time_total += qpu_time
                qpu_run_times.append(qpu_time * 1000)

                most_likely = max(counts.items(), key=lambda x: x[1])[0]
                bitstring = [int(b) for b in most_likely[::-1]]

                for case_index in range(len(case_list)):
                    solution[case_list[case_index]] = bitstring[case_index]

                qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
                result_fval = qubo.objective.evaluate(bitstring)

            else:
                result_fval = None

                while index_end <= 0.15 * len(df):
                    case_list = impact_order[index_begin:index_end]

                    meta = circuits_metadata[metadata_index]
                    metadata_index += 1

                    circuit_path = os.path.join(sampling_dir, meta["qpy_file"])
                    with open(circuit_path, "rb") as f:
                        circuits = qpy.load(f)
                    circuit = circuits[0]

                    start_qpu = time.time()
                    counts = run_circuit_with_batching(circuit, sampling_sampler)
                    end_qpu = time.time()

                    qpu_time = end_qpu - start_qpu
                    qaoa_time_total += qpu_time
                    qpu_run_times.append(qpu_time * 1000)

                    most_likely = max(counts.items(), key=lambda x: x[1])[0]
                    bitstring = [int(b) for b in most_likely[::-1]]

                    for case_index in range(len(case_list)):
                        solution[case_list[case_index]] = bitstring[case_index]

                    qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
                    result_fval = qubo.objective.evaluate(bitstring)

                    index_begin += PROBLEM_SIZE
                    index_end += PROBLEM_SIZE

            energy = result_fval

            if energy < best_energy:
                best_solution = solution.copy()
                best_energy = energy

            start_impact = time.time()
            impact_order = OrderByImpactNum(solution, df, energy)
            end_impact = time.time()
            impact_time = end_impact - start_impact

            impact_times.append(impact_time * 1000)
            execution_times.append((impact_time + qaoa_time_total) * 1000)

            print("best:" + str(best_energy))
            count += 1
            index_begin = 0
            index_end = PROBLEM_SIZE

        final_test_suite_cost = float(df.loc[np.array(best_solution) == 1, "cost"].sum())
        final_suite_pcount = float(df.loc[np.array(best_solution) == 1, "pcount"].sum())
        final_suite_dist = float(df.loc[np.array(best_solution) == 1, "dist"].sum())

        program_results[f"sampling_{sampling_id}"] = {
            "initial_solution": init_data["initial_solution"],
            "initial_energy": init_data["initial_energy"],
            "best_solution": best_solution,
            "best_energy": best_energy,
            "final_test_suite_cost": final_test_suite_cost,
            "final_suite_pcount": final_suite_pcount,
            "final_suite_dist": final_suite_dist,
            "all_qpu_run_times(ms)": qpu_run_times,
            "mean_qpu_run_time(ms)": statistics.mean(qpu_run_times) if len(qpu_run_times) > 0 else 0,
            "stdev_qpu_run_time(ms)": statistics.stdev(qpu_run_times) if len(qpu_run_times) > 1 else 0,
            "all_impact_times(ms)": impact_times,
            "execution_times(ms)": execution_times
        }

    output_file = os.path.join(results_dir, f"{RUN_LABEL}.json")
    with open(output_file, "w") as f:
        json.dump(program_results, f, indent=2)

    print(f"Saved hardware-like results to: {output_file}")


if __name__ == "__main__":
    run_hardware_like_from_saved_circuits()