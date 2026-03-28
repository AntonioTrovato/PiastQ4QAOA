import statistics

from qiskit_aer import Aer
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit.result import QuasiDistribution
from typing import List, Union
from qiskit.quantum_info import Pauli, Statevector
from qiskit.result import QuasiDistribution
import numpy as np
from docplex.mp.model import Model
from qiskit_algorithms import QAOA
from qiskit_aer.primitives import Sampler as AerSampler
import json
from qiskit import qpy
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.converters import QuadraticProgramToQubo
import pandas as pd
import random
import sys

# from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, GradientDescent
from qiskit.primitives import Sampler, BackendSampler
import argparse
import time
import matplotlib.pyplot as plt
import random
import pandas as pd
import os

num_experiments = 30

class TestCaseOptimization(OptimizationApplication):
    """Optimization application for the "knapsack problem" [1].

    References:
        [1]: "Knapsack problem",
        https://en.wikipedia.org/wiki/Knapsack_problem
    """

    def __init__(self, times: List[float], frs: List[float], w1: float, w2: float, w3: float, sample: List[int],
                 solution: List[int]) -> None:
        """
        Args:
            values: A list of the values of items
            weights: A list of the weights of items
            max_weight: The maximum weight capacity
        """
        self._times = times
        self._frs = frs
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3
        self._sample = sample
        self._solution = solution

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a knapsack problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the knapsack problem instance.
        """
        mdl = Model(name="Knapsack")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in self._sample}

        obj_time = 0
        obj_rate = 0
        obj_num = 0

        #         dic_clamp = {}
        for i in range(len(self._solution)):
            if i in self._sample:
                obj_time += self._times[i] * x[i]
                obj_rate += self._frs[i] * x[i]
                obj_num += x[i]
            else:
                obj_time += self._times[i] * self._solution[i]
                obj_rate += self._frs[i] * self._solution[i]
                obj_num += self._solution[i]

        time_sum = sum(self._times)
        rate_sum = sum(self._frs)

        obj_time = pow(obj_time / time_sum, 2)
        obj_rate = pow((obj_rate - rate_sum) / rate_sum, 2)
        obj_num = pow(obj_num / len(self._times), 2)
        #         print("time:",obj_time)
        #         print("rate:",obj_rate)
        #         print("num:",obj_num)

        #         obj_time = sum(self._times[i] * x[i] for i in x)
        #         obj_time = pow(obj_time/time_sum, 2)

        #         if rate_sum == 0:
        #             obj_fr = 0
        #             obj_fr = 0
        #         else:
        #             obj_fr = sum(self._frs[i] * x[i] for i in x)
        #             obj_fr = pow((obj_fr-rate_sum)/rate_sum, 2)

        #         obj_num = pow(sum(x[i] for i in x)/len(self._times), 2)

        obj = self._w1 * obj_time + self._w2 * obj_rate + self._w3 * obj_num

        mdl.minimize(obj)

        op = from_docplex_mp(mdl)

        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as item indices

        Args:
            result : The calculated result of the problem

        Returns:
            A list of items whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, value in enumerate(x) if value]

def create_qubo(times, frs, w1, w2, w3, sample, solution):
    testcase = TestCaseOptimization(times, frs, w1, w2, w3, sample, solution)
    prob = testcase.to_quadratic_program()
    #probQubo = QuadraticProgramToQubo() #parameter: cofficient for constraint
    #qubo = probQubo.convert(prob)
    return prob, testcase

def get_data(data):
    times = data["time"].values.tolist()
    frs = data["rate"].values.tolist()
    return times, frs

def print_diet(sample,data):
    count = 0
    total_time = 0
    total_rate = 0
    time_list = []
    rate_list = []
    for t in range(len(sample)):
        if sample[t] == 1:
            total_time += data.iloc[t]['time']
            total_rate += data.iloc[t]['rate']
            time_list.append(data.iloc[t]['time'])
            rate_list.append(data.iloc[t]['rate'])
            # print(t[1:]+'. ',end=' ')
            # print('time: '+str(foods[t]['time']), end=', ')
            # print('rate: '+str(foods[t]['rate']), end='\n')
            count += 1
    fval = (1 / 3) * pow(sum(time_list) / sum(data['time']), 2) + (1 / 3) * pow((sum(rate_list) - sum(data["rate"]) + 1e-20) / (sum(data["rate"])+1e-20), 2) + (1 / 3) * pow(count / len(data), 2)
    # print("Total time: " + str(total_time))
    # print("Total rate: " + str(total_rate))
#     print("Fval value:" + str(fval))
#     print("Number: "+str(count))
    return fval

def OrderByImpact(best_solution, df, best_energy):
    impact_values = {}
    for case in range(len(best_solution)):
        if best_solution[case] == 1:
            temp = best_solution.copy()
            temp[case] = 0
            impact_values[case] = 1
            impact_values[case] = print_diet(temp, df) - best_energy
        elif best_solution[case] == 0:
            temp = best_solution.copy()
            temp[case]=1
            impact_values[case] = print_diet(temp, df) - best_energy
    print(impact_values)
    impact_values = sorted(impact_values.items(), key = lambda kv:(kv[1], kv[0]))
    impact_list = []
    for case in impact_values:
        impact_list.append(case[0])
    return impact_list

def OrderByImpactNum(best_solution, df, best_energy):
    num = len(best_solution)
    time_array = list(df["time"])
    rate_array = list(df["rate"])
    time_matrix = np.array(time_array).reshape(-1, 1)
    rate_matrix = np.array(rate_array).reshape(-1, 1)
    num_matrix = np.full((num,1), 1)
    matrix = np.array([best_solution] * len(best_solution))
    for i in range(num):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
        elif matrix[i][i] == 1:
            matrix[i][i] = 0
    time_sum = sum(time_array)
    rate_sum = sum(rate_array)
    # time_sum_con = np.full((len(best_solution), 1), time_sum)
    # rate_sum_con = np.full((len(best_solution), 1), rate_sum)
    time_obj = matrix.dot(time_matrix)
    rate_obj = matrix.dot(rate_matrix) - rate_sum + 1e-20
    num_obj = matrix.dot(num_matrix)
    obj = (1/3)*(time_obj/time_sum)**2 + (1/3)*(rate_obj/(rate_sum+1e-20))**2 + (1/3)*((num_obj)/len(best_solution))**2 - best_energy
    # Get the sorted indices
    sorted_indices = np.argsort(obj, axis=0)

    # Convert the sorted indices to a flattened array
    sorted_indices = sorted_indices.flatten()

    return sorted_indices


def run_alg(qubo, reps):
    seed = random.randint(1, 9999999)
    algorithm_globals.random_seed = seed
    optimizer = COBYLA(500)
    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None
    # backend.set_options(device='GPU')
    qaoa = QAOA(sampler=ideal_sampler, optimizer=optimizer, reps=reps)
    operator, offset = qubo.to_ising()
    begin = time.time()
    qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
    end = time.time()
    exe_time = end-begin
    return qaoa_result, exe_time

def print_result(result, testcase):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = testcase.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))

def plot(fval_list, reps, file_name, problem_size):
    plt.plot(fval_list)
    plt.ylabel('fval')
    plt.savefig("results/igdec_qaoa/ideal/qaoa_"+str(reps)+"/" + file_name + "/size_" + str(problem_size) + "/" + str(num_experiments)+"/fval_trend.png")

def scatter_merge(solution, data):
    time = []
    rate = []
    for t in range(len(solution)):
        if solution[t] == 1.0:
            time.append(data.iloc[t]['time'])
            rate.append(data.iloc[t]['rate'])
    plt.scatter(data["time"], data["rate"], c='red')
    plt.scatter(time, rate)
    plt.show()

def get_initial_fval(length,df):
    initial_values = [random.choice([0, 1]) for _ in range(length)]
    fval = print_diet(initial_values, df)
    best_solution=initial_values
    best_energy=fval
    return best_solution, best_energy


def run_circuit_with_batching(circuit, sampler):
    """
    Simulate hardware constraint: max 200 shots per batch.
    Total target shots = 2048 * 30 = 61440
    => 307 batches x 200 shots + 1 batch x 40 shots
    """
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


def save_trained_circuits_and_initial_solutions():
    """
    Train IGDec-QAOA in ideal noiseless simulation and save:
    - one folder per initial random sampling
    - initial random solution
    - one trained circuit per solved subproblem
    - metadata to reconstruct the exact execution order
    """
    base_output_dir = "trained_qaoa_circuits/igdec_qaoa"
    os.makedirs(base_output_dir, exist_ok=True)

    num_experiment = 30
    reps = 1
    problem_size = 7

    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None

    for file_name in ["gsdtsr", "iofrol", "paintcontrol"]:
        print(f"\n========== PROGRAM: {file_name} ==========")

        df = pd.read_csv(
            "../datasets/quantum_sota_datasets/" + file_name + ".csv",
            dtype={"time": float, "rate": float}
        )

        length = len(df)
        times, frs = get_data(df)

        program_dir = os.path.join(base_output_dir, file_name)
        os.makedirs(program_dir, exist_ok=True)

        for sampling_id in range(1, num_experiment + 1):
            print(f"\n----- RANDOM INITIAL SAMPLING #{sampling_id} -----")

            sampling_dir = os.path.join(program_dir, f"sampling_{sampling_id}")
            os.makedirs(sampling_dir, exist_ok=True)

            # initial random solution
            best_solution, best_energy = get_initial_fval(length,df)
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

            index_end = problem_size
            index_begin = 0
            count = 0
            itr_num = 0

            total_qaoa = 0
            total_exe = 0
            total_impact = 0
            execution_times = []
            best_itr_times = []
            best_itr_rates = []

            circuits_metadata = []

            while count < num_experiment:
                df_time = 0
                qaoa_time_total = 0
                exe_count = 0
                itr_num += 1
                total_start = time.time()

                if problem_size > 0.15 * len(df):
                    exe_count += 1
                    case_list = impact_order[index_begin:index_end]
                    qubo, testcase = create_qubo(times, frs, 1 / 3, 1 / 3, 1 / 3, case_list, solution)

                    qaoa = QAOA(
                        sampler=ideal_sampler,
                        optimizer=COBYLA(500),
                        reps=reps
                    )

                    operator, offset = qubo.to_ising()

                    start_qaoa = time.time()
                    qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
                    end_qaoa = time.time()
                    qaoa_time = end_qaoa - start_qaoa

                    qaoa_time_total += qaoa_time

                    # save trained circuit
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
                        bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
                    elif isinstance(most_likely, str):
                        bitstring = [int(b) for b in most_likely[::-1]]
                    else:
                        raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

                    start_df = time.time()
                    origin_solution = []
                    for case in case_list:
                        origin_solution.append(solution[case])

                    for case_index in range(len(case_list)):
                        solution[case_list[case_index]] = bitstring[case_index]

                    result_fval = qubo.objective.evaluate(bitstring)
                    end_df = time.time()
                    df_time += end_df - start_df

                else:
                    subproblem_idx = 0
                    result_fval = None

                    while index_end <= 0.15 * len(df):
                        exe_count += 1
                        subproblem_idx += 1

                        case_list = impact_order[index_begin:index_end]
                        qubo, testcase = create_qubo(times, frs, 1 / 3, 1 / 3, 1 / 3, case_list, solution)

                        qaoa = QAOA(
                            sampler=ideal_sampler,
                            optimizer=COBYLA(500),
                            reps=reps
                        )

                        operator, offset = qubo.to_ising()

                        start_qaoa = time.time()
                        qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
                        end_qaoa = time.time()
                        qaoa_time = end_qaoa - start_qaoa

                        qaoa_time_total += qaoa_time

                        # save trained circuit
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
                            bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
                        elif isinstance(most_likely, str):
                            bitstring = [int(b) for b in most_likely[::-1]]
                        else:
                            raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

                        start_df = time.time()
                        origin_solution = []
                        for case in case_list:
                            origin_solution.append(solution[case])

                        for case_index in range(len(case_list)):
                            solution[case_list[case_index]] = bitstring[case_index]

                        result_fval = qubo.objective.evaluate(bitstring)

                        index_begin += problem_size
                        index_end += problem_size
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

                best_itr_times.append(df.loc[np.array(best_solution) == 1, "time"].sum())
                best_itr_rates.append(df.loc[np.array(best_solution) == 1, "rate"].sum())

                start_impact = time.time()
                impact_order = OrderByImpactNum(solution, df, energy)
                end_impact = time.time()
                impact_time = end_impact - start_impact

                print("best:" + str(best_energy))
                count += 1
                index_begin = 0
                index_end = problem_size

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
                        "final_test_suite_costs": [float(x) for x in best_itr_times],
                        "final_failure_rates": [float(x) for x in best_itr_rates]
                    },
                    f,
                    indent=2
                )

            print(f"Saved trained circuits and metadata in: {sampling_dir}")


def run_hardware_like_from_saved_circuits():
    """
    Re-run IGDec-QAOA by loading:
    - the saved initial random solution
    - the saved trained circuit for each solved subproblem
    and executing each circuit with 200-shot batching.
    """
    results_dir = "results/igdec_qaoa/piastq"
    os.makedirs(results_dir, exist_ok=True)

    num_experiment = 30
    problem_size = 7

    provider = AQTProvider("ACCESS_TOKEN")
    backend = provider.get_backend("offline_simulator_no_noise")

    sampling_sampler = AQTSampler(backend)

    sampling_sampler.set_transpile_options(optimization_level=3)

    for file_name in ["gsdtsr", "iofrol", "paintcontrol"]:
        print(f"\n========== HARDWARE-LIKE PROGRAM: {file_name} ==========")

        df = pd.read_csv(
            "../datasets/quantum_sota_datasets/" + file_name + ".csv",
            dtype={"time": float, "rate": float}
        )

        times, frs = get_data(df)
        program_results = {}

        for sampling_id in range(1, num_experiment + 1):
            print(f"\n----- HARDWARE-LIKE SAMPLING #{sampling_id} -----")

            sampling_dir = os.path.join(
                "..",
                "trained_qaoa_circuits",
                "igdec_qaoa",
                file_name,
                f"sampling_{sampling_id}"
            )

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
            index_end = problem_size

            while count < num_experiment:
                qaoa_time_total = 0
                itr_num += 1

                if problem_size > 0.15 * len(df):
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

                    qubo, testcase = create_qubo(times, frs, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
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

                        qubo, testcase = create_qubo(times, frs, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
                        result_fval = qubo.objective.evaluate(bitstring)

                        index_begin += problem_size
                        index_end += problem_size

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
                index_end = problem_size

            final_test_suite_cost = float(df.loc[np.array(best_solution) == 1, "time"].sum())
            final_failure_rate = float(df.loc[np.array(best_solution) == 1, "rate"].sum())

            program_results[f"sampling_{sampling_id}"] = {
                "initial_solution": init_data["initial_solution"],
                "initial_energy": init_data["initial_energy"],
                "best_solution": best_solution,
                "best_energy": best_energy,
                "final_test_suite_cost": final_test_suite_cost,
                "final_failure_rate": final_failure_rate,
                "all_qpu_run_times(ms)": qpu_run_times,
                "mean_qpu_run_time(ms)": statistics.mean(qpu_run_times) if len(qpu_run_times) > 0 else 0,
                "stdev_qpu_run_time(ms)": statistics.stdev(qpu_run_times) if len(qpu_run_times) > 1 else 0,
                "all_impact_times(ms)": impact_times,
                "execution_times(ms)": execution_times
            }

        output_file = os.path.join(results_dir, f"{file_name}.json")
        with open(output_file, "w") as f:
            json.dump(program_results, f, indent=2)

        print(f"Saved hardware-like results to: {output_file}")


if __name__ == '__main__':
    run_hardware_like_from_saved_circuits()
