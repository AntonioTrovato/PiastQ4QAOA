#this cell contains all the imports needed by the pipeline
#to run it on the browser: jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from piastq_execution.raw_counts import run_circuit_with_batching_recorded, RawCountsWriter
from piastq_execution.backend_config import get_backend
from piastq_execution.mitigation import (
    build_trex_twirled_circuit,
    generate_random_twirl_mask,
    load_trex_twirl_instances,
)

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import statistics
import ast
import csv

from scipy.stats import mannwhitneyu, shapiro

from qiskit import qpy
from qiskit_optimization import QuadraticProgram
from qiskit_aqt_provider.primitives import AQTSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_optimization.converters import QuadraticProgramToQubo
from matplotlib import cm

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

bootqa_programs = ["gsdtsr", "paintcontrol", "iofrol", "elevator", "elevator2"]
bootqa_programs_rep_values = {"gsdtsr": 1, "paintcontrol": 1, "iofrol": 1, "elevator": 1, "elevator2": 1}
# Per-dataset QUBO alpha (elevator2: alpha, beta, gamma) used when (re)training
# circuits -- matches SelectQAOA/MOQ-Pipeline.ipynb's single-objective
# training cell, i.e. what the already-ported circuits were trained with.
bootqa_alphas = {"gsdtsr": 0, "paintcontrol": 0.45, "iofrol": 0.50, "elevator": 0.20, "elevator2": (0.1, 0.9, 0.9)}
experiments = 10

def get_data(data_name):
    """Read the datasets"""
    if data_name == "elevator":
        data = pd.read_csv("../../datasets/quantum_sota_datasets/elevator.csv", dtype={"cost": int, "input_div": float})
    elif data_name == "elevator2":
        data = pd.read_csv("../../datasets/quantum_sota_datasets/elevator.csv", dtype={"cost": int, "pcount": int, "dist": int})
    else:
        data = pd.read_csv("../../datasets/quantum_sota_datasets/" + data_name + ".csv", dtype={"time": float, "rate": float})
        data = data[data['rate'] > 0]
    return data

if not callable(list):
    del list

from collections import defaultdict

bootqa_clusters = dict()

for bootqa_program in bootqa_programs:
    data = get_data(bootqa_program)

    # Total suite metrics
    if bootqa_program == "elevator" or bootqa_program == "elevator2":
        test_cases_costs = data["cost"].tolist()
    else:
        test_cases_costs = data["time"].tolist()

    if bootqa_program == "elevator":
        test_cases_effectiveness = data["input_div"].tolist()
        # print(f"Tot suite cost: {sum(test_cases_costs)}")
        # print(f"Tot suite input div: {sum(test_cases_effectiveness)}")
    elif bootqa_program == "elevator2":
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()
        # print(f"Tot suite cost: {sum(test_cases_costs)}")
        # print(f"Tot suite pcount: {sum(test_cases_pcount)}")
        # print(f"Tot suite dist: {sum(test_cases_dist)}")
    else:
        test_cases_effectiveness = data["rate"].tolist()
        # print(f"Tot suite cost: {sum(test_cases_costs)}")
        # print(f"Tot suite rate: {sum(test_cases_effectiveness)}")

    # Normalize data
    if bootqa_program != "elevator2":
        cluster_data = np.column_stack((test_cases_costs, test_cases_effectiveness))
    else:
        cluster_data = np.column_stack((test_cases_costs, test_cases_pcount, test_cases_dist))

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cluster_data)

    if bootqa_program == "elevator" or bootqa_program == "elevator2":
        num_clusters = 800
    if bootqa_program == "gsdtsr":
        num_clusters = 60
    if bootqa_program == "iofrol":
        num_clusters = 324
    if bootqa_program == "paintcontrol":
        num_clusters = 16

    max_cluster_dim = 7

    # Step 2: Perform K-Means Clustering
    start = time.time()
    linkage_matrix = linkage(normalized_data, method='ward')
    clusters = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    # Organize test cases by cluster
    clustered_data = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        clustered_data[cluster_id].append(idx)

    # Process clusters to ensure none exceed max_cluster_dim
    new_cluster_id = max(clustered_data.keys()) + 1  # Start new IDs after existing ones
    to_add = []  # Collect new smaller clusters

    for cluster_id, elements in list(clustered_data.items()):  # Avoid modifying dict during iteration
        if len(elements) > max_cluster_dim:
            num_splits = -(-len(elements) // max_cluster_dim)  # Ceiling division to get the required number of splits
            split_size = -(-len(elements) // num_splits)  # Recalculate to distribute elements evenly

            # Split while keeping sizes balanced
            parts = [elements[i:i + split_size] for i in range(0, len(elements), split_size)]

            # Ensure all new clusters are within max_cluster_dim
            for part in parts:
                if len(part) > max_cluster_dim:
                    raise ValueError(
                        f"A split cluster still exceeds max_cluster_dim ({len(part)} > {max_cluster_dim})!")

            # Add new parts to the new clusters
            to_add.extend(parts)

            # Remove original large cluster
            del clustered_data[cluster_id]

    # Assign new IDs to split parts
    for part in to_add:
        if part:  # Only add if the part is non-empty
            clustered_data[new_cluster_id] = part
            new_cluster_id += 1
    end = time.time()
    # print("SelectQAOA Decomposition Time(ms): " + str((end-start)*1000))

    bootqa_clusters[bootqa_program] = clustered_data

    # Step 3: Calculate the metrics for each refined cluster
    cluster_metrics = {}
    for cluster_id, members in clustered_data.items():
        tot_cluster_costs = sum(test_cases_costs[i] for i in members)
        if bootqa_program != "elevator2":
            tot_cluster_effectiveness = sum(test_cases_effectiveness[i] for i in members)
        else:
            tot_cluster_pcount = sum(test_cases_pcount[i] for i in members)
            tot_cluster_dist = sum(test_cases_dist[i] for i in members)
        if bootqa_program != "elevator2":
            cluster_metrics[cluster_id] = {
                "tot_cluster_cost": tot_cluster_costs,
                "tot_cluster_rates": tot_cluster_effectiveness
            }
        else:
            cluster_metrics[cluster_id] = {
                "tot_cluster_cost": tot_cluster_costs,
                "tot_cluster_pcount": tot_cluster_pcount,
                "tot_cluster_dist": tot_cluster_dist
            }
        # print(f"Cluster {cluster_id + 1} metrics:")
        # print(f"Test Cases: {members}")
        # print(f" - Num. Test Cases: {len(members):.2f}")
        # print(f" - Execution Cost: {tot_cluster_costs:.2f}")
        if bootqa_program != "elevator2":
            print(f" - Failure Rate: {tot_cluster_effectiveness}")
        else:
            print(f" - PCount: {tot_cluster_pcount}")
            print(f" - Dist: {tot_cluster_dist}")

    print("===========================================================================")

    for cluster_id in clustered_data.keys():
        if len(clustered_data[cluster_id]) > max_cluster_dim:
            print("Program: " + bootqa_program)
            print("Test cases of cluster " + str(cluster_id) + ": " + str(len(clustered_data[cluster_id])))

    # Plotting the clusters in 3D space
    fig = plt.figure(figsize=(10, 7))
    if bootqa_program != "elevator2":
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    # Extracting data for plotting
    exec_costs = np.array(test_cases_costs)
    if bootqa_program != "elevator2":
        effectiveness = np.array(test_cases_effectiveness)
    else:
        pcounts = np.array(test_cases_pcount)
        dists = np.array(test_cases_dist)

    # Plot each refined cluster with a different color
    colors = plt.cm.get_cmap("tab10", len(clustered_data))  # A colormap with as many colors as clusters
    for cluster_id, members in clustered_data.items():
        if bootqa_program != "elevator2":
            ax.scatter(
                exec_costs[members],
                effectiveness[members],
                color=colors(cluster_id % 10),
                label=f"Cluster {cluster_id + 1}"
            )
        else:
            ax.scatter(
                exec_costs[members],
                pcounts[members],
                dists[members],
                color=colors(cluster_id % 10),
                label=f"Cluster {cluster_id + 1}"
            )

    # Label the axes
    ax.set_xlabel("Execution Cost")
    if bootqa_program != "elevator2":
        ax.set_ylabel("Effectiveness")
    else:
        ax.set_ylabel("Passengers Count")
        ax.set_zlabel("Travel Distance")
    ax.legend()
    ax.set_title("Test Case Clustering Visualization for: " + bootqa_program)

    # Display the plot
    # plt.show()


def make_linear_terms_bootqa(cluster_test_cases, test_cases_costs, test_cases_rates, alpha):
    """Making the linear terms of the QUBO"""
    max_cost = max(test_cases_costs)

    estimated_costs = []

    # linear coefficients, that are the diagonal of the matrix encoding the QUBO
    for test_case in cluster_test_cases:
        estimated_costs.append(
            (alpha * ((test_cases_costs[test_case]) / max_cost)) - ((1 - alpha) * test_cases_rates[test_case]))

    return np.array(estimated_costs)


def make_linear_terms_bootqa2(cluster_test_cases, test_cases_costs, pcount, dist, alpha, beta, gamma):
    """Making the linear terms of the QUBO for the elevator2 problem"""
    max_cost = max(test_cases_costs)
    max_pcount = max(pcount)
    max_dist = max(dist)

    estimated_costs = []

    # linear coefficients, that are the diagonal of the matrix encoding the QUBO
    for test_case in cluster_test_cases:
        estimated_costs.append(
            ((alpha) * ((test_cases_costs[test_case]) / max_cost)) - ((beta) * ((pcount[test_case]) / max_pcount)) - (
                        (gamma) * ((dist[test_case]) / max_dist)))

    return np.array(estimated_costs)


def create_linear_qubo(linear_terms):
    """This function is the one that has to encode the QUBO problem that QAOA will have to solve. The QUBO problem specifies the optimization to solve and a quadratic binary unconstrained problem"""
    qubo = QuadraticProgram()

    for i in range(0, len(linear_terms)):
        qubo.binary_var('x%s' % (i))

    qubo.minimize(linear=linear_terms)

    return qubo


def bootstrap_confidence_interval(data, num_samples, confidence_alpha=0.95):
    """This function determines the statistical range within we would expect the mean value of execution times to fall; it relies on the bootstrapping strategy, which allows the calculation of the confidence interval by repeated sampling (with replacement) from the existing data to obtain an estimate of the confidence interval."""
    sample_means = []
    for _ in range(num_samples):
        bootstrap_sample = [random.choice(data) for _ in range(len(data))]
        sample_mean = np.mean(bootstrap_sample)
        sample_means.append(sample_mean)

    lower_percentile = (1 - confidence_alpha) / 2 * 100
    upper_percentile = (confidence_alpha + (1 - confidence_alpha) / 2) * 100
    lower_bound = np.percentile(sample_means, lower_percentile)
    upper_bound = np.percentile(sample_means, upper_percentile)

    return lower_bound, upper_bound


# Only rep_1 is used by this replication package (matching the circuits
# already ported and the execution loop below); add more values here (e.g.
# 2, 4, 8, 16, matching the original SelectQAOA study) if you need them too.
TRAINING_REPS = [1]


def save_trained_circuits():
    """
    Train QAOA-TCS single-objective circuits in ideal noiseless simulation
    and save them under trained_qaoa_circuits/qaoa_tcs/<dataset>/rep_<reps>/.

    NOT executed by default (invoke via `python single_obj.py train`) and
    NOT run in an authoring/review session -- run this on the machine that
    will actually spend the compute. Ported from SelectQAOA/
    MOQ-Pipeline.ipynb's single-objective circuit-training cell; reuses this
    file's already-computed `bootqa_clusters` so cluster indices/ordering
    exactly match what run_hardware_execution() expects to load.
    """
    base_output_dir = os.path.join("..", "..", "trained_qaoa_circuits", "qaoa_tcs")
    os.makedirs(base_output_dir, exist_ok=True)

    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None

    for bootqa_program in bootqa_programs:
        print(f"\n=== PROGRAM: {bootqa_program} ===")

        program_dir = os.path.join(base_output_dir, bootqa_program)
        os.makedirs(program_dir, exist_ok=True)

        data = get_data(bootqa_program)

        if bootqa_program in ["elevator", "elevator2"]:
            test_cases_costs = data["cost"].tolist()
        else:
            test_cases_costs = data["time"].tolist()

        if bootqa_program == "elevator":
            test_cases_effectiveness = data["input_div"].tolist()
        elif bootqa_program == "elevator2":
            test_cases_pcount = data["pcount"].tolist()
            test_cases_dist = data["dist"].tolist()
        else:
            test_cases_effectiveness = data["rate"].tolist()

        for reps in TRAINING_REPS:
            print(f"\n--- REPS: {reps} ---")

            reps_dir = os.path.join(program_dir, f"rep_{reps}")
            os.makedirs(reps_dir, exist_ok=True)

            qaoa = QAOA(
                sampler=ideal_sampler,
                optimizer=COBYLA(maxiter=500),
                reps=reps
            )

            cluster_items = list(bootqa_clusters[bootqa_program].items())

            for cluster_idx, (cluster_id, cluster_test_cases) in enumerate(cluster_items):
                print(f"Training cluster {cluster_idx} (cluster_id={cluster_id})")

                if bootqa_program != "elevator2":
                    linear_terms = make_linear_terms_bootqa(
                        cluster_test_cases=cluster_test_cases,
                        test_cases_costs=test_cases_costs,
                        test_cases_rates=test_cases_effectiveness,
                        alpha=bootqa_alphas[bootqa_program]
                    )
                else:
                    linear_terms = make_linear_terms_bootqa2(
                        cluster_test_cases=cluster_test_cases,
                        test_cases_costs=test_cases_costs,
                        pcount=test_cases_pcount,
                        dist=test_cases_dist,
                        alpha=bootqa_alphas[bootqa_program][0],
                        beta=bootqa_alphas[bootqa_program][1],
                        gamma=bootqa_alphas[bootqa_program][2]
                    )

                qubo = create_linear_qubo(linear_terms)
                operator, offset = qubo.to_ising()

                # TRAINING
                result = qaoa.compute_minimum_eigenvalue(operator)

                # OPTIMAL CIRCUIT
                optimal_params = result.optimal_point
                ansatz = qaoa.ansatz
                bound_circuit = ansatz.assign_parameters(optimal_params)

                # SAVE
                filename = os.path.join(
                    reps_dir,
                    f"{bootqa_program}_rep{reps}_cluster{cluster_idx}.qpy"
                )

                with open(filename, "wb") as f:
                    qpy.dump(bound_circuit, f)

                print(f"Saved: {filename}")


def run_hardware_execution():
    """
    Re-run QAOA-TCS single-objective by loading the trained circuits and
    executing each on the configured backend.
    """
    backend = get_backend()

    sampling_sampler = AQTSampler(backend)

    sampling_sampler.set_transpile_options(optimization_level=3)

    base_results_dir = os.path.join("..", "..", "results", "qaoa_tcs")
    os.makedirs(base_results_dir, exist_ok=True)

    num_piast_experiments = 10

    # TREx: number of independently-twirled hardware passes per circuit, read
    # from configs/execution_plan.yaml so it matches what budget_planner.py
    # priced (see piastq_execution.mitigation.load_trex_twirl_instances).
    trex_twirl_instances = load_trex_twirl_instances()
    trex_rng = random.Random()

    for bootqa_program in bootqa_programs:
        program_results_dir = os.path.join(base_results_dir, bootqa_program)
        os.makedirs(program_results_dir, exist_ok=True)

        data = get_data(bootqa_program)

        if bootqa_program in ["elevator", "elevator2"]:
            test_cases_costs = data["cost"].tolist()
        else:
            test_cases_costs = data["time"].tolist()

        if bootqa_program == "elevator":
            test_cases_effectiveness = data["input_div"].tolist()
        elif bootqa_program == "elevator2":
            test_cases_pcount = data["pcount"].tolist()
            test_cases_dist = data["dist"].tolist()
        else:
            test_cases_effectiveness = data["rate"].tolist()

        for reps in [1]:
            print(f"\n=== PROGRAM: {bootqa_program}, REPS: {reps} ===")

            file_path = os.path.join(
                program_results_dir,
                f"{bootqa_program}-rep-{reps}.json"
            )

            subsuites_file_path = os.path.join(
                program_results_dir,
                f"{bootqa_program}-rep-{reps}-subsuites.json"
            )

            raw_counts_file_path = os.path.join(
                program_results_dir,
                f"{bootqa_program}-rep-{reps}-raw_counts.jsonl"
            )

            trex_raw_counts_file_path = os.path.join(
                program_results_dir,
                f"{bootqa_program}-rep-{reps}-trex-raw_counts.jsonl"
            )

            json_data = {}
            solutions = {}
            subsuites_data = {}
            qpu_run_times = []

            raw_counts_writer = RawCountsWriter(raw_counts_file_path)
            trex_counts_writer = RawCountsWriter(trex_raw_counts_file_path)

            for exp_id in range(1, num_piast_experiments + 1):
                print(f"\n--- Experiment {exp_id} ---")

                final_selected_tests = []
                experiment_cluster_assignments = []

                cluster_items = list(bootqa_clusters[bootqa_program].items())

                for cluster_idx, (cluster_id, cluster_tests) in enumerate(cluster_items):
                    filename = os.path.join(
                        "..",
                        "..",
                        "trained_qaoa_circuits",
                        "qaoa_tcs",
                        bootqa_program,
                        f"rep_{reps}",
                        f"{bootqa_program}_rep{reps}_cluster{cluster_idx}.qpy"
                    )

                    with open(filename, "rb") as f:
                        circuits = qpy.load(f)

                    circuit = circuits[0]

                    s = time.time()
                    counts, raw_record = run_circuit_with_batching_recorded(
                        circuit,
                        sampling_sampler,
                        algorithm="qaoa_tcs",
                        objective_mode="single_objective",
                        dataset=bootqa_program,
                        circuit_id=f"{bootqa_program}_rep{reps}_cluster{cluster_idx}",
                        backend=backend,
                        cluster_id=int(cluster_id),
                        iteration_id=exp_id,
                        shots_per_batch=80,
                        num_batches=1,
                    )
                    e = time.time()

                    raw_counts_writer.write(raw_record)

                    # TREx: a distinct hardware pass per twirl instance --
                    # cannot reuse the raw counts above. Written to a
                    # separate file since it's mitigation data, not the raw
                    # baseline; evaluation-time undoing/aggregation is
                    # piastq_execution.mitigation.aggregate_trex_records().
                    for twirl_idx in range(trex_twirl_instances):
                        twirl_mask = generate_random_twirl_mask(circuit.num_qubits, rng=trex_rng)
                        twirled_circuit = build_trex_twirled_circuit(circuit, twirl_mask)
                        _trex_counts, trex_record = run_circuit_with_batching_recorded(
                            twirled_circuit,
                            sampling_sampler,
                            algorithm="qaoa_tcs",
                            objective_mode="single_objective",
                            dataset=bootqa_program,
                            circuit_id=f"{bootqa_program}_rep{reps}_cluster{cluster_idx}_trex{twirl_idx}",
                            backend=backend,
                            cluster_id=int(cluster_id),
                            iteration_id=exp_id,
                            shots_per_batch=80,
                            num_batches=1,
                            twirl_mask=twirl_mask,
                        )
                        trex_counts_writer.write(trex_record)

                    qpu_run_times.append((e - s) * 1000)

                    most_likely = max(counts.items(), key=lambda x: x[1])[0]
                    bitstring = [int(b) for b in most_likely[::-1]]

                    indexes_selected_tests = [
                        index for index, value in enumerate(bitstring) if value == 1
                    ]

                    selected_tests = []
                    for index in indexes_selected_tests:
                        if index < len(cluster_tests):
                            selected_tests.append(cluster_tests[index])

                    for test in selected_tests:
                        if test not in final_selected_tests:
                            final_selected_tests.append(test)

                    experiment_cluster_assignments.append({
                        "cluster_id": int(cluster_id),
                        "cluster_test_cases": list(cluster_tests),
                        "bit_values": bitstring,
                        "selected_test_indexes_in_cluster": indexes_selected_tests,
                        "selected_tests_global_ids": selected_tests
                    })

                final_selected_tests = sorted(final_selected_tests)
                solutions[f"selected_test_suite_{exp_id}"] = final_selected_tests

                subsuites_data[f"experiment_{exp_id}"] = {
                    "final_selected_tests": final_selected_tests,
                    "cluster_assignments": experiment_cluster_assignments
                }

                if bootqa_program != "elevator2":
                    total_cost = sum(test_cases_costs[i] for i in final_selected_tests)
                    total_effectiveness = sum(test_cases_effectiveness[i] for i in final_selected_tests)

                    solutions[f"metrics_{exp_id}"] = {
                        "total_cost": total_cost,
                        "total_effectiveness": total_effectiveness,
                        "suite_size": len(final_selected_tests)
                    }
                else:
                    total_cost = sum(test_cases_costs[i] for i in final_selected_tests)
                    total_pcount = sum(test_cases_pcount[i] for i in final_selected_tests)
                    total_dist = sum(test_cases_dist[i] for i in final_selected_tests)

                    solutions[f"metrics_{exp_id}"] = {
                        "total_cost": total_cost,
                        "total_pcount": total_pcount,
                        "total_dist": total_dist,
                        "suite_size": len(final_selected_tests)
                    }

            json_data.update(solutions)
            json_data["mean_qpu_run_time(ms)"] = statistics.mean(qpu_run_times) if qpu_run_times else 0
            json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times) if len(qpu_run_times) > 1 else 0
            json_data["all_qpu_run_times(ms)"] = qpu_run_times

            with open(file_path, "w") as f:
                json.dump(json_data, f, indent=2)

            with open(subsuites_file_path, "w") as f:
                json.dump(subsuites_data, f, indent=2)

            raw_counts_writer.close()
            trex_counts_writer.close()

            print(f"Saved results: {file_path}")
            print(f"Saved cluster assignments: {subsuites_file_path}")
            print(f"Saved raw counts: {raw_counts_file_path}")
            print(f"Saved TREx raw counts: {trex_raw_counts_file_path}")


if __name__ == "__main__":
    # `python single_obj.py train` (re)trains QAOA-TCS single-objective
    # circuits on an ideal simulator -- this must run on the machine that
    # does the actual training. Default mode executes already-trained
    # circuits on hardware.
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        save_trained_circuits()
    else:
        run_hardware_execution()

