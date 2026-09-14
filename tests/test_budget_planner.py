"""Synthetic unit tests for piastq_execution.budget_planner.

All inventories here are tiny, hand-built (1-3 circuits, 2 qubits) so every
expected cost can be verified by hand. Nothing touches trained_qaoa_circuits/
on disk or any backend -- except TestResolveAllComboSettings, which builds a
tiny real (temp-dir) circuits-on-disk + YAML config layout, since
resolve_all_combo_settings() exercises the real file-reading path end to end.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.budget_planner import (
    DEFAULT_TREX_TWIRL_INSTANCES,
    CircuitRecord,
    ComboInventory,
    CostModel,
    PoolSpec,
    load_config,
    plan_pool,
    _shots_to_batches,
    resolve_all_combo_settings,
)


def reusable_inventory(combo, circuit_ids, width, methods=None):
    records = [CircuitRecord(circuit_id=cid, num_qubits=width) for cid in circuit_ids]
    return ComboInventory(
        combo=combo,
        algorithm="qaoa_tcs",
        objective_mode="single_objective",
        reusable_across_repetitions=True,
        repetition_units=[records],
    )


def nonreusable_inventory(combo, samplings, width):
    """`samplings` is a list of circuit-id-lists, one per (non-reusable) repetition unit."""
    units = [[CircuitRecord(circuit_id=cid, num_qubits=width) for cid in s] for s in samplings]
    return ComboInventory(
        combo=combo,
        algorithm="igdec_qaoa",
        objective_mode="single_objective",
        reusable_across_repetitions=False,
        repetition_units=units,
    )


STANDARD_COST_MODEL = CostModel(seconds_per_batch=15.0, shots_per_batch_cap=200, calibration_shots_per_circuit=200)


class TestCostModel(unittest.TestCase):
    def test_batches_ceiling(self):
        self.assertEqual(STANDARD_COST_MODEL.batches(200), 1)
        self.assertEqual(STANDARD_COST_MODEL.batches(201), 2)
        self.assertEqual(STANDARD_COST_MODEL.batches(400), 2)
        self.assertEqual(STANDARD_COST_MODEL.batches(0), 0)

    def test_circuit_seconds(self):
        self.assertEqual(STANDARD_COST_MODEL.circuit_seconds(200), 15.0)
        self.assertEqual(STANDARD_COST_MODEL.circuit_seconds(1048), 6 * 15.0)

    def test_mem_and_m3_calibration_circuit_counts(self):
        self.assertEqual(CostModel.mem_calibration_circuits(2), 4)
        self.assertEqual(CostModel.mem_calibration_circuits(7), 128)
        self.assertEqual(CostModel.m3_calibration_circuits(2), 4)
        self.assertEqual(CostModel.m3_calibration_circuits(7), 14)


class TestComboInventory(unittest.TestCase):
    def test_reusable_effective_circuits_scale_with_repetitions(self):
        inv = reusable_inventory("A", ["c1", "c2"], width=2)
        self.assertEqual(len(inv.effective_circuits(1)), 2)
        self.assertEqual(len(inv.effective_circuits(3)), 6)
        self.assertTrue(inv.available_repetitions == float("inf"))

    def test_nonreusable_effective_circuits_bounded_by_available_samplings(self):
        inv = nonreusable_inventory("B", [["s1c1"], ["s2c1"], ["s3c1"]], width=2)
        self.assertEqual(inv.available_repetitions, 3)
        self.assertEqual(len(inv.effective_circuits(1)), 1)
        self.assertEqual(len(inv.effective_circuits(2)), 2)
        self.assertEqual(len(inv.effective_circuits(5)), 3)  # capped at what's available

    def test_dropped_ids_are_excluded(self):
        inv = reusable_inventory("A", ["c1", "c2", "c3"], width=2)
        remaining = inv.effective_circuits(1, dropped_ids={"c3"})
        self.assertEqual([r.circuit_id for r in remaining], ["c1", "c2"])


class TestPlanPoolNoDegradationNeeded(unittest.TestCase):
    def test_fits_target_as_is(self):
        inv = reusable_inventory("A", ["c1", "c2"], width=2)
        pool = PoolSpec(name="p", total_hours=100, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw"]}, STANDARD_COST_MODEL,
            target_repetitions=3, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.repetitions, 3)
        self.assertEqual(plan.shots_per_circuit, 200)
        self.assertEqual(plan.dropped_circuits, {})
        self.assertEqual(plan.dropped_trex_combos, [])
        # 2 circuits * 3 reps * 1 batch * 15s = 90s
        self.assertAlmostEqual(plan.estimated_seconds, 90.0)


class TestTrexDrop(unittest.TestCase):
    def test_trex_cost_scales_with_configured_twirl_instances(self):
        # 1 circuit, raw=15s. TREx must run trex_twirl_instances *separate*
        # hardware passes over that same circuit (it can't reuse raw counts),
        # so with 32 instances (the default) it costs 32*15s=480s, not 15s --
        # this is the exact accounting bug the planner had before
        # CostModel.trex_twirl_instances existed (it used to price TREx as a
        # single extra pass regardless of how many twirl instances the
        # execution scripts actually run).
        inv = reusable_inventory("A", ["c1"], width=2)
        pool = PoolSpec(name="p", total_hours=1000, combos=["A"])  # ample budget, no degradation
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw", "trex"]}, STANDARD_COST_MODEL,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertEqual(STANDARD_COST_MODEL.trex_twirl_instances, 32)
        self.assertAlmostEqual(plan.estimated_seconds, 15.0 + 32 * 15.0)

    def test_drops_trex_when_that_alone_fits(self):
        inv = reusable_inventory("A", ["c1"], width=2)
        # raw=15s, +trex(32 instances)=480s => 495s total; a budget that only
        # fits the raw-only cost forces the drop.
        pool = PoolSpec(name="p", total_hours=20 / 3600, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw", "trex"]}, STANDARD_COST_MODEL,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.dropped_trex_combos, ["A"])
        self.assertEqual(plan.methods_by_combo["A"], ["raw"])
        self.assertEqual(plan.repetitions, 1)
        self.assertEqual(plan.shots_per_circuit, 200)
        self.assertAlmostEqual(plan.estimated_seconds, 15.0)

    def test_custom_twirl_instances_count_changes_cost(self):
        from piastq_execution.budget_planner import CostModel as _CostModel

        custom_model = _CostModel(seconds_per_batch=15.0, shots_per_batch_cap=200,
                                   calibration_shots_per_circuit=200, trex_twirl_instances=4)
        inv = reusable_inventory("A", ["c1"], width=2)
        pool = PoolSpec(name="p", total_hours=1000, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw", "trex"]}, custom_model,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertAlmostEqual(plan.estimated_seconds, 15.0 + 4 * 15.0)


class TestRepetitionReduction(unittest.TestCase):
    def test_reduces_repetitions_when_trex_drop_is_not_enough(self):
        inv = reusable_inventory("A", ["c1"], width=2)
        # raw only, 1 circuit * 15s/rep. target reps=5 -> 75s; budget=45s -> fits at reps=3 (45s).
        pool = PoolSpec(name="p", total_hours=45 / 3600, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw"]}, STANDARD_COST_MODEL,
            target_repetitions=5, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.repetitions, 3)
        self.assertEqual(plan.shots_per_circuit, 200)
        self.assertAlmostEqual(plan.estimated_seconds, 45.0)
        self.assertTrue(any("Reduced repetitions" in n for n in plan.notes))

    def test_nonreusable_repetition_reduction_drops_whole_sampling_units(self):
        inv = nonreusable_inventory("B", [["s1"], ["s2"], ["s3"]], width=2)
        # 3 samplings * 15s = 45s at target reps=3; budget=30s -> fits at reps=2 (30s).
        pool = PoolSpec(name="p", total_hours=30 / 3600, combos=["B"])
        plan = plan_pool(
            pool, {"B": inv}, {"B": ["raw"]}, STANDARD_COST_MODEL,
            target_repetitions=3, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.repetitions, 2)
        self.assertAlmostEqual(plan.estimated_seconds, 30.0)


class TestShotReduction(unittest.TestCase):
    def test_reduces_shots_once_repetitions_floor_is_reached(self):
        inv = reusable_inventory("A", ["c1"], width=2)
        pool = PoolSpec(name="p", total_hours=100 / 3600, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw"]}, STANDARD_COST_MODEL,
            target_repetitions=3, target_shots_per_circuit=2048, min_shots_per_circuit=200,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.repetitions, 1)
        # largest S (stepping down by 200 from 2048) with ceil(S/200)*15 <= 100s
        # is S=1048 -> 6 batches * 15s = 90s <= 100s; S=1248 -> 7*15=105s > 100s.
        self.assertEqual(plan.shots_per_circuit, 1048)
        self.assertAlmostEqual(plan.estimated_seconds, 90.0)


class TestCircuitSubsampling(unittest.TestCase):
    def test_drops_circuits_as_last_resort_and_reports_which(self):
        inv = reusable_inventory("A", ["c1", "c2", "c3"], width=2)
        # at reps=1, shots=floor=200: 3 circuits * 15s = 45s > 30s budget;
        # dropping the last circuit brings it to 2 * 15s = 30s <= 30s.
        pool = PoolSpec(name="p", total_hours=30 / 3600, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw"]}, STANDARD_COST_MODEL,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.repetitions, 1)
        self.assertEqual(plan.shots_per_circuit, 200)
        self.assertEqual(plan.dropped_circuits, {"A": ["c3"]})
        self.assertAlmostEqual(plan.estimated_seconds, 30.0)
        self.assertTrue(any("Dropped circuit c3" in n for n in plan.notes))

    def test_absurdly_small_budget_drops_every_circuit_and_still_fits_at_zero(self):
        # Dropping every circuit of a width removes the need to calibrate that
        # width too, so the trivial "run nothing" solution (cost 0) always
        # fits a non-negative budget -- verify degradation actually drives
        # all the way there instead of looping or crashing.
        inv = reusable_inventory("A", ["c1", "c2"], width=7)
        pool = PoolSpec(name="p", total_hours=0, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw", "mem"]}, STANDARD_COST_MODEL,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(sorted(plan.dropped_circuits["A"]), ["c1", "c2"])
        self.assertAlmostEqual(plan.estimated_seconds, 0.0)

    def test_negative_budget_is_flagged_infeasible(self):
        inv = reusable_inventory("A", ["c1"], width=2)
        pool = PoolSpec(name="p", total_hours=-1, combos=["A"])
        plan = plan_pool(
            pool, {"A": inv}, {"A": ["raw"]}, STANDARD_COST_MODEL,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        self.assertFalse(plan.fits_budget)
        self.assertTrue(any("INFEASIBLE" in n for n in plan.notes))


class TestLoadConfigTrexTwirlInstances(unittest.TestCase):
    def test_reads_trex_twirl_instances_into_cost_model(self):
        path = "/tmp/piastq_test_execution_plan.yaml"
        with open(path, "w") as f:
            f.write(
                "seconds_per_batch: 15\n"
                "shots_per_batch_cap: 200\n"
                "calibration_shots_per_circuit: 200\n"
                "trex_twirl_instances: 5\n"
                "target_repetitions: 10\n"
                "target_shots_per_circuit: 2048\n"
                "min_shots_per_circuit: 200\n"
                "combos: {}\n"
                "pools: []\n"
            )
        try:
            config = load_config(path)
            self.assertEqual(config.cost_model.trex_twirl_instances, 5)
        finally:
            os.remove(path)

    def test_missing_key_falls_back_to_default(self):
        path = "/tmp/piastq_test_execution_plan_no_trex.yaml"
        with open(path, "w") as f:
            f.write("combos: {}\npools: []\n")
        try:
            config = load_config(path)
            self.assertEqual(config.cost_model.trex_twirl_instances, DEFAULT_TREX_TWIRL_INSTANCES)
        finally:
            os.remove(path)


class TestCalibrationSharing(unittest.TestCase):
    def test_mem_calibration_counted_once_per_width_shared_across_combos_in_pool(self):
        inv_a = reusable_inventory("A", ["a1"], width=2)
        inv_b = reusable_inventory("B", ["b1"], width=2)  # same width as A
        pool = PoolSpec(name="p", total_hours=1, combos=["A", "B"])
        plan = plan_pool(
            pool, {"A": inv_a, "B": inv_b}, {"A": ["raw", "mem"], "B": ["raw", "mem"]},
            STANDARD_COST_MODEL, target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        # raw: (1+1) circuits * 15s = 30s
        # mem calibration: width=2 shared -> counted ONCE: 2**2=4 circuits * 15s = 60s
        # if calibration were double-counted this would be 150s instead of 90s.
        self.assertAlmostEqual(plan.estimated_seconds, 90.0)

    def test_independent_pools_do_not_share_calibration(self):
        inv_a = reusable_inventory("A", ["a1"], width=2)
        inv_b = reusable_inventory("B", ["b1"], width=2)
        pool_a = PoolSpec(name="pool_a", total_hours=1, combos=["A"])
        pool_b = PoolSpec(name="pool_b", total_hours=1, combos=["B"])

        plan_a = plan_pool(
            pool_a, {"A": inv_a}, {"A": ["raw", "mem"]}, STANDARD_COST_MODEL,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        plan_b = plan_pool(
            pool_b, {"B": inv_b}, {"B": ["raw", "mem"]}, STANDARD_COST_MODEL,
            target_repetitions=1, target_shots_per_circuit=200, min_shots_per_circuit=200,
        )
        # each pool pays its own calibration: 15s (raw) + 60s (mem, width=2) = 75s
        self.assertAlmostEqual(plan_a.estimated_seconds, 75.0)
        self.assertAlmostEqual(plan_b.estimated_seconds, 75.0)


class TestShotsToBatches(unittest.TestCase):
    def test_exact_multiple_of_cap_is_one_batch_no_remainder(self):
        self.assertEqual(_shots_to_batches(200, 200), (1, 200, 0))

    def test_below_cap_is_zero_full_batches_plus_remainder(self):
        self.assertEqual(_shots_to_batches(80, 200), (0, 200, 80))

    def test_matches_the_2048_shot_pattern_from_the_original_study(self):
        # 2048 shots at 200/batch = 10 full batches + 48-shot remainder.
        self.assertEqual(_shots_to_batches(2048, 200), (10, 200, 48))

    def test_zero_shots_is_zero_batches_zero_remainder(self):
        self.assertEqual(_shots_to_batches(0, 200), (0, 200, 0))


class TestResolveAllComboSettings(unittest.TestCase):
    """Builds a tiny real (temp-dir) trained_qaoa_circuits/ + execution_plan.yaml
    layout -- mirroring exactly what the real repo's files look like -- and
    exercises resolve_all_combo_settings() end to end, no mocking of the
    planner itself."""

    def _write_qpy(self, path, num_qubits):
        from qiskit import QuantumCircuit, qpy

        os.makedirs(os.path.dirname(path), exist_ok=True)
        qc = QuantumCircuit(num_qubits)
        qc.measure_all()
        with open(path, "wb") as f:
            qpy.dump([qc], f)

    def test_resolves_repetitions_shots_batches_and_methods_per_combo(self):
        with tempfile.TemporaryDirectory() as tmp:
            circuits_dir = os.path.join(tmp, "trained_qaoa_circuits")
            # QAOA-TCS combo: 2 reusable circuits, width 2.
            self._write_qpy(os.path.join(circuits_dir, "qaoa_tcs", "toy", "rep_1", "toy_rep1_cluster0.qpy"), 2)
            self._write_qpy(os.path.join(circuits_dir, "qaoa_tcs", "toy", "rep_1", "toy_rep1_cluster1.qpy"), 2)

            config_path = os.path.join(tmp, "execution_plan.yaml")
            with open(config_path, "w") as f:
                f.write(
                    "seconds_per_batch: 15\n"
                    "shots_per_batch_cap: 200\n"
                    "target_shots_per_circuit: 2048\n"
                    "min_shots_per_circuit: 200\n"
                    "target_repetitions: 10\n"
                    "calibration_shots_per_circuit: 200\n"
                    "trex_twirl_instances: 32\n"
                    "combos:\n"
                    "  toy_qaoa_tcs:\n"
                    "    algorithm: qaoa_tcs\n"
                    "    objective_mode: single_objective\n"
                    "    dataset: toy\n"
                    "    circuits_dir: toy\n"
                    "    methods: [raw, mem, m3, trex]\n"
                    "pools:\n"
                    "  - name: toy_pool\n"
                    "    total_hours: 15\n"
                    "    combos: [toy_qaoa_tcs]\n"
                )

            settings = resolve_all_combo_settings(config_path=config_path, trained_circuits_dir=circuits_dir)

            self.assertIn("toy_qaoa_tcs", settings)
            s = settings["toy_qaoa_tcs"]
            self.assertEqual(s.pool_name, "toy_pool")
            # 2 circuits x 10 reps x 2048 shots comfortably fits 15h for
            # raw+mem+m3 alone (~1h), but TREx's 32x multiplier alone pushes
            # it to ~30h -- so reps/shots stay at target and only TREx gets
            # dropped (matches plan_pool's phase-2 "drop TREx first" order).
            self.assertEqual(s.repetitions, 10)
            self.assertEqual(s.shots_per_circuit, 2048)
            self.assertEqual((s.num_batches, s.shots_per_batch, s.remainder_shots), (10, 200, 48))
            self.assertEqual(s.methods, ["raw", "mem", "m3"])
            self.assertFalse(s.trex_enabled)

    def test_degraded_combo_has_trex_disabled_and_reduced_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            circuits_dir = os.path.join(tmp, "trained_qaoa_circuits")
            # A deliberately large reusable inventory so a tiny pool budget
            # forces real degradation.
            for i in range(50):
                self._write_qpy(os.path.join(circuits_dir, "qaoa_tcs", "big", "rep_1", f"big_rep1_cluster{i}.qpy"), 7)

            config_path = os.path.join(tmp, "execution_plan.yaml")
            with open(config_path, "w") as f:
                f.write(
                    "seconds_per_batch: 15\n"
                    "shots_per_batch_cap: 200\n"
                    "target_shots_per_circuit: 2048\n"
                    "min_shots_per_circuit: 200\n"
                    "target_repetitions: 10\n"
                    "calibration_shots_per_circuit: 200\n"
                    "trex_twirl_instances: 32\n"
                    "combos:\n"
                    "  big_qaoa_tcs:\n"
                    "    algorithm: qaoa_tcs\n"
                    "    objective_mode: single_objective\n"
                    "    dataset: big\n"
                    "    circuits_dir: big\n"
                    "    methods: [raw, mem, m3, trex]\n"
                    "pools:\n"
                    "  - name: tiny_pool\n"
                    "    total_hours: 0.05\n"  # ~3 minutes -- forces heavy degradation
                    "    combos: [big_qaoa_tcs]\n"
                )

            settings = resolve_all_combo_settings(config_path=config_path, trained_circuits_dir=circuits_dir)
            s = settings["big_qaoa_tcs"]
            self.assertFalse(s.trex_enabled, "TREx must be dropped in the resolved settings, not just the plan")
            self.assertNotIn("trex", s.methods)


if __name__ == "__main__":
    unittest.main()
