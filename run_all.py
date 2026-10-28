"""One-click full pipeline: press play and it runs everything, start to end.

    python run_all.py

Stages, in order (each one is the same script you would otherwise start by
hand; the first failing stage stops the run):

  1. unit tests
  1b. QUBO backfill                   (adds the QUBO coefficients evaluation
                                       needs to already-trained circuits; no
                                       retraining, skipped if already there)
  2. budget plan                      (prints/saves the QPU-time estimate)
  3. MEM + M3 calibration             (also saves the backend T1/T2 snapshot)
  4. pilot gate                       (measured seconds/circuit -> projected
                                       total QPU hours; aborts if > limit)
  5. QAOA-TCS single-objective        (raw + TREx counts saved)
  6. QAOA-TCS multi-objective
  7. IGDec-QAOA                       (only if an IGDec combo is in the plan)
  8. end-of-session drift check       (re-measures M3)
  9. evaluation + statistical comparison  (local, no QPU)

Which backend it uses is decided ONLY by configs/backend.yaml (or the
PIASTQ_* env vars): the default `offline_simulator_no_noise` is a free local
dry run. To run on the real machine, set the real backend there AND flip
ALLOW_REAL_HARDWARE below -- without it this script refuses to start on
anything but an offline simulator.

Output: console + logs/run_all_<timestamp>.log, and a final
logs/run_summary_<timestamp>.json (stage times, predicted vs measured QPU
time). Raw-count files from an earlier attempt are moved to
results/_archive_<timestamp>/ first, because the writers append and a
re-run would otherwise mix two runs in one file.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
ALLOW_REAL_HARDWARE = False        # must be True to run on anything but an offline simulator
RUN_UNIT_TESTS = True
RUN_DRIFT_CHECK = True             # ~56 extra calibration circuits at the end
QPU_HOURS_LIMIT = 25.0             # total QPU time available
ABORT_IF_PROJECTION_EXCEEDS_LIMIT = True   # pilot gate (stage 4)
DRIFT_CIRCUITS = 56                # M3 circuits for widths 1..7 (2 * sum(1..7))
SINGLE_OBJECTIVE_DATASETS = {"gsdtsr", "iofrol", "paintcontrol", "elevator", "elevator2"}
MULTI_OBJECTIVE_DATASETS = {"flex", "grep", "gzip", "sed"}

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
PY = sys.executable

sys.path.insert(0, SRC)


class StageFailed(Exception):
    pass


def _banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def _run(name: str, args: list, cwd: str, log) -> float:
    """Runs one stage as a subprocess, echoing output to console + log.
    Returns elapsed seconds; raises StageFailed on a non-zero exit."""
    _banner(f"STAGE: {name}")
    start = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [PY] + args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    proc.wait()
    elapsed = time.time() - start
    log.flush()
    if proc.returncode != 0:
        raise StageFailed(f"stage '{name}' failed with exit code {proc.returncode}")
    print(f"[{name}] done in {elapsed:.1f}s", flush=True)
    return elapsed


def _check_backend_allowed() -> str:
    from piastq_execution.backend_config import load_backend_config

    _token, backend_name = load_backend_config()
    is_offline = backend_name.startswith("offline_simulator")
    print(f"Backend: {backend_name}  ({'offline simulator - dry run' if is_offline else 'REAL HARDWARE'})")
    if not is_offline and not ALLOW_REAL_HARDWARE:
        raise SystemExit(
            f"Refusing to start: backend '{backend_name}' is not an offline simulator and "
            f"ALLOW_REAL_HARDWARE is False in run_all.py."
        )
    return backend_name


def _archive_old_raw_counts(stamp: str) -> None:
    stale = [
        p for p in glob.glob(os.path.join(ROOT, "results", "**", "*raw_counts.jsonl"), recursive=True)
        if "_archive_" not in p
    ]
    if not stale:
        return
    archive = os.path.join(ROOT, "results", f"_archive_{stamp}")
    for path in stale:
        dest = os.path.join(archive, os.path.relpath(path, os.path.join(ROOT, "results")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(path, dest)
    print(f"Moved {len(stale)} raw-counts file(s) from an earlier run to {archive}")


def _plan():
    from piastq_execution.budget_planner import build_inventories_from_config, load_config, plan_all

    config_path = os.path.join(ROOT, "configs", "execution_plan.yaml")
    config = load_config(config_path)
    inventories = build_inventories_from_config(config_path, os.path.join(ROOT, "trained_qaoa_circuits"))
    plans = plan_all(config, inventories)
    return config, plans


def _planned_batches(config, plans) -> float:
    return sum(p.estimated_seconds for p in plans) / config.cost_model.seconds_per_batch


def _measured_seconds_per_circuit() -> float | None:
    """Mean measured wall-clock per 200-shot calibration circuit, from the
    MEM/M3 calibration files the calibration stage just wrote."""
    total_seconds = 0.0
    total_circuits = 0
    for path in glob.glob(os.path.join(ROOT, "calibration", "*_q*.json")):
        with open(path) as f:
            rec = json.load(f)
        width = rec["num_qubits"]
        circuits = 2 ** width if rec["method"] == "mem" else 2 * width
        total_seconds += rec["calibration_wall_clock_seconds"]
        total_circuits += circuits
    return total_seconds / total_circuits if total_circuits else None


def _pilot_gate(config, plans) -> dict:
    batches = _planned_batches(config, plans) + (DRIFT_CIRCUITS if RUN_DRIFT_CHECK else 0)
    planned_hours = batches * config.cost_model.seconds_per_batch / 3600
    measured = _measured_seconds_per_circuit()
    info = {"planned_batches": batches, "planned_hours_at_assumed_rate": planned_hours,
            "assumed_seconds_per_batch": config.cost_model.seconds_per_batch,
            "measured_seconds_per_batch": measured}
    if measured is None:
        print("Pilot gate: no calibration data found, cannot project QPU time.")
        return info
    projected = batches * measured / 3600
    info["projected_hours_at_measured_rate"] = projected
    print(f"Pilot gate: {batches:.0f} batches planned; measured {measured:.2f}s/batch "
          f"(assumed {config.cost_model.seconds_per_batch}s) -> projected {projected:.2f}h "
          f"of {QPU_HOURS_LIMIT}h")
    if projected > QPU_HOURS_LIMIT and ABORT_IF_PROJECTION_EXCEEDS_LIMIT:
        raise StageFailed(
            f"projected QPU time {projected:.2f}h exceeds the {QPU_HOURS_LIMIT}h limit at the measured "
            f"rate. Reduce repetitions / trex_twirl_instances in configs/execution_plan.yaml and re-run "
            f"(or set ABORT_IF_PROJECTION_EXCEEDS_LIMIT = False if queue time inflates the measurement)."
        )
    return info


def _measured_qpu_seconds() -> float:
    """Sum of measured wall-clock over every execution + calibration record
    of this run."""
    total = 0.0
    for path in glob.glob(os.path.join(ROOT, "results", "**", "*raw_counts.jsonl"), recursive=True):
        if "_archive_" in path:
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    total += json.loads(line)["total_wall_clock_seconds"]
    for path in glob.glob(os.path.join(ROOT, "calibration", "**", "*_q*.json"), recursive=True):
        with open(path) as f:
            total += json.load(f).get("calibration_wall_clock_seconds", 0.0)
    return total


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log_path = os.path.join(ROOT, "logs", f"run_all_{stamp}.log")
    summary: dict = {"started": stamp, "stages": {}}

    with open(log_path, "w") as log:
        try:
            _banner("PIAST-Q4QAOA full pipeline")
            summary["backend"] = _check_backend_allowed()
            import qiskit_aqt_provider  # noqa: F401  (fail early with a clear error)

            _archive_old_raw_counts(stamp)
            config, plans = _plan()

            stages = []
            if RUN_UNIT_TESTS:
                stages.append(("unit tests", ["-m", "unittest", "discover", "-s", "tests"], ROOT))
            planned = set()
            for plan in plans:
                planned |= set(plan.methods_by_combo)
            has_single = any(c.endswith("_qaoa_tcs") and c.split("_")[0] in SINGLE_OBJECTIVE_DATASETS for c in planned)
            has_multi = any(c.endswith("_qaoa_tcs") and c.split("_")[0] in MULTI_OBJECTIVE_DATASETS for c in planned)
            has_igdec = any(c.endswith("_igdec_qaoa") for c in planned)
            if has_single:
                stages.append(("QUBO backfill: QAOA-TCS single-objective", ["single_obj.py", "qubos"], os.path.join(SRC, "qaoa_tcs")))
            if has_multi:
                stages.append(("QUBO backfill: QAOA-TCS multi-objective", ["multi_obj.py", "qubos"], os.path.join(SRC, "qaoa_tcs")))
            if has_igdec:
                stages.append(("QUBO backfill: IGDec-QAOA", ["loch_qaoa_tcm_extract_circuits.py", "qubos"], os.path.join(SRC, "igdec_qaoa")))
            stages.append(("budget plan", ["budget_planner.py"], os.path.join(SRC, "piastq_execution")))
            stages.append(("calibration (MEM + M3)", ["run_calibration.py"], os.path.join(SRC, "piastq_execution")))

            def run_stage(stage):
                name, args, cwd = stage
                summary["stages"][name] = round(_run(name, args, cwd, log), 1)

            for stage in stages:
                run_stage(stage)

            summary["pilot_gate"] = _pilot_gate(config, plans)

            execution_stages = []
            if has_single:
                execution_stages.append(("QAOA-TCS single-objective", ["single_obj.py"], os.path.join(SRC, "qaoa_tcs")))
            if has_multi:
                execution_stages.append(("QAOA-TCS multi-objective", ["multi_obj.py"], os.path.join(SRC, "qaoa_tcs")))
            if has_igdec:
                execution_stages.append(("IGDec-QAOA", ["loch_qaoa_tcm_extract_circuits.py"], os.path.join(SRC, "igdec_qaoa")))
            for stage in execution_stages:
                run_stage(stage)

            if RUN_DRIFT_CHECK:
                run_stage(("drift check", ["run_calibration.py", "drift"], os.path.join(SRC, "piastq_execution")))

            run_stage(("evaluation", ["evaluate_all.py"], os.path.join(SRC, "piastq_execution")))
            run_stage(("statistical comparison", ["compare_all.py"], os.path.join(SRC, "piastq_execution")))

            measured_hours = _measured_qpu_seconds() / 3600
            summary["measured_qpu_hours"] = round(measured_hours, 3)
            summary["status"] = "completed"
            _banner(f"ALL STAGES COMPLETED. Measured QPU-side time: {measured_hours:.2f}h "
                    f"(limit {QPU_HOURS_LIMIT}h). Report: results/evaluation/comparison_report.txt")
            code = 0
        except StageFailed as e:
            summary["status"] = f"failed: {e}"
            _banner(f"RUN STOPPED: {e}")
            code = 1
        finally:
            summary_path = os.path.join(ROOT, "logs", f"run_summary_{stamp}.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Log: {log_path}\nSummary: {summary_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
