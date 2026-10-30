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
dry run. To run on the real machine, set `backend_name: piast_q` there (token
via PIASTQ_API_TOKEN) AND flip ALLOW_REAL_HARDWARE below -- without it this
script refuses to start on anything but an offline simulator. The real
machine is reached through `pcss-qapi` (Python >= 3.11; README section 2).

When does it stop?
  * a stage that fails (non-zero exit) stops the whole run -- no later stage
    starts on top of a broken one;
  * the pilot gate (stage 4) aborts BEFORE the expensive stages if the
    projected billed QPU time exceeds QPU_HOURS_LIMIT;
  * during execution every hardware batch first checks the account's billed
    QPU time (PCSS /me/usage) and refuses to submit one that would cross
    QPU_HOURS_LIMIT (piastq_execution/qpu_usage.py) -- the stage stops
    cleanly with everything so far saved.
  On the real machine up to MAX_JOBS_IN_FLIGHT circuits are submitted at once:
  every job waits minutes in the service's shared queue, so one-at-a-time
  submission would take weeks even though the QPU time is only 18.5 s per job.
  * It does NOT stop on its own for network trouble: dropped connections are
    retried for ~30 min per call; if that fails the stage dies, and
    `python run_all.py --resume` continues where it stopped (circuits already
    in the raw-counts files are reused, not re-run; finished calibration
    widths are skipped) instead of archiving the partial data and starting over.

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
ALLOW_REAL_HARDWARE = os.environ.get("PIASTQ_ALLOW_REAL_HARDWARE") == "1"
                                   # off unless run_cluster.sh (or you) sets PIASTQ_ALLOW_REAL_HARDWARE=1 -- or edit to True
RUN_UNIT_TESTS = True
RUN_DRIFT_CHECK = True             # ~56 extra calibration circuits at the end
QPU_HOURS_LIMIT = 25.0             # total QPU time available (BILLED time, queue wait is free)
MAX_JOBS_IN_FLIGHT = 16            # real machine only: circuits submitted at once (waits in the queue overlap)
ABORT_IF_PROJECTION_EXCEEDS_LIMIT = True   # pilot gate (stage 4)
MINUTES_PER_JOB_ESTIMATE = 3.4     # measured machine time per job on PIAST-Q (wall-clock, informational)
SINGLE_OBJECTIVE_DATASETS = {"gsdtsr", "iofrol", "paintcontrol", "elevator", "elevator2"}
MULTI_OBJECTIVE_DATASETS = {"flex", "grep", "gzip", "sed"}

RESUME = "--resume" in sys.argv[1:]   # continue an interrupted run instead of starting over

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
PY = sys.executable

sys.path.insert(0, SRC)


class StageFailed(Exception):
    pass


def _banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def _is_real_backend() -> bool:
    from piastq_execution.backend_config import load_backend_config

    return not load_backend_config()[1].startswith("offline_simulator")


def _run(name: str, args: list, cwd: str, log, clean_env: bool = False) -> float:
    """Runs one stage as a subprocess, echoing output to console + log.
    Returns elapsed seconds; raises StageFailed on a non-zero exit."""
    _banner(f"STAGE: {name}")
    start = time.time()
    # With N jobs in flight, up to N already-submitted jobs are still billed after
    # the last budget check, so the stop margin covers them (plus two spare).
    margin = (MAX_JOBS_IN_FLIGHT + 2) * 18.4974
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PIASTQ_QPU_LIMIT_SECONDS=str(QPU_HOURS_LIMIT * 3600 - margin))
    if _is_real_backend():
        env["PIASTQ_MAX_IN_FLIGHT"] = str(MAX_JOBS_IN_FLIGHT)
    if RESUME:
        env["PIASTQ_RESUME"] = "1"
    if clean_env:
        # the unit tests must never see (or touch) the real-run settings
        env = {k: v for k, v in env.items() if not k.startswith("PIASTQ_")}
        env["PYTHONUNBUFFERED"] = "1"
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
    if not is_offline:
        _check_real_backend_ready()
    return backend_name


def _check_real_backend_ready() -> None:
    """Fails fast, before any stage, if the real machine can't be used."""
    from piastq_execution.backend_config import is_pcss_backend, load_backend_config
    from piastq_execution.qpu_usage import billed_seconds

    backend_name = load_backend_config()[1]
    if not is_pcss_backend(backend_name):
        return
    if sys.version_info < (3, 11):
        raise SystemExit(f"PIAST-Q access (pcss-qapi) needs Python >= 3.11; this is {sys.version.split()[0]}. "
                         f"Start run_all.py from the Python 3.12 qiskit_env (README section 2).")
    try:
        import pcss_qapi  # noqa: F401
    except ImportError:
        raise SystemExit("pcss_qapi is not installed in this environment: pip install pcss-qapi")
    used = billed_seconds()
    if used is None:
        raise SystemExit("Cannot read the PIAST-Q usage counter (VPN connected? token valid?). Not starting.")
    print(f"PIAST-Q billed so far: {used:.1f}s of {QPU_HOURS_LIMIT * 3600:.0f}s "
          f"({(QPU_HOURS_LIMIT * 3600 - used) / 3600:.2f}h left)")


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
    return config, plans, inventories


def _packed_job_estimate(config, plans, inventories) -> dict:
    """Hardware jobs the run needs with circuits packed side by side
    (piastq_execution/packing.py): each (dataset, repetition) packs its raw
    circuits plus the TREx twirl copies; plus calibration and the drift check.
    The budget planner's own hours are the conservative one-circuit-per-job
    upper bound."""
    from piastq_execution.packing import count_jobs
    from piastq_execution.run_calibration import calibration_job_count

    execution = 0
    for plan in plans:
        for combo, methods in plan.methods_by_combo.items():
            widths = [r.num_qubits for r in inventories[combo].repetition_units[0]]
            passes = 1 + (config.cost_model.trex_twirl_instances if "trex" in methods else 0)
            execution += plan.repetitions * count_jobs(widths * passes)
    calibration = calibration_job_count()
    drift = 2 if RUN_DRIFT_CHECK else 0
    return {"execution_jobs": execution, "calibration_jobs": calibration, "drift_jobs": drift,
            "total_jobs": execution + calibration + drift}


def _measured_seconds_per_job() -> float | None:
    """Simulator fallback: mean measured wall-clock per calibration job, from
    the calibration files the calibration stage just wrote."""
    rates = []
    for path in glob.glob(os.path.join(ROOT, "calibration", "*_q*.json")):
        with open(path) as f:
            rec = json.load(f)
        rates.append(rec["calibration_wall_clock_seconds"] / max(1, rec["data"].get("calibration_jobs", 1)))
    return sum(rates) / len(rates) if rates else None


def _pilot_gate(config, jobs, calibration_billed_seconds=None) -> dict:
    """Projects the whole run from the calibration stage. With a real backend
    the rate is the BILLED seconds per calibration job (queue wait is not
    billed, so it must not inflate the projection); on a simulator there is no
    billing and the wall-clock figure is used instead."""
    from piastq_execution.run_calibration import calibration_job_count

    total_jobs = jobs["total_jobs"]
    planned_hours = total_jobs * config.cost_model.seconds_per_batch / 3600
    calibration_jobs = calibration_job_count()
    if calibration_billed_seconds is not None and calibration_billed_seconds > 0:
        measured = calibration_billed_seconds / calibration_jobs
        source = "billed"
    else:
        measured = _measured_seconds_per_job()
        source = "wall-clock"
    info = {**jobs, "planned_hours_at_assumed_rate": planned_hours,
            "assumed_seconds_per_job": config.cost_model.seconds_per_batch,
            "measured_seconds_per_job": measured, "measured_source": source,
            "expected_wall_clock_hours": total_jobs * MINUTES_PER_JOB_ESTIMATE / 60}
    print(f"Jobs: {jobs['execution_jobs']} execution + {jobs['calibration_jobs']} calibration + "
          f"{jobs['drift_jobs']} drift = {total_jobs}; at ~{MINUTES_PER_JOB_ESTIMATE} min of machine time "
          f"per job that is ~{info['expected_wall_clock_hours']:.0f} h of wall-clock.")
    if measured is None:
        print("Pilot gate: no calibration data found, cannot project QPU time.")
        return info
    projected = total_jobs * measured / 3600
    info["projected_hours_at_measured_rate"] = projected
    print(f"Pilot gate: measured {measured:.2f}s/job ({source}) (assumed "
          f"{config.cost_model.seconds_per_batch}s) -> projected {projected:.2f}h of billed QPU time "
          f"(limit {QPU_HOURS_LIMIT}h)")
    if projected > QPU_HOURS_LIMIT and ABORT_IF_PROJECTION_EXCEEDS_LIMIT:
        raise StageFailed(
            f"projected QPU time {projected:.2f}h exceeds the {QPU_HOURS_LIMIT}h limit at the measured "
            f"rate. Reduce repetitions / trex_twirl_instances in configs/execution_plan.yaml and re-run."
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

            from piastq_execution.qpu_usage import billed_seconds

            billed_start = billed_seconds()
            summary["billed_seconds_at_start"] = billed_start
            if RESUME:
                print("RESUME mode: keeping existing raw counts / calibration, only missing work is executed.")
            else:
                _archive_old_raw_counts(stamp)
            config, plans, inventories = _plan()

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
                try:
                    summary["stages"][name] = round(_run(name, args, cwd, log, clean_env=(name == "unit tests")), 1)
                finally:
                    summary.setdefault("billed_seconds_after_stage", {})[name] = billed_seconds()

            calibration_billed = None
            for stage in stages:
                before = billed_seconds() if stage[0].startswith("calibration") else None
                run_stage(stage)
                if before is not None:
                    after = billed_seconds()
                    if after is not None:
                        calibration_billed = after - before

            summary["pilot_gate"] = _pilot_gate(config, _packed_job_estimate(config, plans, inventories), calibration_billed)

            execution_stages = []
            if has_single:
                execution_stages.append(("QAOA-TCS single-objective", ["single_obj.py"], os.path.join(SRC, "qaoa_tcs")))
            if has_multi:
                execution_stages.append(("QAOA-TCS multi-objective", ["multi_obj.py"], os.path.join(SRC, "qaoa_tcs")))
            if has_igdec:
                execution_stages.append(("IGDec-QAOA", ["loch_qaoa_tcm_extract_circuits.py"], os.path.join(SRC, "igdec_qaoa")))
            for stage in execution_stages:
                run_stage(stage)

            drift_report = os.path.join(ROOT, "calibration", "drift", "m3_drift_report.json")
            if RUN_DRIFT_CHECK and RESUME and os.path.exists(drift_report):
                print("Drift check already done, skipping (resume).")
            elif RUN_DRIFT_CHECK:
                run_stage(("drift check", ["run_calibration.py", "drift"], os.path.join(SRC, "piastq_execution")))

            run_stage(("evaluation", ["evaluate_all.py"], os.path.join(SRC, "piastq_execution")))
            run_stage(("statistical comparison", ["compare_all.py"], os.path.join(SRC, "piastq_execution")))

            billed_end = billed_seconds()
            if billed_end is not None:
                measured_hours = billed_end / 3600       # whole account: what the 25h limit counts
                summary["billed_qpu_hours_account_total"] = round(measured_hours, 3)
                if billed_start is not None:
                    summary["billed_qpu_hours_this_invocation"] = round((billed_end - billed_start) / 3600, 3)
                kind = "billed"
            else:
                measured_hours = _measured_qpu_seconds() / 3600   # simulator: wall-clock only
                kind = "wall-clock (simulator, nothing is billed)"
            summary["measured_qpu_hours"] = round(measured_hours, 3)
            summary["status"] = "completed"
            _banner(f"ALL STAGES COMPLETED. QPU time, {kind}: {measured_hours:.2f}h "
                    f"(limit {QPU_HOURS_LIMIT}h). Report: results/evaluation/comparison_report.txt")
            code = 0
        except StageFailed as e:
            summary["status"] = f"failed: {e}"
            _banner(f"RUN STOPPED: {e}\nEverything executed so far is saved. To continue without "
                    f"repeating it: python run_all.py --resume")
            code = 1
        finally:
            summary_path = os.path.join(ROOT, "logs", f"run_summary_{stamp}.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Log: {log_path}\nSummary: {summary_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
