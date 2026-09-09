"""Checkpointed six-angle FEMM validation runner.

Default behavior runs only the two candidates marked ``smoke`` and enumerates
the same three physically reasonable material mappings for both candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


VALIDATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VALIDATION_ROOT.parent
FEMM_SCRIPTS = PROJECT_ROOT / "femm_zone" / "scripts"
sys.path.insert(0, str(FEMM_SCRIPTS))

from build_femm_topology import (  # noqa: E402
    DEFAULT_MAT,
    DEFAULT_TEMPLATE,
    MAPPING_CANDIDATES,
    build_topology,
)


ANGLES_DEG = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
POLE_PAIRS = 4
CURRENT_AMPLITUDE_A = 3.5
SELECTED_CSV = VALIDATION_ROOT / "selected_candidates.csv"
CHECKPOINT_PATH = VALIDATION_ROOT / "checkpoint.json"
RESULTS_CSV = VALIDATION_ROOT / "validation_results.csv"
WORKER_SCRIPT = VALIDATION_ROOT / "scripts" / "solve_one_angle.py"
RUN_LOCK_PATH = VALIDATION_ROOT / ".validation.lock"


@contextmanager
def validation_lock():
    """Hold an OS-released lock so two launch terminals cannot share outputs."""
    RUN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    stream = RUN_LOCK_PATH.open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError(
            f"Another validation runner already holds {RUN_LOCK_PATH}; do not launch a second copy"
        ) from exc
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_candidates(stage: str) -> list[dict[str, str]]:
    with SELECTED_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if stage == "smoke":
        rows = [row for row in rows if row["run_stage"] == "smoke"]
    return rows


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {
        "status": "new",
        "angles_deg": ANGLES_DEG,
        "pole_pairs": POLE_PAIRS,
        "current_amplitude_a": CURRENT_AMPLITUDE_A,
        "torque_method": "mo_gapintegral('sliding_airgap', 0)",
        "tasks": {},
    }


def result_metrics(torques: np.ndarray, historical_tavg: float, historical_delta: float) -> dict[str, float | str]:
    angles = np.asarray(ANGLES_DEG, dtype=np.float64)
    tavg_candidates = {
        "tavg_6": float(np.mean(torques)),
        "tavg_5": float(np.mean(torques[:-1])),
        "tavg_trapz": float(np.trapezoid(torques, angles) / 15.0),
    }
    peak_to_peak = float(np.max(torques) - np.min(torques))
    tavg6 = tavg_candidates["tavg_6"]
    delta_candidates = {
        "delta_abs": peak_to_peak,
        "delta_ratio": peak_to_peak / tavg6 if tavg6 else float("nan"),
        "delta_percent": 100.0 * peak_to_peak / tavg6 if tavg6 else float("nan"),
        "delta_half": peak_to_peak / (2.0 * tavg6) if tavg6 else float("nan"),
    }
    best_tavg_name = min(tavg_candidates, key=lambda key: abs(tavg_candidates[key] - historical_tavg))
    finite_delta = {key: value for key, value in delta_candidates.items() if np.isfinite(value)}
    best_delta_name = min(finite_delta, key=lambda key: abs(finite_delta[key] - historical_delta))

    output: dict[str, float | str] = {}
    output.update(tavg_candidates)
    output.update(delta_candidates)
    output.update(
        {
            "historical_tavg": historical_tavg,
            "historical_delta_t": historical_delta,
            "best_tavg_formula": best_tavg_name,
            "best_tavg_value": tavg_candidates[best_tavg_name],
            "best_tavg_absolute_error": abs(tavg_candidates[best_tavg_name] - historical_tavg),
            "best_tavg_relative_error": abs(tavg_candidates[best_tavg_name] - historical_tavg) / abs(historical_tavg) if historical_tavg else float("nan"),
            "best_delta_formula": best_delta_name,
            "best_delta_value": finite_delta[best_delta_name],
            "best_delta_absolute_error": abs(finite_delta[best_delta_name] - historical_delta),
            "best_delta_relative_error": abs(finite_delta[best_delta_name] - historical_delta) / abs(historical_delta) if historical_delta else float("nan"),
        }
    )
    return output


def run_worker(
    base_fem: Path,
    angle_dir: Path,
    angle: float,
    log_prefix: Path,
    timeout_seconds: float,
    maximum_attempts: int,
) -> dict[str, Any]:
    result_path = angle_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing

    last_error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        command = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--base-fem",
            str(base_fem),
            "--output-dir",
            str(angle_dir),
            "--angle-deg",
            str(angle),
            "--pole-pairs",
            str(POLE_PAIRS),
            "--amplitude-a",
            str(CURRENT_AMPLITUDE_A),
        ]
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            log_payload = {
                "attempt": attempt,
                "command": command,
                "worker_pid": process.pid,
                "returncode": process.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "stdout": stdout,
                "stderr": stderr,
            }
            atomic_json(log_prefix.with_name(log_prefix.name + f"_attempt_{attempt}.json"), log_payload)
            if process.returncode == 0 and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") == "complete":
                    return result
            last_error = RuntimeError(f"worker return code {process.returncode}")
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            cleanup = {"method": "process.kill", "returncode": None, "stdout": "", "stderr": ""}
            if os.name == "nt":
                killed = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                cleanup = {
                    "method": "taskkill /T /F",
                    "returncode": killed.returncode,
                    "stdout": killed.stdout,
                    "stderr": killed.stderr,
                }
            if process.poll() is None:
                process.kill()
            process.communicate()
            atomic_json(
                log_prefix.with_name(log_prefix.name + f"_attempt_{attempt}.json"),
                {
                    "attempt": attempt,
                    "command": command,
                    "status": "timeout",
                    "timeout_seconds": timeout_seconds,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                    "worker_pid": process.pid,
                    "process_tree_cleanup": cleanup,
                },
            )
    raise RuntimeError(f"FEMM angle failed after {maximum_attempts} attempts: {last_error}")


def write_curve(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_mechanical_deg", "angle_electrical_deg", "Ia_A", "Ib_A", "Ic_A", "torque_Nm", "status", "total_seconds"])
        for result in results:
            currents = result["phase_currents_a"]
            writer.writerow(
                [
                    result["angle_mechanical_deg"],
                    result["angle_electrical_deg"],
                    currents["A"],
                    currents["B"],
                    currents["C"],
                    result["torque_nm"],
                    result["status"],
                    result["timing_seconds"]["total"],
                ]
            )


def write_results(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(stage: str, mappings: list[str], timeout_seconds: float, maximum_attempts: int) -> None:
    candidates = load_candidates(stage)
    checkpoint = load_checkpoint()
    checkpoint["status"] = "running"
    checkpoint["stage"] = stage
    atomic_json(CHECKPOINT_PATH, checkpoint)
    result_rows: list[dict[str, Any]] = []

    generated_root = VALIDATION_ROOT / "generated_fem"
    curve_root = VALIDATION_ROOT / "torque_curves"
    log_root = VALIDATION_ROOT / "logs"
    generated_root.mkdir(parents=True, exist_ok=True)
    curve_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    try:
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            before_bits = np.fromiter((int(char) for char in candidate["gene_before_correction"]), dtype=np.uint8, count=200)
            after_bits = np.fromiter((int(char) for char in candidate["gene_after_correction"]), dtype=np.uint8, count=200)
            if not np.array_equal(before_bits, after_bits):
                raise AssertionError(f"{candidate_id} violates smoke-test equality requirement")
            for mapping_name in mappings:
                task_prefix = f"{candidate_id}/{mapping_name}"
                base_dir = generated_root / candidate_id / mapping_name / "base"
                base_fem = base_dir / "model.fem"
                if not base_fem.exists():
                    build_topology(
                        after_bits,
                        DEFAULT_TEMPLATE,
                        DEFAULT_MAT,
                        base_dir,
                        provenance={
                            "kind": "six_angle_validation_candidate",
                            "candidate_id": candidate_id,
                            "state_index_0based": int(candidate["state_index_0based"]),
                            "population_row_0based": int(candidate["individual_0based"]),
                            "before_after_equal": True,
                        },
                        code_to_class=MAPPING_CANDIDATES[mapping_name],
                        mapping_version=f"candidate_{mapping_name}_v1",
                    )
                angle_results: list[dict[str, Any]] = []
                for angle in ANGLES_DEG:
                    angle_key = f"angle_{int(angle):03d}"
                    task_key = f"{task_prefix}/{angle_key}"
                    angle_dir = generated_root / candidate_id / mapping_name / angle_key
                    log_prefix = log_root / f"{candidate_id}_{mapping_name}_{angle_key}"
                    checkpoint["tasks"].setdefault(task_key, {})
                    checkpoint["tasks"][task_key].update({"status": "running", "updated_at_epoch": time.time()})
                    atomic_json(CHECKPOINT_PATH, checkpoint)
                    try:
                        result = run_worker(base_fem, angle_dir, angle, log_prefix, timeout_seconds, maximum_attempts)
                    except Exception as exc:
                        checkpoint["tasks"][task_key].update(
                            {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc), "updated_at_epoch": time.time()}
                        )
                        checkpoint["status"] = "failed"
                        atomic_json(CHECKPOINT_PATH, checkpoint)
                        raise
                    checkpoint["tasks"][task_key].update(
                        {"status": "complete", "torque_nm": result["torque_nm"], "updated_at_epoch": time.time()}
                    )
                    atomic_json(CHECKPOINT_PATH, checkpoint)
                    angle_results.append(result)

                curve_path = curve_root / f"{candidate_id}_{mapping_name}.csv"
                write_curve(curve_path, angle_results)
                torques = np.asarray([result["torque_nm"] for result in angle_results], dtype=np.float64)
                metrics = result_metrics(
                    torques,
                    historical_tavg=float(candidate["historical_tavg"]),
                    historical_delta=float(candidate["historical_delta_t"]),
                )
                result_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "mapping_candidate": mapping_name,
                        "state_index_0based": int(candidate["state_index_0based"]),
                        "individual_0based": int(candidate["individual_0based"]),
                        "category": candidate["category"],
                        "gene_source": "before=after for selected candidate",
                        "angles_deg": ";".join(str(value) for value in ANGLES_DEG),
                        "torques_nm": ";".join(f"{value:.17g}" for value in torques),
                        **metrics,
                        "tavg_within_0_5_percent": bool(float(metrics["best_tavg_relative_error"]) < 0.005),
                        "delta_within_2_percent": bool(float(metrics["best_delta_relative_error"]) < 0.02),
                        "torque_curve_csv": str(curve_path.relative_to(VALIDATION_ROOT)),
                    }
                )
                write_results(result_rows)
        checkpoint["status"] = "smoke_complete" if stage == "smoke" else "complete"
        checkpoint["completed_result_rows"] = len(result_rows)
        atomic_json(CHECKPOINT_PATH, checkpoint)
    except KeyboardInterrupt:
        checkpoint["status"] = "paused_by_user"
        atomic_json(CHECKPOINT_PATH, checkpoint)
        raise

    print(f"Stage: {stage}; candidates={len(candidates)}; mappings={mappings}; result rows={len(result_rows)}")
    print(f"Results: {RESULTS_CSV}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--mappings", nargs="+", choices=sorted(MAPPING_CANDIDATES), default=sorted(MAPPING_CANDIDATES))
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    with validation_lock():
        run(args.stage, args.mappings, args.timeout_seconds, args.maximum_attempts)


if __name__ == "__main__":
    main()
