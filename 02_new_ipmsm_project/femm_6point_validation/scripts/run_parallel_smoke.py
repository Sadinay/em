"""Run the frozen two-candidate smoke test with isolated parallel FEMM workers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


VALIDATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VALIDATION_ROOT.parent
sys.path.insert(0, str(VALIDATION_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "femm_zone" / "scripts"))

from build_femm_topology import DEFAULT_MAT, DEFAULT_TEMPLATE, MAPPING_CANDIDATES, build_topology  # noqa: E402
from run_validation import (  # noqa: E402
    ANGLES_DEG,
    CHECKPOINT_PATH,
    RESULTS_CSV,
    atomic_json,
    load_candidates,
    load_checkpoint,
    result_metrics,
    run_worker,
    validation_lock,
    write_curve,
    write_results,
)


def run(workers: int, timeout_seconds: float, maximum_attempts: int) -> None:
    candidates = load_candidates("smoke")
    mappings = sorted(MAPPING_CANDIDATES)
    generated_root = VALIDATION_ROOT / "generated_fem"
    curve_root = VALIDATION_ROOT / "torque_curves"
    log_root = VALIDATION_ROOT / "logs"
    for path in (generated_root, curve_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()
    checkpoint.update({"status": "running_parallel_smoke", "stage": "smoke", "workers": workers})
    tasks: list[dict] = []
    combinations: dict[tuple[str, str], tuple[dict, Path]] = {}

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        before_bits = np.fromiter((int(char) for char in candidate["gene_before_correction"]), dtype=np.uint8, count=200)
        after_bits = np.fromiter((int(char) for char in candidate["gene_after_correction"]), dtype=np.uint8, count=200)
        if not np.array_equal(before_bits, after_bits):
            raise AssertionError(f"{candidate_id}: before/after genes differ")
        for mapping_name in mappings:
            base_dir = generated_root / candidate_id / mapping_name / "base"
            base_fem = base_dir / "model.fem"
            # Always rebuild the base deterministically to prevent stale mapping files.
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
            combinations[(candidate_id, mapping_name)] = (candidate, base_fem)
            for angle in ANGLES_DEG:
                angle_key = f"angle_{int(angle):03d}"
                task_key = f"{candidate_id}/{mapping_name}/{angle_key}"
                tasks.append(
                    {
                        "task_key": task_key,
                        "candidate_id": candidate_id,
                        "mapping_name": mapping_name,
                        "angle": angle,
                        "base_fem": base_fem,
                        "angle_dir": generated_root / candidate_id / mapping_name / angle_key,
                        "log_prefix": log_root / f"{candidate_id}_{mapping_name}_{angle_key}",
                    }
                )
                checkpoint["tasks"].setdefault(task_key, {})
                result_path = generated_root / candidate_id / mapping_name / angle_key / "result.json"
                complete = False
                if result_path.exists():
                    try:
                        complete = json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete"
                    except Exception:
                        complete = False
                checkpoint["tasks"][task_key]["status"] = "complete" if complete else "queued"
    atomic_json(CHECKPOINT_PATH, checkpoint)

    failures: list[tuple[str, str]] = []
    future_to_task = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for task in tasks:
            result_path = task["angle_dir"] / "result.json"
            if result_path.exists():
                try:
                    if json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
                        continue
                except Exception:
                    pass
            checkpoint["tasks"][task["task_key"]].update({"status": "running", "updated_at_epoch": time.time()})
            future = executor.submit(
                run_worker,
                task["base_fem"],
                task["angle_dir"],
                task["angle"],
                task["log_prefix"],
                timeout_seconds,
                maximum_attempts,
            )
            future_to_task[future] = task
        atomic_json(CHECKPOINT_PATH, checkpoint)

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                checkpoint["tasks"][task["task_key"]].update(
                    {"status": "complete", "torque_nm": result["torque_nm"], "updated_at_epoch": time.time()}
                )
            except Exception as exc:
                checkpoint["tasks"][task["task_key"]].update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "updated_at_epoch": time.time(),
                    }
                )
                failures.append((task["task_key"], str(exc)))
            atomic_json(CHECKPOINT_PATH, checkpoint)

    if failures:
        checkpoint["status"] = "failed"
        checkpoint["failures"] = failures
        atomic_json(CHECKPOINT_PATH, checkpoint)
        raise RuntimeError(f"{len(failures)} angle tasks failed; see checkpoint and logs")

    output_rows = []
    for (candidate_id, mapping_name), (candidate, _) in combinations.items():
        angle_results = []
        for angle in ANGLES_DEG:
            result_path = generated_root / candidate_id / mapping_name / f"angle_{int(angle):03d}" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                raise RuntimeError(f"Incomplete result: {result_path}")
            angle_results.append(result)
        curve_path = curve_root / f"{candidate_id}_{mapping_name}.csv"
        write_curve(curve_path, angle_results)
        torques = np.asarray([result["torque_nm"] for result in angle_results], dtype=np.float64)
        metrics = result_metrics(torques, float(candidate["historical_tavg"]), float(candidate["historical_delta_t"]))
        output_rows.append(
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
    write_results(output_rows)
    checkpoint.update({"status": "smoke_complete", "completed_result_rows": len(output_rows)})
    atomic_json(CHECKPOINT_PATH, checkpoint)
    print(f"Smoke complete: {len(tasks)} angle tasks, {len(output_rows)} candidate/mapping rows")
    print(f"Results: {RESULTS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Smoke validation permits 1..3 FEMM workers")
    with validation_lock():
        run(args.workers, args.timeout_seconds, args.maximum_attempts)


if __name__ == "__main__":
    main()
