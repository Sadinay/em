"""Targeted 3..30 degree test for candidate_002/air0 only.

The already completed 3..15 degree results are reused.  Only 18, 21, 24,
27, and 30 mechanical degrees are submitted to FEMM.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_validation import VALIDATION_ROOT, atomic_json, run_worker, validation_lock


CANDIDATE_ID = "candidate_002"
MAPPING = "air0"
ANGLES = tuple(float(value) for value in range(3, 31, 3))
BASE_FEM = VALIDATION_ROOT / "generated_fem" / CANDIDATE_ID / MAPPING / "base" / "model.fem"
EXTENDED_ROOT = VALIDATION_ROOT / "extended_30deg"
CHECKPOINT = EXTENDED_ROOT / "checkpoint.json"
CURVE_CSV = EXTENDED_ROOT / "candidate_002_air0_3_to_30deg.csv"
SUMMARY_JSON = EXTENDED_ROOT / "analysis_summary.json"


def selected_history() -> tuple[float, float]:
    with (VALIDATION_ROOT / "selected_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["candidate_id"] == CANDIDATE_ID:
                return float(row["historical_tavg"]), float(row["historical_delta_t"])
    raise KeyError(CANDIDATE_ID)


def result_path(angle: float) -> Path:
    if angle <= 15.0:
        return VALIDATION_ROOT / "generated_fem" / CANDIDATE_ID / MAPPING / f"angle_{int(angle):03d}" / "result.json"
    return EXTENDED_ROOT / CANDIDATE_ID / MAPPING / f"angle_{int(angle):03d}" / "result.json"


def analyze(results: list[dict]) -> dict:
    historical_tavg, historical_delta = selected_history()
    raw = np.asarray([float(item["torque_nm"]) for item in results])
    signed = -raw
    first_period = signed[:5]
    second_period = signed[5:]
    pair_difference = second_period - first_period
    pair_abs = np.abs(pair_difference)
    scale = max(float(np.ptp(signed)), 1e-12)

    candidates_tavg = {
        "mean_10_points_3_to_30": float(np.mean(signed)),
        "mean_first_5_points_3_to_15": float(np.mean(first_period)),
        "mean_second_5_points_18_to_30": float(np.mean(second_period)),
        "trapz_3_to_30": float(np.trapezoid(signed, np.asarray(ANGLES)) / 27.0),
    }
    candidates_delta = {
        "peak_to_peak_10_points": float(np.ptp(signed)),
        "peak_to_peak_first_5": float(np.ptp(first_period)),
        "peak_to_peak_second_5": float(np.ptp(second_period)),
    }
    return {
        "candidate_id": CANDIDATE_ID,
        "mapping": MAPPING,
        "torque_sign_multiplier_for_history_comparison": -1,
        "angles_mechanical_deg": list(ANGLES),
        "raw_femm_torques_nm": raw.tolist(),
        "sign_reversed_torques_nm": signed.tolist(),
        "historical_tavg_nm": historical_tavg,
        "historical_delta_t": historical_delta,
        "tavg_candidates_nm": candidates_tavg,
        "tavg_relative_errors": {
            name: abs(value - historical_tavg) / abs(historical_tavg) for name, value in candidates_tavg.items()
        },
        "delta_candidates": candidates_delta,
        "delta_relative_errors": {
            name: abs(value - historical_delta) / abs(historical_delta) for name, value in candidates_delta.items()
        },
        "fifteen_degree_periodicity": {
            "pairs_deg": [[int(angle), int(angle + 15)] for angle in ANGLES[:5]],
            "signed_pair_differences_nm": pair_difference.tolist(),
            "absolute_pair_differences_nm": pair_abs.tolist(),
            "mean_absolute_pair_difference_nm": float(np.mean(pair_abs)),
            "max_absolute_pair_difference_nm": float(np.max(pair_abs)),
            "rmse_pair_difference_nm": float(np.sqrt(np.mean(pair_difference**2))),
            "rmse_normalized_by_10_point_range": float(np.sqrt(np.mean(pair_difference**2)) / scale),
            "periodic_within_1_percent_of_range_all_pairs": bool(np.all(pair_abs <= 0.01 * scale)),
        },
    }


def run(workers: int, timeout_seconds: float, maximum_attempts: int) -> None:
    if not BASE_FEM.exists():
        raise FileNotFoundError(BASE_FEM)
    EXTENDED_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "status": "running",
        "candidate_id": CANDIDATE_ID,
        "mapping": MAPPING,
        "angles_mechanical_deg": list(ANGLES),
        "reused_angles_deg": [value for value in ANGLES if value <= 15.0],
        "new_femm_angles_deg": [value for value in ANGLES if value > 15.0],
        "tasks": {},
    }
    if CHECKPOINT.exists():
        previous = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        checkpoint["tasks"] = previous.get("tasks", {})
    atomic_json(CHECKPOINT, checkpoint)

    for angle in ANGLES[:5]:
        path = result_path(angle)
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("status") != "complete":
            raise RuntimeError(f"Required cached result is missing or incomplete: {path}")
        checkpoint["tasks"][f"angle_{int(angle):03d}"] = {"status": "reused", "source": str(path)}
    atomic_json(CHECKPOINT, checkpoint)

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for angle in ANGLES[5:]:
            path = result_path(angle)
            if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete":
                checkpoint["tasks"][f"angle_{int(angle):03d}"] = {"status": "complete_cached", "result": str(path)}
                continue
            angle_dir = path.parent
            log_prefix = EXTENDED_ROOT / "logs" / f"{CANDIDATE_ID}_{MAPPING}_angle_{int(angle):03d}"
            log_prefix.parent.mkdir(parents=True, exist_ok=True)
            checkpoint["tasks"][f"angle_{int(angle):03d}"] = {"status": "running", "updated_at_epoch": time.time()}
            future = executor.submit(
                run_worker,
                BASE_FEM,
                angle_dir,
                angle,
                log_prefix,
                timeout_seconds,
                maximum_attempts,
            )
            futures[future] = angle
        atomic_json(CHECKPOINT, checkpoint)

        for future in as_completed(futures):
            angle = futures[future]
            key = f"angle_{int(angle):03d}"
            try:
                result = future.result()
                checkpoint["tasks"][key] = {
                    "status": "complete",
                    "torque_nm": result["torque_nm"],
                    "updated_at_epoch": time.time(),
                }
            except Exception as exc:
                checkpoint["tasks"][key] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "updated_at_epoch": time.time(),
                }
                checkpoint["status"] = "failed"
                atomic_json(CHECKPOINT, checkpoint)
                raise
            atomic_json(CHECKPOINT, checkpoint)

    results = [json.loads(result_path(angle).read_text(encoding="utf-8")) for angle in ANGLES]
    with CURVE_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_mechanical_deg", "angle_electrical_deg", "Ia_A", "Ib_A", "Ic_A", "raw_femm_torque_nm"])
        for item in results:
            current = item["phase_currents_a"]
            writer.writerow(
                [
                    item["angle_mechanical_deg"],
                    item["angle_electrical_deg"],
                    current["A"],
                    current["B"],
                    current["C"],
                    item["torque_nm"],
                ]
            )
    summary = analyze(results)
    atomic_json(SUMMARY_JSON, summary)
    checkpoint["status"] = "complete"
    checkpoint["summary"] = str(SUMMARY_JSON)
    atomic_json(CHECKPOINT, checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("This targeted validation permits 1..3 workers")
    with validation_lock():
        run(args.workers, args.timeout_seconds, args.maximum_attempts)


if __name__ == "__main__":
    main()
