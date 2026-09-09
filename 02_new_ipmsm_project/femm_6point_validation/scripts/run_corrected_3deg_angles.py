"""Run the corrected candidate_002 topology at 0,3,...,15 mechanical degrees."""

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
sys.path.insert(0, str(VALIDATION_ROOT / "scripts"))
from run_validation import atomic_json, run_worker, validation_lock  # noqa: E402


ANGLES = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
BASE = VALIDATION_ROOT / "corrected_pm_direction" / "base" / "model.fem"
ROOT = VALIDATION_ROOT / "corrected_pm_direction" / "angles_3deg"
CHECKPOINT = ROOT / "checkpoint.json"
SUMMARY = ROOT / "analysis_summary.json"
CURVE = ROOT / "candidate_002_air0_angles_0_to_15.csv"
REUSE_15_PATH: Path | None = VALIDATION_ROOT / "corrected_pm_direction" / "angle_15" / "result.json"
VARIANT_DESCRIPTION = "reference nonlinear Pure Iron with 21-point BH curve"
MAPPING_NAME = "air0"


def history() -> tuple[float, float]:
    with (VALIDATION_ROOT / "selected_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(row for row in csv.DictReader(stream) if row["candidate_id"] == "candidate_002")
    return float(row["historical_tavg"]), float(row["historical_delta_t"])


def run(workers: int, timeout_seconds: float, maximum_attempts: int) -> None:
    manifest = json.loads((BASE.parent / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["validation"]["all_pm_magnetization_directions_match_template_rule"]:
        raise AssertionError("Corrected PM-direction base validation failed")
    ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = {"status": "running", "angles_mechanical_deg": list(ANGLES), "tasks": {}}
    if CHECKPOINT.exists():
        checkpoint["tasks"] = json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("tasks", {})
    futures = {}
    for angle in ANGLES:
        key = f"angle_{int(angle):03d}"
        # Reuse corrected 15-degree result from the 2.5-degree experiment.
        if angle == 15.0 and REUSE_15_PATH is not None:
            source = REUSE_15_PATH
            if source.exists() and json.loads(source.read_text(encoding="utf-8")).get("status") == "complete":
                checkpoint["tasks"][key] = {"status": "reused", "source": str(source)}
                continue
        output = ROOT / key
        result_path = output / "result.json"
        if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
            checkpoint["tasks"][key] = {"status": "complete_cached", "result": str(result_path)}
    atomic_json(CHECKPOINT, checkpoint)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for angle in ANGLES:
            key = f"angle_{int(angle):03d}"
            if checkpoint["tasks"].get(key, {}).get("status") == "reused":
                continue
            if checkpoint["tasks"].get(key, {}).get("status") == "complete_cached":
                continue
            output = ROOT / key
            log_prefix = ROOT / "logs" / f"candidate_002_air0_{key}"
            log_prefix.parent.mkdir(parents=True, exist_ok=True)
            checkpoint["tasks"][key] = {"status": "running", "updated_at_epoch": time.time()}
            future = executor.submit(run_worker, BASE, output, angle, log_prefix, timeout_seconds, maximum_attempts)
            futures[future] = key
        atomic_json(CHECKPOINT, checkpoint)
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                checkpoint["tasks"][key] = {"status": "complete", "torque_nm": result["torque_nm"]}
            except Exception as exc:
                checkpoint["tasks"][key] = {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}
                checkpoint["status"] = "failed"
                atomic_json(CHECKPOINT, checkpoint)
                raise
            atomic_json(CHECKPOINT, checkpoint)

    results = []
    for angle in ANGLES:
        reusable = angle == 15.0 and REUSE_15_PATH is not None and REUSE_15_PATH.exists()
        path = REUSE_15_PATH if reusable else ROOT / f"angle_{int(angle):03d}" / "result.json"
        results.append(json.loads(path.read_text(encoding="utf-8")))
    raw = np.asarray([float(row["torque_nm"]) for row in results])
    torque = -raw
    hist_tavg, hist_delta = history()
    avg = {
        "mean6": float(np.mean(torque)),
        "mean5_excluding_15": float(np.mean(torque[:-1])),
        "trapz_0_to_15": float(np.trapezoid(torque, np.asarray(ANGLES)) / 15.0),
    }
    peak = float(np.ptp(torque))
    summary = {
        "status": "complete",
        "candidate_id": "candidate_002",
        "mapping": MAPPING_NAME,
        "topology_fix": "all PM labels receive pole-dependent radial magnetization",
        "iron_material_variant": VARIANT_DESCRIPTION,
        "angles_mechanical_deg": list(ANGLES),
        "raw_femm_torques_nm": raw.tolist(),
        "sign_reversed_torques_nm": torque.tolist(),
        "historical_tavg_nm": hist_tavg,
        "tavg_candidates_nm": avg,
        "tavg_relative_errors": {name: abs(value - hist_tavg) / abs(hist_tavg) for name, value in avg.items()},
        "historical_delta_t": hist_delta,
        "delta_absolute_peak_to_peak": peak,
        "delta_absolute_peak_to_peak_relative_error": abs(peak - hist_delta) / abs(hist_delta),
    }
    with CURVE.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_mechanical_deg", "angle_electrical_deg", "Ia_A", "Ib_A", "Ic_A", "raw_femm_torque_nm"])
        for result in results:
            current = result["phase_currents_a"]
            writer.writerow([result["angle_mechanical_deg"], result["angle_electrical_deg"], current["A"], current["B"], current["C"], result["torque_nm"]])
    atomic_json(SUMMARY, summary)
    checkpoint["status"] = "complete"
    checkpoint["summary"] = str(SUMMARY)
    atomic_json(CHECKPOINT, checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use 1..3 workers")
    with validation_lock():
        run(args.workers, args.timeout_seconds, args.maximum_attempts)


if __name__ == "__main__":
    main()
