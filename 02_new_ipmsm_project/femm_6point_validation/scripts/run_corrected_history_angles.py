"""Run candidate_002 after repairing PM directions at inferred 2.5-degree steps."""

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
sys.path.insert(0, str(PROJECT_ROOT / "femm_zone" / "scripts"))

from build_femm_topology import DEFAULT_MAT, DEFAULT_TEMPLATE, MAPPING_CANDIDATES, build_topology  # noqa: E402
from run_validation import atomic_json, run_worker, validation_lock  # noqa: E402


CANDIDATE_ID = "candidate_002"
MAPPING = "air0"
ANGLES = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
ROOT = VALIDATION_ROOT / "corrected_pm_direction"
BASE_DIR = ROOT / "base"
CHECKPOINT = ROOT / "checkpoint.json"
CURVE = ROOT / "candidate_002_air0_angles_2p5_to_15.csv"
SUMMARY = ROOT / "analysis_summary.json"


def candidate() -> dict[str, str]:
    with (VALIDATION_ROOT / "selected_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["candidate_id"] == CANDIDATE_ID:
                return row
    raise KeyError(CANDIDATE_ID)


def angle_key(angle: float) -> str:
    return f"angle_{angle:g}".replace(".", "p")


def analyze(rows: list[dict], selected: dict[str, str]) -> dict:
    raw = np.asarray([float(row["torque_nm"]) for row in rows])
    signed = -raw
    hist_tavg = float(selected["historical_tavg"])
    hist_delta = float(selected["historical_delta_t"])
    tavg = float(np.mean(signed))
    delta = float(np.ptp(signed))
    return {
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "mapping": MAPPING,
        "topology_fix": "all PM labels receive pole-dependent radial magnetization",
        "angles_mechanical_deg": list(ANGLES),
        "raw_femm_torques_nm": raw.tolist(),
        "sign_reversed_torques_nm": signed.tolist(),
        "historical_tavg_nm": hist_tavg,
        "recomputed_tavg_mean6_nm": tavg,
        "absolute_error_tavg_nm": abs(tavg - hist_tavg),
        "relative_error_tavg": abs(tavg - hist_tavg) / abs(hist_tavg),
        "historical_delta_t": hist_delta,
        "recomputed_delta_t_absolute_peak_to_peak": delta,
        "absolute_error_delta_t": abs(delta - hist_delta),
        "relative_error_delta_t": abs(delta - hist_delta) / abs(hist_delta),
        "tavg_within_0_5_percent": abs(tavg - hist_tavg) / abs(hist_tavg) < 0.005,
        "delta_within_2_percent": abs(delta - hist_delta) / abs(hist_delta) < 0.02,
    }


def run(workers: int, timeout_seconds: float, maximum_attempts: int) -> None:
    selected = candidate()
    before = np.fromiter((int(char) for char in selected["gene_before_correction"]), dtype=np.uint8, count=200)
    after = np.fromiter((int(char) for char in selected["gene_after_correction"]), dtype=np.uint8, count=200)
    if not np.array_equal(before, after):
        raise AssertionError("The controlled candidate must have identical before/after genes")
    ROOT.mkdir(parents=True, exist_ok=True)
    base_fem = build_topology(
        after,
        DEFAULT_TEMPLATE,
        DEFAULT_MAT,
        BASE_DIR,
        provenance={
            "kind": "corrected_pm_direction_history_angle_test",
            "candidate_id": CANDIDATE_ID,
            "state_index_0based": int(selected["state_index_0based"]),
            "individual_0based": int(selected["individual_0based"]),
        },
        code_to_class=MAPPING_CANDIDATES[MAPPING],
        mapping_version="candidate_air0_pm_direction_fixed_v2",
    )
    manifest = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["validation"]["all_pm_magnetization_directions_match_template_rule"]:
        raise AssertionError("PM direction validation failed before FEMM")

    checkpoint = {
        "status": "running",
        "candidate_id": CANDIDATE_ID,
        "mapping": MAPPING,
        "angles_mechanical_deg": list(ANGLES),
        "base_manifest": str(BASE_DIR / "manifest.json"),
        "tasks": {},
    }
    if CHECKPOINT.exists():
        checkpoint["tasks"] = json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("tasks", {})
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for angle in ANGLES:
            key = angle_key(angle)
            output_dir = ROOT / key
            result_path = output_dir / "result.json"
            if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
                checkpoint["tasks"][key] = {"status": "complete_cached", "result": str(result_path)}
                continue
            checkpoint["tasks"][key] = {"status": "running", "updated_at_epoch": time.time()}
            log_prefix = ROOT / "logs" / f"{CANDIDATE_ID}_{MAPPING}_{key}"
            log_prefix.parent.mkdir(parents=True, exist_ok=True)
            future = executor.submit(run_worker, base_fem, output_dir, angle, log_prefix, timeout_seconds, maximum_attempts)
            futures[future] = (key, result_path)
        atomic_json(CHECKPOINT, checkpoint)
        for future in as_completed(futures):
            key, result_path = futures[future]
            try:
                result = future.result()
                checkpoint["tasks"][key] = {"status": "complete", "torque_nm": result["torque_nm"]}
            except Exception as exc:
                checkpoint["tasks"][key] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                checkpoint["status"] = "failed"
                atomic_json(CHECKPOINT, checkpoint)
                raise
            atomic_json(CHECKPOINT, checkpoint)

    results = [json.loads((ROOT / angle_key(angle) / "result.json").read_text(encoding="utf-8")) for angle in ANGLES]
    with CURVE.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_mechanical_deg", "angle_electrical_deg", "Ia_A", "Ib_A", "Ic_A", "raw_femm_torque_nm"])
        for result in results:
            current = result["phase_currents_a"]
            writer.writerow(
                [result["angle_mechanical_deg"], result["angle_electrical_deg"], current["A"], current["B"], current["C"], result["torque_nm"]]
            )
    summary = analyze(results, selected)
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
        raise ValueError("Use 1..3 isolated FEMM workers")
    with validation_lock():
        run(args.workers, args.timeout_seconds, args.maximum_attempts)


if __name__ == "__main__":
    main()
