"""Test whether a common stator-current phase offset explains DeltaT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from validate_history_replay import ANGLES, DEFAULT_OUTPUT, atomic_json


CANDIDATE_IDS = ("median_tavg", "high_tavg", "final_median_tavg")
OFFSETS_ELECTRICAL_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)


def angle_tag(angle: float) -> str:
    return f"{angle:g}".replace("-", "m").replace(".", "p")


def solve_task(base_fem: Path, output_dir: Path, angle: float, offset: float, timeout: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "complete":
            return cached
    command = [
        sys.executable,
        str(Path(__file__).with_name("solve_one_angle.py")),
        "--base-fem", str(base_fem),
        "--output-dir", str(output_dir),
        "--angle-deg", str(angle),
        "--current-phase-offset-deg", str(offset),
        "--discard-ans",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"Phase diagnostic failed: returncode={completed.returncode}; {completed.stderr[-1000:]}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError(f"Incomplete result at {output_dir}")
    return result


def analyze(selection: list[dict], curves: dict[float, dict[str, list[float]]]) -> dict:
    by_id = {row["candidate_id"]: row for row in selection}
    configurations = []
    for offset in OFFSETS_ELECTRICAL_DEG:
        rows = []
        for candidate_id in CANDIDATE_IDS:
            # The validated mean-torque scaling is -2 for this supplied model.
            values = -2 * np.asarray(curves[offset][candidate_id], dtype=np.float64)
            tavg = float(np.mean(values))
            absolute_peak_to_peak = float(np.ptp(values))
            relative_peak_to_peak = absolute_peak_to_peak / max(abs(tavg), 1e-12)
            historical_tavg = float(by_id[candidate_id]["historical_tavg_nm"])
            historical_delta = float(by_id[candidate_id]["historical_delta_t_nm"])
            rows.append({
                "candidate_id": candidate_id,
                "historical_tavg": historical_tavg,
                "recomputed_tavg": tavg,
                "relative_error_tavg": abs(tavg - historical_tavg) / abs(historical_tavg),
                "historical_delta_t": historical_delta,
                "absolute_peak_to_peak": absolute_peak_to_peak,
                "relative_peak_to_peak": relative_peak_to_peak,
                "relative_error_if_delta_is_absolute": abs(absolute_peak_to_peak - historical_delta) / abs(historical_delta),
                "relative_error_if_delta_is_relative": abs(relative_peak_to_peak - historical_delta) / abs(historical_delta),
                "torques_nm": values.tolist(),
            })
        configurations.append({
            "current_phase_offset_electrical_deg": offset,
            "mean_relative_error_tavg": float(np.mean([r["relative_error_tavg"] for r in rows])),
            "mean_relative_error_delta_if_absolute": float(np.mean([r["relative_error_if_delta_is_absolute"] for r in rows])),
            "mean_relative_error_delta_if_relative": float(np.mean([r["relative_error_if_delta_is_relative"] for r in rows])),
            "candidates": rows,
        })
    return {
        "status": "complete",
        "candidate_ids": list(CANDIDATE_IDS),
        "angles_mechanical_deg": list(ANGLES),
        "torque_multiplier": -2,
        "tavg_method": "six-point arithmetic mean",
        "configuration_results": configurations,
    }


def run(output_root: Path, workers: int, timeout: float) -> dict:
    selection = json.loads((output_root / "selected_candidates.json").read_text(encoding="utf-8"))["candidates"]
    diagnostic_root = output_root / "current_phase_offset_diagnostics"
    results: dict[tuple[float, str, float], dict] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in OFFSETS_ELECTRICAL_DEG:
            for candidate_id in CANDIDATE_IDS:
                base_fem = output_root / candidate_id / "base" / "model.fem"
                for angle in ANGLES:
                    if offset == 0:
                        result_path = output_root / candidate_id / f"angle_{angle_tag(angle)}" / "result.json"
                        results[(offset, candidate_id, angle)] = json.loads(result_path.read_text(encoding="utf-8"))
                        continue
                    output_dir = diagnostic_root / f"offset_{angle_tag(offset)}" / candidate_id / f"angle_{angle_tag(angle)}"
                    future = executor.submit(solve_task, base_fem, output_dir, angle, offset, timeout)
                    futures[future] = (offset, candidate_id, angle)
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(f"complete: offset={key[0]:g}, {key[1]}, angle={key[2]:g}", flush=True)
    curves = {
        offset: {
            candidate_id: [results[(offset, candidate_id, angle)]["torque_nm"] for angle in ANGLES]
            for candidate_id in CANDIDATE_IDS
        }
        for offset in OFFSETS_ELECTRICAL_DEG
    }
    summary = analyze(selection, curves)
    atomic_json(diagnostic_root / "analysis_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    summary = run(args.output_root, args.workers, args.timeout_seconds)
    compact = [{k: v for k, v in row.items() if k != "candidates"} for row in summary["configuration_results"]]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
