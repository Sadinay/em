"""Diagnose rotor-angle and three-phase current direction conventions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from validate_history_replay import DEFAULT_OUTPUT, ANGLES, atomic_json


CANDIDATE_IDS = ("median_tavg", "high_tavg", "final_median_tavg")
CONFIGURATIONS = {
    "rotor_plus_current_minus": (1, -1),
    "rotor_minus_current_plus": (-1, 1),
    "rotor_minus_current_minus": (-1, -1),
}


def solve_task(base_fem: Path, output_dir: Path, angle: float, rotor_factor: int, current_factor: int, timeout: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "complete":
            return cached
    command = [
        sys.executable,
        str(Path(__file__).with_name("solve_one_angle.py")),
        "--base-fem",
        str(base_fem),
        "--output-dir",
        str(output_dir),
        "--angle-deg",
        str(angle),
        "--rotor-angle-factor",
        str(rotor_factor),
        "--current-angle-factor",
        str(current_factor),
        "--discard-ans",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"Direction diagnostic failed: returncode={completed.returncode}; {completed.stderr[-1000:]}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError(f"Incomplete result at {output_dir}")
    return result


def mean_value(values: np.ndarray, method: str) -> float:
    if method == "trapezoidal":
        return float(np.trapezoid(values, np.asarray(ANGLES)) / (ANGLES[-1] - ANGLES[0]))
    if method == "mean6":
        return float(np.mean(values))
    if method == "mean_first5":
        return float(np.mean(values[:-1]))
    raise ValueError(method)


def analyze(selection: list[dict], curves: dict[str, dict[str, list[float]]]) -> dict:
    by_id = {row["candidate_id"]: row for row in selection}
    configurations = []
    for config_id in CONFIGURATIONS:
        trials = []
        for multiplier in (1, -1, 2, -2, 4, -4):
            for method in ("trapezoidal", "mean6", "mean_first5"):
                rows = []
                for candidate_id in CANDIDATE_IDS:
                    values = multiplier * np.asarray(curves[config_id][candidate_id], dtype=np.float64)
                    historical_tavg = float(by_id[candidate_id]["historical_tavg_nm"])
                    historical_delta = float(by_id[candidate_id]["historical_delta_t_nm"])
                    predicted_tavg = mean_value(values, method)
                    predicted_delta = float(np.ptp(values))
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "historical_tavg_nm": historical_tavg,
                            "recomputed_tavg_nm": predicted_tavg,
                            "relative_error_tavg": abs(predicted_tavg - historical_tavg) / abs(historical_tavg),
                            "historical_delta_t_nm": historical_delta,
                            "recomputed_delta_t_nm": predicted_delta,
                            "relative_error_delta_t": abs(predicted_delta - historical_delta) / abs(historical_delta),
                            "scaled_torques_nm": values.tolist(),
                        }
                    )
                trials.append(
                    {
                        "torque_multiplier": multiplier,
                        "tavg_method": method,
                        "mean_relative_error_tavg": float(np.mean([row["relative_error_tavg"] for row in rows])),
                        "mean_relative_error_delta_t": float(np.mean([row["relative_error_delta_t"] for row in rows])),
                        "mean_combined_relative_error": float(
                            np.mean([row["relative_error_tavg"] + row["relative_error_delta_t"] for row in rows])
                        ),
                        "candidates": rows,
                    }
                )
        trials.sort(key=lambda row: row["mean_combined_relative_error"])
        configurations.append(
            {
                "configuration_id": config_id,
                "rotor_angle_factor": CONFIGURATIONS[config_id][0],
                "current_angle_factor": CONFIGURATIONS[config_id][1],
                "best_metric_trial": trials[0],
                "all_metric_trials": trials,
            }
        )
    configurations.sort(key=lambda row: row["best_metric_trial"]["mean_combined_relative_error"])
    return {
        "status": "complete",
        "candidate_ids": list(CANDIDATE_IDS),
        "angles_mechanical_deg": list(ANGLES),
        "configuration_results": configurations,
    }


def run(output_root: Path, workers: int, timeout: float) -> dict:
    selection = json.loads((output_root / "selected_candidates.json").read_text(encoding="utf-8"))["candidates"]
    diagnostic_root = output_root / "direction_diagnostics"
    futures = {}
    results: dict[tuple[str, str, float], dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for config_id, (rotor_factor, current_factor) in CONFIGURATIONS.items():
            for candidate_id in CANDIDATE_IDS:
                base_fem = output_root / candidate_id / "base" / "model.fem"
                baseline_zero = output_root / candidate_id / "angle_0" / "result.json"
                results[(config_id, candidate_id, 0.0)] = json.loads(baseline_zero.read_text(encoding="utf-8"))
                for angle in ANGLES[1:]:
                    future = executor.submit(
                        solve_task,
                        base_fem,
                        diagnostic_root / config_id / candidate_id / f"angle_{angle:g}".replace(".", "p"),
                        angle,
                        rotor_factor,
                        current_factor,
                        timeout,
                    )
                    futures[future] = (config_id, candidate_id, angle)
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(f"complete: {key[0]}, {key[1]}, angle={key[2]:g}", flush=True)
    curves = {
        config_id: {
            candidate_id: [results[(config_id, candidate_id, angle)]["torque_nm"] for angle in ANGLES]
            for candidate_id in CANDIDATE_IDS
        }
        for config_id in CONFIGURATIONS
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
    print(json.dumps(summary["configuration_results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
