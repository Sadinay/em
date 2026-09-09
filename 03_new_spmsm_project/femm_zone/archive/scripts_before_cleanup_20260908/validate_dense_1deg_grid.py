"""Run the six selected SPMSM genes from 0 to 15 mechanical degrees in 1-degree steps."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from validate_history_replay import (
    CURRENT_AMPLITUDE_A,
    DEFAULT_OUTPUT,
    POLE_PAIRS,
    angle_key,
    atomic_json,
    run_one,
)


ANGLES = tuple(float(value) for value in range(16))


def run(output_root: Path, workers: int, timeout_seconds: float, maximum_attempts: int) -> dict:
    selection = json.loads((output_root / "selected_candidates.json").read_text(encoding="utf-8"))["candidates"]
    results_by_candidate: dict[str, dict[float, dict]] = {item["candidate_id"]: {} for item in selection}
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for candidate in selection:
            candidate_id = candidate["candidate_id"]
            base_fem = output_root / candidate_id / "base" / "model.fem"
            for angle in ANGLES:
                future = executor.submit(
                    run_one,
                    base_fem,
                    output_root / candidate_id / angle_key(angle),
                    angle,
                    timeout_seconds,
                    maximum_attempts,
                )
                futures[future] = (candidate_id, angle)
        for future in as_completed(futures):
            candidate_id, angle = futures[future]
            results_by_candidate[candidate_id][angle] = future.result()
            print(f"complete: {candidate_id}, angle={angle:g} deg", flush=True)

    comparisons = []
    long_rows = []
    angle_array = np.asarray(ANGLES, dtype=np.float64)
    for candidate in selection:
        candidate_id = candidate["candidate_id"]
        raw = np.asarray([results_by_candidate[candidate_id][angle]["torque_nm"] for angle in ANGLES], dtype=np.float64)
        torque = -2 * raw
        tavg_trapezoidal = float(np.trapezoid(torque, angle_array) / (angle_array[-1] - angle_array[0]))
        tavg_mean16 = float(np.mean(torque))
        peak_to_peak = float(np.ptp(torque))
        relative_ripple_trapezoidal = peak_to_peak / max(abs(tavg_trapezoidal), 1e-12)
        relative_ripple_mean16 = peak_to_peak / max(abs(tavg_mean16), 1e-12)
        historical_tavg = float(candidate["historical_tavg_nm"])
        historical_delta = float(candidate["historical_delta_t_nm"])
        comparisons.append(
            {
                "candidate_id": candidate_id,
                "state_index_0based": candidate["state_index_0based"],
                "population_row_0based": candidate["population_row_0based"],
                "pm_cells": candidate["pm_cells"],
                "historical_tavg_nm": historical_tavg,
                "recomputed_tavg_trapezoidal_nm": tavg_trapezoidal,
                "relative_error_tavg_trapezoidal": abs(tavg_trapezoidal - historical_tavg) / abs(historical_tavg),
                "recomputed_tavg_mean16_nm": tavg_mean16,
                "relative_error_tavg_mean16": abs(tavg_mean16 - historical_tavg) / abs(historical_tavg),
                "historical_delta_t": historical_delta,
                "recomputed_absolute_peak_to_peak_nm": peak_to_peak,
                "relative_error_if_historical_delta_is_absolute": abs(peak_to_peak - historical_delta) / abs(historical_delta),
                "recomputed_relative_peak_to_peak_trapezoidal": relative_ripple_trapezoidal,
                "relative_error_if_historical_delta_is_relative_trapezoidal": abs(relative_ripple_trapezoidal - historical_delta) / abs(historical_delta),
                "recomputed_relative_peak_to_peak_mean16": relative_ripple_mean16,
                "relative_error_if_historical_delta_is_relative_mean16": abs(relative_ripple_mean16 - historical_delta) / abs(historical_delta),
                "full_machine_sign_corrected_torques_nm": torque.tolist(),
                "raw_femm_torques_nm": raw.tolist(),
            }
        )
        for angle, raw_value, torque_value in zip(ANGLES, raw, torque, strict=True):
            long_rows.append(
                {
                    "candidate_id": candidate_id,
                    "state_index_0based": candidate["state_index_0based"],
                    "population_row_0based": candidate["population_row_0based"],
                    "angle_mechanical_deg": angle,
                    "raw_femm_torque_nm": raw_value,
                    "full_machine_sign_corrected_torque_nm": torque_value,
                }
            )

    summary = {
        "status": "complete",
        "interpretation": "16 samples from 0 through 15 mechanical degrees at 1-degree spacing",
        "angles_mechanical_deg": list(ANGLES),
        "pole_pairs": POLE_PAIRS,
        "current_amplitude_a": CURRENT_AMPLITUDE_A,
        "torque_multiplier": -2,
        "mean_relative_error_tavg_trapezoidal": float(np.mean([row["relative_error_tavg_trapezoidal"] for row in comparisons])),
        "mean_relative_error_tavg_mean16": float(np.mean([row["relative_error_tavg_mean16"] for row in comparisons])),
        "mean_relative_error_delta_if_absolute": float(np.mean([row["relative_error_if_historical_delta_is_absolute"] for row in comparisons])),
        "mean_relative_error_delta_if_relative_trapezoidal": float(np.mean([row["relative_error_if_historical_delta_is_relative_trapezoidal"] for row in comparisons])),
        "mean_relative_error_delta_if_relative_mean16": float(np.mean([row["relative_error_if_historical_delta_is_relative_mean16"] for row in comparisons])),
        "candidates": comparisons,
    }
    atomic_json(output_root / "dense_1deg_summary.json", summary)

    comparison_fields = [key for key in comparisons[0] if key not in {"full_machine_sign_corrected_torques_nm", "raw_femm_torques_nm"}]
    with (output_root / "dense_1deg_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({key: row[key] for key in comparison_fields})

    with (output_root / "dense_1deg_torque_curves.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    summary = run(args.output_root, args.workers, args.timeout_seconds, args.maximum_attempts)
    compact = [{key: value for key, value in row.items() if not key.endswith("torques_nm")} for row in summary["candidates"]]
    print(json.dumps({"summary": {key: value for key, value in summary.items() if key != "candidates"}, "candidates": compact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
