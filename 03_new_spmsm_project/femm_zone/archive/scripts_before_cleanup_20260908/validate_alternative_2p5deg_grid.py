"""Check the six non-duplicated samples of a 15-degree SPMSM torque period."""

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


ANGLES = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5)


def run(output_root: Path, workers: int, timeout_seconds: float, maximum_attempts: int) -> dict:
    selection = json.loads((output_root / "selected_candidates.json").read_text(encoding="utf-8"))["candidates"]
    futures = {}
    rows_by_candidate: dict[str, dict[float, dict]] = {item["candidate_id"]: {} for item in selection}
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
            rows_by_candidate[candidate_id][angle] = future.result()
            print(f"complete: {candidate_id}, angle={angle:g} deg", flush=True)

    comparisons = []
    for candidate in selection:
        candidate_id = candidate["candidate_id"]
        raw = np.asarray([rows_by_candidate[candidate_id][angle]["torque_nm"] for angle in ANGLES], dtype=np.float64)
        full_machine = -2 * raw
        recomputed_tavg = float(np.mean(full_machine))
        recomputed_delta = float(np.ptp(full_machine))
        historical_tavg = float(candidate["historical_tavg_nm"])
        historical_delta = float(candidate["historical_delta_t_nm"])
        comparisons.append(
            {
                "candidate_id": candidate_id,
                "state_index_0based": candidate["state_index_0based"],
                "population_row_0based": candidate["population_row_0based"],
                "pm_cells": candidate["pm_cells"],
                "historical_tavg_nm": historical_tavg,
                "recomputed_tavg_nm": recomputed_tavg,
                "relative_error_tavg": abs(recomputed_tavg - historical_tavg) / abs(historical_tavg),
                "historical_delta_t_nm": historical_delta,
                "recomputed_delta_t_nm": recomputed_delta,
                "relative_error_delta_t": abs(recomputed_delta - historical_delta) / abs(historical_delta),
                "raw_femm_torques_nm": raw.tolist(),
                "full_machine_sign_corrected_torques_nm": full_machine.tolist(),
            }
        )
    summary = {
        "status": "complete",
        "interpretation": "six non-duplicated uniform samples over one 15-degree torque period",
        "angles_mechanical_deg": list(ANGLES),
        "pole_pairs": POLE_PAIRS,
        "current_amplitude_a": CURRENT_AMPLITUDE_A,
        "torque_multiplier": -2,
        "tavg_method": "arithmetic mean of six periodic samples",
        "delta_t_method": "absolute peak-to-peak",
        "mean_relative_error_tavg": float(np.mean([row["relative_error_tavg"] for row in comparisons])),
        "mean_relative_error_delta_t": float(np.mean([row["relative_error_delta_t"] for row in comparisons])),
        "candidates": comparisons,
    }
    atomic_json(output_root / "alternative_2p5deg_summary.json", summary)
    with (output_root / "alternative_2p5deg_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "candidate_id",
            "state_index_0based",
            "population_row_0based",
            "pm_cells",
            "historical_tavg_nm",
            "recomputed_tavg_nm",
            "relative_error_tavg",
            "historical_delta_t_nm",
            "recomputed_delta_t_nm",
            "relative_error_delta_t",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({key: row[key] for key in fields})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    summary = run(args.output_root, args.workers, args.timeout_seconds, args.maximum_attempts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
