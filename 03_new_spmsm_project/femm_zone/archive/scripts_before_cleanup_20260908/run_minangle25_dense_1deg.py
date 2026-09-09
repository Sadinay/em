"""Create MinAngle=25 SPMSM variants, run the dense scan, and compare with MinAngle=15."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from validate_dense_1deg_grid import run as run_dense
from validate_history_replay import DEFAULT_OUTPUT, ROOT, atomic_json


DEFAULT_TARGET = ROOT / "femm_zone" / "results" / "history_replay_validation_minangle25"


def prepare_models(source_root: Path, target_root: Path) -> list[dict]:
    selection = json.loads((source_root / "selected_candidates.json").read_text(encoding="utf-8"))["candidates"]
    target_root.mkdir(parents=True, exist_ok=True)
    atomic_json(target_root / "selected_candidates.json", {"candidates": selection})
    for candidate in selection:
        candidate_id = candidate["candidate_id"]
        source = source_root / candidate_id / "base" / "model.fem"
        target = target_root / candidate_id / "base" / "model.fem"
        text = source.read_text(encoding="utf-8")
        updated, replacements = re.subn(
            r"(?m)^\[MinAngle\]\s*=\s*[^\r\n]+",
            "[MinAngle]    =  25",
            text,
        )
        if replacements != 1:
            raise RuntimeError(f"Expected one MinAngle field in {source}, found {replacements}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8")
    return selection


def compare(source_root: Path, target_root: Path) -> dict:
    old = json.loads((source_root / "dense_1deg_summary.json").read_text(encoding="utf-8"))
    new = json.loads((target_root / "dense_1deg_summary.json").read_text(encoding="utf-8"))
    old_by_id = {row["candidate_id"]: row for row in old["candidates"]}
    rows = []
    for current in new["candidates"]:
        candidate_id = current["candidate_id"]
        previous = old_by_id[candidate_id]
        old_curve = np.asarray(previous["full_machine_sign_corrected_torques_nm"], dtype=np.float64)
        new_curve = np.asarray(current["full_machine_sign_corrected_torques_nm"], dtype=np.float64)
        old_tavg = float(previous["recomputed_tavg_mean16_nm"])
        new_tavg = float(current["recomputed_tavg_mean16_nm"])
        old_ptp = float(previous["recomputed_absolute_peak_to_peak_nm"])
        new_ptp = float(current["recomputed_absolute_peak_to_peak_nm"])
        old_relative = float(previous["recomputed_relative_peak_to_peak_mean16"])
        new_relative = float(current["recomputed_relative_peak_to_peak_mean16"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "state_index_0based": current["state_index_0based"],
                "population_row_0based": current["population_row_0based"],
                "historical_tavg_nm": current["historical_tavg_nm"],
                "historical_delta_t": current["historical_delta_t"],
                "minangle15_tavg_mean16_nm": old_tavg,
                "minangle25_tavg_mean16_nm": new_tavg,
                "tavg_change_nm": new_tavg - old_tavg,
                "tavg_change_percent_of_minangle15": 100 * (new_tavg - old_tavg) / abs(old_tavg),
                "minangle15_absolute_peak_to_peak_nm": old_ptp,
                "minangle25_absolute_peak_to_peak_nm": new_ptp,
                "absolute_peak_to_peak_change_nm": new_ptp - old_ptp,
                "absolute_peak_to_peak_change_percent_of_minangle15": 100 * (new_ptp - old_ptp) / abs(old_ptp),
                "minangle15_relative_peak_to_peak": old_relative,
                "minangle25_relative_peak_to_peak": new_relative,
                "relative_peak_to_peak_change": new_relative - old_relative,
                "max_absolute_pointwise_torque_change_nm": float(np.max(np.abs(new_curve - old_curve))),
                "mean_absolute_pointwise_torque_change_nm": float(np.mean(np.abs(new_curve - old_curve))),
            }
        )
    summary = {
        "status": "complete",
        "source_min_angle_deg": 15,
        "target_min_angle_deg": 25,
        "angles_mechanical_deg": new["angles_mechanical_deg"],
        "mean_absolute_tavg_change_nm": float(np.mean([abs(row["tavg_change_nm"]) for row in rows])),
        "mean_absolute_peak_to_peak_change_nm": float(np.mean([abs(row["absolute_peak_to_peak_change_nm"]) for row in rows])),
        "mean_max_absolute_pointwise_torque_change_nm": float(np.mean([row["max_absolute_pointwise_torque_change_nm"] for row in rows])),
        "minangle25_metrics": {key: value for key, value in new.items() if key.startswith("mean_relative_error_")},
        "candidates": rows,
    }
    atomic_json(target_root / "minangle15_vs25_comparison.json", summary)
    with (target_root / "minangle15_vs25_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    prepare_models(args.source_root, args.target_root)
    run_dense(args.target_root, args.workers, args.timeout_seconds, args.maximum_attempts)
    result = compare(args.source_root, args.target_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
