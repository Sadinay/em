"""Recompute every requested scalar definition and rank uniform configurations.

The ranking is deliberately global: one material mapping, torque sign, average
definition, and ripple definition must be used for every smoke-test sample.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANGLES_DEG = np.asarray([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference) if reference else float("nan")


def average_candidates(torques: np.ndarray) -> dict[str, float]:
    return {
        "mean_6": float(np.mean(torques)),
        "mean_5_excluding_15_deg": float(np.mean(torques[:-1])),
        "trapz_0_to_15_deg": float(np.trapezoid(torques, ANGLES_DEG) / 15.0),
    }


def delta_candidates(torques: np.ndarray, average: float) -> dict[str, float]:
    peak_to_peak = float(np.max(torques) - np.min(torques))
    if average == 0.0:
        ratio = percent = half = float("nan")
    else:
        ratio = peak_to_peak / average
        percent = 100.0 * ratio
        half = peak_to_peak / (2.0 * average)
    return {
        "absolute_peak_to_peak": peak_to_peak,
        "peak_to_peak_over_average": ratio,
        "percent_peak_to_peak_over_average": percent,
        "half_peak_to_peak_over_average": half,
    }


def load_selected() -> dict[str, dict[str, str]]:
    with (ROOT / "selected_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {row["candidate_id"]: row for row in csv.DictReader(stream) if row["run_stage"] == "smoke"}


def load_curves() -> dict[tuple[str, str], np.ndarray]:
    curves: dict[tuple[str, str], np.ndarray] = {}
    for path in sorted((ROOT / "torque_curves").glob("candidate_*_air*.csv")):
        stem = path.stem
        candidate_id, mapping = stem.rsplit("_", 1)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        angles = np.asarray([float(row["angle_mechanical_deg"]) for row in rows])
        if not np.array_equal(angles, ANGLES_DEG):
            raise ValueError(f"Unexpected angles in {path}: {angles.tolist()}")
        curves[(candidate_id, mapping)] = np.asarray([float(row["torque_Nm"]) for row in rows])
    return curves


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    selected = load_selected()
    curves = load_curves()
    expected = {(candidate_id, mapping) for candidate_id in selected for mapping in ("air0", "air2", "air3")}
    missing = expected - set(curves)
    if missing:
        raise FileNotFoundError(f"Missing smoke curves: {sorted(missing)}")

    detailed: list[dict[str, object]] = []
    for (candidate_id, mapping), raw_torques in sorted(curves.items()):
        history = selected[candidate_id]
        historical_tavg = float(history["historical_tavg"])
        historical_delta = float(history["historical_delta_t"])
        for torque_sign in (1, -1):
            torques = torque_sign * raw_torques
            for average_name, average in average_candidates(torques).items():
                for delta_name, delta in delta_candidates(torques, average).items():
                    detailed.append(
                        {
                            "candidate_id": candidate_id,
                            "mapping_candidate": mapping,
                            "torque_sign_multiplier": torque_sign,
                            "average_formula": average_name,
                            "delta_formula": delta_name,
                            "angles_mechanical_deg": ";".join(f"{value:g}" for value in ANGLES_DEG),
                            "raw_femm_torques_nm": ";".join(f"{value:.17g}" for value in raw_torques),
                            "historical_tavg_nm": historical_tavg,
                            "recomputed_tavg_nm": average,
                            "absolute_error_tavg_nm": abs(average - historical_tavg),
                            "relative_error_tavg": relative_error(average, historical_tavg),
                            "historical_delta_t": historical_delta,
                            "recomputed_delta_t": delta,
                            "absolute_error_delta_t": abs(delta - historical_delta),
                            "relative_error_delta_t": relative_error(delta, historical_delta),
                            "tavg_within_0_5_percent": bool(relative_error(average, historical_tavg) < 0.005),
                            "delta_within_2_percent": bool(relative_error(delta, historical_delta) < 0.02),
                        }
                    )
    write_csv(ROOT / "validation_results.csv", detailed)

    config_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in detailed:
        key = (
            row["mapping_candidate"],
            row["torque_sign_multiplier"],
            row["average_formula"],
            row["delta_formula"],
        )
        config_groups.setdefault(key, []).append(row)

    ranking: list[dict[str, object]] = []
    for key, rows in config_groups.items():
        if len(rows) != len(selected):
            raise AssertionError(f"Configuration {key} does not cover every smoke sample")
        tavg_errors = np.asarray([float(row["relative_error_tavg"]) for row in rows])
        delta_errors = np.asarray([float(row["relative_error_delta_t"]) for row in rows])
        ranking.append(
            {
                "mapping_candidate": key[0],
                "torque_sign_multiplier": key[1],
                "average_formula": key[2],
                "delta_formula": key[3],
                "sample_count": len(rows),
                "mean_relative_error_tavg": float(np.mean(tavg_errors)),
                "median_relative_error_tavg": float(np.median(tavg_errors)),
                "max_relative_error_tavg": float(np.max(tavg_errors)),
                "mean_relative_error_delta_t": float(np.mean(delta_errors)),
                "median_relative_error_delta_t": float(np.median(delta_errors)),
                "max_relative_error_delta_t": float(np.max(delta_errors)),
                "joint_mean_relative_error": float(np.mean(tavg_errors) + np.mean(delta_errors)),
                "samples_meeting_both_tolerances": sum(
                    bool(row["tavg_within_0_5_percent"]) and bool(row["delta_within_2_percent"]) for row in rows
                ),
            }
        )
    ranking.sort(key=lambda row: (float(row["joint_mean_relative_error"]), float(row["max_relative_error_tavg"])))
    write_csv(ROOT / "global_configuration_ranking.csv", ranking)

    best = ranking[0]
    summary = {
        "comparison_policy": "one uniform configuration for all smoke-test samples",
        "smoke_sample_count": len(selected),
        "material_mapping_count": 3,
        "angle_solve_count": len(curves) * len(ANGLES_DEG),
        "best_global_configuration_by_joint_mean_relative_error": best,
        "thresholds": {"tavg_relative_error": 0.005, "delta_t_relative_error": 0.02},
        "conclusion": "not_reproduced_within_tolerance",
    }
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
