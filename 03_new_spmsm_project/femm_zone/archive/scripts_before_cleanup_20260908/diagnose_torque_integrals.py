"""Compare air-gap and block-integral torque using retained FEMM solutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validate_history_replay import ANGLES, DEFAULT_OUTPUT, atomic_json


GROUP_SETS = {
    "block22_rotor_iron_group1": (1,),
    "block22_design_group8": (8,),
    "block22_design_group9": (9,),
    "block22_design_groups8_9": (8, 9),
    "block22_rotor_and_design_groups1_8_9": (1, 8, 9),
}


def real_value(value) -> float:
    if isinstance(value, complex):
        if abs(value.imag) > 1e-9:
            raise ValueError(f"Unexpected complex DC result: {value!r}")
        value = value.real
    return float(value)


def extract_integrals(output_root: Path, selection: list[dict]) -> dict[str, dict[str, list[float]]]:
    import femm

    result = {name: {} for name in GROUP_SETS}
    result["gap_integral"] = {}
    opened = False
    try:
        femm.openfemm(1)
        opened = True
        for candidate in selection:
            candidate_id = candidate["candidate_id"]
            for name in result:
                result[name][candidate_id] = []
            for angle in ANGLES:
                angle_dir = output_root / candidate_id / f"angle_{angle:g}".replace(".", "p")
                model = angle_dir / "model.fem"
                stored = json.loads((angle_dir / "result.json").read_text(encoding="utf-8"))
                femm.opendocument(str(model.resolve()))
                femm.mi_loadsolution()
                result["gap_integral"][candidate_id].append(float(stored["torque_nm"]))
                for name, groups in GROUP_SETS.items():
                    femm.mo_clearblock()
                    for group in groups:
                        femm.mo_groupselectblock(group)
                    result[name][candidate_id].append(real_value(femm.mo_blockintegral(22)))
                femm.mo_close()
                femm.mi_close()
                print(f"extracted: {candidate_id}, angle={angle:g}", flush=True)
    finally:
        if opened:
            try:
                femm.closefemm()
            except Exception:
                pass
    return result


def average(values: np.ndarray, method: str) -> float:
    if method == "trapezoidal":
        return float(np.trapezoid(values, np.asarray(ANGLES)) / (ANGLES[-1] - ANGLES[0]))
    if method == "mean6":
        return float(np.mean(values))
    if method == "mean_first5":
        return float(np.mean(values[:-1]))
    raise ValueError(method)


def analyze(selection: list[dict], curves: dict[str, dict[str, list[float]]]) -> dict:
    trials = []
    for integral_name, by_candidate in curves.items():
        integral_trials = []
        for multiplier in (1, -1, 2, -2, 4, -4):
            for method in ("trapezoidal", "mean6", "mean_first5"):
                rows = []
                for candidate in selection:
                    candidate_id = candidate["candidate_id"]
                    values = multiplier * np.asarray(by_candidate[candidate_id])
                    historical_tavg = float(candidate["historical_tavg_nm"])
                    historical_delta = float(candidate["historical_delta_t_nm"])
                    tavg = average(values, method)
                    delta = float(np.ptp(values))
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "historical_tavg_nm": historical_tavg,
                            "recomputed_tavg_nm": tavg,
                            "relative_error_tavg": abs(tavg - historical_tavg) / abs(historical_tavg),
                            "historical_delta_t_nm": historical_delta,
                            "recomputed_delta_t_nm": delta,
                            "relative_error_delta_t": abs(delta - historical_delta) / abs(historical_delta),
                            "scaled_torques_nm": values.tolist(),
                        }
                    )
                integral_trials.append(
                    {
                        "multiplier": multiplier,
                        "tavg_method": method,
                        "mean_relative_error_tavg": float(np.mean([row["relative_error_tavg"] for row in rows])),
                        "mean_relative_error_delta_t": float(np.mean([row["relative_error_delta_t"] for row in rows])),
                        "mean_combined_relative_error": float(
                            np.mean([row["relative_error_tavg"] + row["relative_error_delta_t"] for row in rows])
                        ),
                        "candidates": rows,
                    }
                )
        integral_trials.sort(key=lambda row: row["mean_combined_relative_error"])
        trials.append(
            {
                "integral_name": integral_name,
                "best_trial": integral_trials[0],
                "all_trials": integral_trials,
            }
        )
    trials.sort(key=lambda row: row["best_trial"]["mean_combined_relative_error"])
    return {"status": "complete", "angles_mechanical_deg": list(ANGLES), "integral_results": trials}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selection = json.loads((args.output_root / "selected_candidates.json").read_text(encoding="utf-8"))["candidates"]
    curves = extract_integrals(args.output_root, selection)
    summary = analyze(selection, curves)
    target = args.output_root / "torque_integral_diagnostics" / "analysis_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, summary)
    print(json.dumps([{"integral": row["integral_name"], "best": row["best_trial"]} for row in summary["integral_results"]], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
