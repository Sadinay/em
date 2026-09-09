"""Use a periodic interpolation only to propose new FEMM phases, never as solver output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from analyze_phase360_search import OUTPUT, gather, norm_phase
from replay_fixed_current_5genes import save


def suggest(output: Path = OUTPUT) -> dict:
    data = gather(output)
    assert data["coarse_complete"]
    coarse = data["coarse_phases_degrees"]
    actual = {p["phase_deg"]: p for p in data["phase_results"]}
    candidates = data["candidates"]
    # Array dimensions: phase, gene, mechanical angle. Interpolate the sampled torque,
    # not ripple quotients with singular denominators.
    torque = np.array([[r["comparison_torques_nm"] for r in actual[p]["rows"]] for p in coarse])
    interpolation = CubicSpline([*coarse, 360], np.concatenate([torque, torque[:1]], axis=0), bc_type="periodic", axis=0)
    fine = np.arange(0, 360, .1)
    predicted_torque = interpolation(fine)
    tavg = predicted_torque.mean(axis=-1)
    ripple = np.ptp(predicted_torque, axis=-1) / np.maximum(np.abs(tavg), 1e-12)
    reference_tavg = np.array([c["historical_tavg_nm"] for c in candidates])
    reference_delta = np.array([c["historical_delta_t"] for c in candidates])
    t_err = 100 * np.abs(tavg / reference_tavg - 1)
    d_err = 100 * np.abs(ripple / reference_delta - 1)
    scores = {"tavg_mape": t_err.mean(axis=1), "joint_mape": np.concatenate([t_err,d_err],axis=1).mean(axis=1),
              "worst_error": np.concatenate([t_err,d_err],axis=1).max(axis=1)}
    best = {key: {"predicted_phase_deg": float(fine[np.argmin(values)]),
                  "predicted_error_percent": float(values.min())} for key, values in scores.items()}
    # Withheld already-solved +/-1 and +/-3 phases validate the interpolation locally.
    holdout = []
    for phase, point in actual.items():
        if phase in coarse:
            continue
        predicted = interpolation(phase)
        measured = np.array([r["comparison_torques_nm"] for r in point["rows"]])
        holdout.append({"phase_deg": phase, "max_torque_absolute_error_nm": float(np.max(np.abs(predicted-measured))),
                        "max_error_as_percent_of_reference_tavg": float(np.max(100*np.abs(predicted-measured)/reference_tavg[:,None]))})
    centers = sorted({norm_phase(round(v["predicted_phase_deg"])) for v in best.values()})
    proposed = sorted({norm_phase(center+offset) for center in centers for offset in range(-3,4)})
    missing = [p for p in proposed if p not in actual]
    result = {"purpose": "Interpolation selects candidate phases for real FEMM validation; predictions are not historical reproduction results.",
              "coarse_step_deg": 30, "prediction_grid_step_deg": .1,
              "predicted_optima": best, "withheld_phase_validation": holdout,
              "fine_centers_deg": centers, "fine_step_deg": 1,
              "fine_phases_including_existing": proposed, "new_phases_to_solve": missing,
              "new_FEMM_solves_required": 30*len(missing)}
    save(output / "fine_plan.json", result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    suggest()
